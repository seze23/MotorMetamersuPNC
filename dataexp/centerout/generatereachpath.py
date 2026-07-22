"""
Generate desired XYZ end-effector positions for all 8
center-out directions across 1152 timepoints at 240Hz.

Source: real monkey reach trajectories from C_20170913_COactpas_TD.mat
- 4 primary directions (0, 90, 180, 270 deg) from actual monkey data
- 4 interpolated directions (45, 135, 225, 315 deg) averaged from adjacent

Monkey pos is 2D (XY horizontal plane) in cm. We set Z = 0.

Output: dataexp/centerout/desired_xyz_<direction>.npz
  - xyz: (1152, 3) desired wrist positions in monkey pos frame (cm), Z=0
  - times: (1152,) time vector in seconds
"""

import numpy as np
import os
import resampy
from scipy.io import loadmat

MAT_PATH      = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/monkeydata/s1-kinematics/reaching_experiments/C_20170913_COactpas_TD.mat"
CENTEROUT_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout"
os.makedirs(CENTEROUT_DIR, exist_ok=True)

SAMPLE_RATE = 240
N_TOTAL     = 1152
DURATION    = N_TOTAL / SAMPLE_RATE  # 4.8s

# Timing at 240Hz
N_HOLD_PRE   = int(1.65 * SAMPLE_RATE)   # 396
N_REACH      = int(0.50 * SAMPLE_RATE)   # 120
N_HOLD_MID   = int(0.50 * SAMPLE_RATE)   # 120
N_RETURN     = int(0.50 * SAMPLE_RATE)   # 120
N_HOLD_POST  = N_TOTAL - N_HOLD_PRE - N_REACH - N_HOLD_MID - N_RETURN  # 396
assert N_HOLD_PRE + N_REACH + N_HOLD_MID + N_RETURN + N_HOLD_POST == N_TOTAL

times_240 = np.linspace(0, DURATION, N_TOTAL)

# Monkey data at 100Hz
PRE_BINS  = 50   # bins before go cue = center hold
POST_BINS = 50   # bins after go cue = reach phase
MONKEY_HZ = 100
dt_monkey = 1.0 / MONKEY_HZ

print("Loading monkey data...")
mat = loadmat(MAT_PATH, simplify_cells=True)
td  = mat['trial_data']

start  = td['idx_startTime']
gocue  = td['idx_goCueTime']
end    = td['idx_endTime']
pos    = td['pos']      # (N, 2) cm, horizontal plane
tgtdir = td['tgtDir']

def extract_median_reach(direction_deg):
    """
    Extract median reach trajectory for a given direction.
    Returns (reach_xy, center_xy):
      reach_xy: (POST_BINS, 2) median reach trajectory in cm, relative to center
      center_xy: (2,) median center position in absolute coords
    """
    trials = np.where(tgtdir == direction_deg)[0]
    reach_trajs = []
    centers = []

    for i in trials:
        if i >= len(gocue): continue
        s = int(start[i])
        g = int(gocue[i])
        if np.any(np.isnan(pos[s:g+POST_BINS])): continue

        center = np.median(pos[s:g], axis=0)      # center position
        reach  = pos[g:g+POST_BINS] - center       # reach relative to center
        if reach.shape[0] < POST_BINS: continue

        reach_trajs.append(reach)
        centers.append(center)

    reach_trajs = np.array(reach_trajs)   # (n_trials, POST_BINS, 2)
    centers     = np.array(centers)       # (n_trials, 2)
    return np.median(reach_trajs, axis=0), np.median(centers, axis=0)

# Extract 4 primary directions
print("Extracting monkey reach trajectories...")
reach_data = {}
center_positions = []
for d in [0, 90, 180, 270]:
    reach_xy, center_xy = extract_median_reach(d)
    reach_data[d] = reach_xy
    center_positions.append(center_xy)
    peak = np.max(np.linalg.norm(reach_xy, axis=1))
    print(f"  {d:3d}deg: peak reach = {peak:.2f}cm  "
          f"endpoint = ({reach_xy[-1,0]:+.2f}, {reach_xy[-1,1]:+.2f})")

# Use median center across all directions (the common starting point)
center_xy = np.median(center_positions, axis=0)
print(f"\nCenter position (monkey workspace): ({center_xy[0]:.2f}, {center_xy[1]:.2f}) cm")

# Interpolate 4 diagonal directions by averaging adjacent pairs
reach_data[45]  = (reach_data[0]   + reach_data[90])  / 2.0
reach_data[135] = (reach_data[90]  + reach_data[180]) / 2.0
reach_data[225] = (reach_data[180] + reach_data[270]) / 2.0
reach_data[315] = (reach_data[270] + reach_data[0])   / 2.0

DIRECTION_NAMES = {
    0:   "0_right",
    45:  "45_fwd_right",
    90:  "90_forward",
    135: "135_fwd_left",
    180: "180_left",
    225: "225_back_left",
    270: "270_backward",
    315: "315_back_right",
}

def min_jerk(n):
    t = np.linspace(0, 1, n)
    return 10*t**3 - 15*t**4 + 6*t**5

print("\nGenerating 1152-timepoint XYZ trajectories...")
for deg, name in DIRECTION_NAMES.items():
    reach_xy = reach_data[deg]  # (POST_BINS, 2) at 100Hz

    # Resample reach from 100Hz to 240Hz over N_REACH frames
    # reach_xy is (POST_BINS, 2) at 100Hz, resample to N_REACH frames at 240Hz
    # resampy expects (channels, samples) so transpose, resample, transpose back
    reach_resampled = resampy.resample(
        reach_xy.T,          # (2, POST_BINS)
        MONKEY_HZ,           # source sample rate: 100Hz
        SAMPLE_RATE,         # target sample rate: 240Hz
        axis=1
    ).T                      # (resampled_frames, 2)

    # resampy output length = ceil(POST_BINS * 240/100) = ceil(50 * 2.4) = 120
    # which matches N_REACH = 120 exactly -- verify this:
    assert reach_resampled.shape[0] == N_REACH, \
    f"Expected {N_REACH} frames after resampling, got {reach_resampled.shape[0]}"

    # Target = endpoint of reach
    target_xy = reach_resampled[-1]

    # Return = time-reversed reach
    return_resampled = reach_resampled[::-1]

    # Build full XY trajectory (relative to center = origin)
    traj_xy = np.zeros((N_TOTAL, 2))
    traj_xy[:N_HOLD_PRE]  = 0.0              # hold at center
    traj_xy[N_HOLD_PRE:N_HOLD_PRE+N_REACH] = reach_resampled
    traj_xy[N_HOLD_PRE+N_REACH:N_HOLD_PRE+N_REACH+N_HOLD_MID] = target_xy
    traj_xy[N_HOLD_PRE+N_REACH+N_HOLD_MID:N_HOLD_PRE+N_REACH+N_HOLD_MID+N_RETURN] = return_resampled
    traj_xy[N_HOLD_PRE+N_REACH+N_HOLD_MID+N_RETURN:] = 0.0   # back at center

    # Convert to absolute monkey workspace coords (add center back)
    traj_xy_abs = traj_xy + center_xy[np.newaxis, :]

    # Full XYZ: monkey pos is horizontal (XY), Z=0 (planar task)
    xyz = np.zeros((N_TOTAL, 3), dtype=np.float32)
    xyz[:, 0] = traj_xy_abs[:, 0]   # horizontal X
    xyz[:, 1] = traj_xy_abs[:, 1]   # horizontal Y
    xyz[:, 2] = 0.0                  # Z = 0 (horizontal plane)

    reach_dist = np.linalg.norm(target_xy)
    print(f"  {name:<18}: reach dist={reach_dist:.2f}cm  "
          f"target=({target_xy[0]:+.2f},{target_xy[1]:+.2f})")

    out_path = os.path.join(CENTEROUT_DIR, f"desired_xyz_{name}.npz")
    np.savez(out_path,
             xyz=xyz,
             times=times_240,
             center_xy=center_xy,
             direction_deg=deg,
             direction_name=name)

print(f"\nSaved {len(DIRECTION_NAMES)} xyz trajectory files to {CENTEROUT_DIR}")
print("Next: run ik_centerout.py")