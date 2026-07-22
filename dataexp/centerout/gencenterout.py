"""
Generate center-out reach .mot files using real monkey kinematic deltas
applied on top of a confirmed MoBL-ARMS rest posture.

Source: C_20170913_COactpas_TD.mat (Miller lab, Chowdhury et al. 2020)
Method (Path B): monkey reach DIRECTION and AMPLITUDE preserved as
  delta joint angles, applied on top of MoBL-ARMS rest posture so
  all trajectories remain within EF3D training distribution.

Rest posture (confirmed in OpenSim, centered in EF3D training bounds):
  elv_angle=30, shoulder_elv=35, shoulder_rot=24, elbow_flexion=87

4 primary directions from monkey data (0, 90, 180, 270 deg).
4 interpolated directions (45, 135, 225, 315) from averaging adjacent pairs.
Total: 8 directions.

Timing (1152 timepoints at 240Hz = 4.8s):
  Hold at rest:    1.65s (396 frames)
  Reach out:       0.30s (72 frames)  - resampled from monkey 100Hz to 240Hz
  Hold at target:  0.50s (120 frames)
  Return:          0.30s (72 frames)  - time-reversed reach
  Hold at rest:    2.05s (492 frames) - pad to 1152

Output: dataexp/centerout/center_out_<name>.mot
"""

import numpy as np
import os
from scipy.interpolate import interp1d

CENTEROUT_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout"
DELTA_DIR     = "/tmp"
os.makedirs(CENTEROUT_DIR, exist_ok=True)

SAMPLE_RATE = 240
N_TOTAL     = 1152

# Timing
N_HOLD_PRE  = 396
N_REACH     = 72    # 0.3s at 240Hz
N_HOLD_MID  = 120
N_RETURN    = 72
N_HOLD_POST = 492
assert N_HOLD_PRE + N_REACH + N_HOLD_MID + N_RETURN + N_HOLD_POST == N_TOTAL

times = np.linspace(0, N_TOTAL/SAMPLE_RATE, N_TOTAL)

# MoBL-ARMS rest posture
REST = {
    'elv_angle':     30.0,
    'shoulder_elv':  35.0,
    'shoulder_rot':  24.0,
    'elbow_flexion': 87.0,
    'pro_sup':        0.0,
    'deviation':      0.0,
    'flexion':        0.0,
}

# Monkey joint order -> MoBL-ARMS coordinate names
MONKEY_TO_MOBL = [
    'elv_angle',      # shoulder_adduction
    'shoulder_rot',   # shoulder_rotation
    'shoulder_elv',   # shoulder_flexion
    'elbow_flexion',  # elbow_flexion
    'pro_sup',        # radial_pronation
    'flexion',        # wrist_flexion
    'deviation',      # wrist_abduction
]

COORD_ORDER = ["elv_angle", "shoulder_elv", "shoulder_rot",
               "elbow_flexion", "pro_sup", "deviation", "flexion"]

# Load and resample monkey delta trajectories (300 bins at 100Hz -> N_REACH at 240Hz)
# The reach happens roughly bins 50-150 of the 300-bin window (go cue at bin 100)
# Extract the reach phase only (50 bins = 0.5s at 100Hz)

def load_reach_delta(direction_deg, reach_start_bin=100, reach_n_bins=50):
    """
    Load monkey delta trajectory, extract reach phase,
    resample to N_REACH frames at 240Hz.
    """
    delta = np.load(f"{DELTA_DIR}/delta_{direction_deg}deg.npy")  # (300, 7)
    # Extract reach phase: from go cue (bin 100) for 50 bins
    reach = delta[reach_start_bin:reach_start_bin + reach_n_bins, :]  # (50, 7)
    # Resample from 50 bins (0.5s at 100Hz) to N_REACH bins (0.3s at 240Hz)
    t_orig = np.linspace(0, 1, len(reach))
    t_new  = np.linspace(0, 1, N_REACH)
    resampled = np.zeros((N_REACH, 7))
    for j in range(7):
        f = interp1d(t_orig, reach[:, j], kind='cubic')
        resampled[:, j] = f(t_new)
    return resampled  # (N_REACH, 7) delta angles in monkey's convention

def interpolate_directions(delta_a, delta_b):
    """Average two adjacent monkey deltas to get intermediate direction."""
    return (delta_a + delta_b) / 2.0

def make_trajectory(rest_val, delta_vals):
    """
    Build full 1152-frame trajectory for one coordinate.
    rest_val: scalar rest position (degrees)
    delta_vals: (N_REACH,) array of delta angles during reach phase
    """
    target_seq = rest_val + delta_vals  # actual angle during reach
    return_seq = rest_val + delta_vals[::-1]  # time-reverse for return

    return np.concatenate([
        np.full(N_HOLD_PRE,  rest_val),
        target_seq,
        np.full(N_HOLD_MID,  rest_val + delta_vals[-1]),  # hold at target
        return_seq,
        np.full(N_HOLD_POST, rest_val),
    ])

def write_mot(filepath, trajectories_deg):
    lines = ["inDegrees=no\n","DataType=double\n","version=3\n",
             "OpenSimVersion=4.4-2022-10-11-798caa840\n","endheader\n",
             "\t".join(["time"]+COORD_ORDER)+"\n"]
    for i in range(N_TOTAL):
        row = [times[i]] + [np.radians(trajectories_deg[c][i]) for c in COORD_ORDER]
        lines.append("\t".join(f"{v:.10f}" for v in row)+"\n")
    with open(filepath,"w") as f:
        f.writelines(lines)

def check_bounds(traj_dict, name):
    """Check all joint angles stay within EF3D training bounds."""
    bounds = {
        'elv_angle':    (19, 79),
        'shoulder_elv': (39, 99),
        'shoulder_rot': (-6, 54),
        'elbow_flexion':(45,130),
    }
    warnings = []
    for coord, (lo, hi) in bounds.items():
        vals = traj_dict[coord]
        if vals.min() < lo or vals.max() > hi:
            warnings.append(f"  WARNING {coord}: {vals.min():.1f}-{vals.max():.1f} "
                           f"outside [{lo},{hi}]")
    if warnings:
        print(f"{name} bound violations:")
        for w in warnings: print(w)
    return len(warnings) == 0

# Load 4 primary monkey delta trajectories
print("Loading monkey delta trajectories...")
deltas = {}
for d in [0, 90, 180, 270]:
    deltas[d] = load_reach_delta(d)
    print(f"  {d}deg: elbow delta range "
          f"[{deltas[d][:,3].min():+.2f}, {deltas[d][:,3].max():+.2f}] deg")

# Interpolate 4 additional directions
deltas[45]  = interpolate_directions(deltas[0],   deltas[90])
deltas[135] = interpolate_directions(deltas[90],  deltas[180])
deltas[225] = interpolate_directions(deltas[180], deltas[270])
deltas[315] = interpolate_directions(deltas[270], deltas[0])

print()

# Direction names
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

print(f"Generating 8 center-out .mot files -> {CENTEROUT_DIR}")
print()

for deg, name in DIRECTION_NAMES.items():
    delta = deltas[deg]  # (N_REACH, 7)

    traj = {}
    for j, mobl_name in enumerate(MONKEY_TO_MOBL):
        rest_val = REST[mobl_name]
        traj[mobl_name] = make_trajectory(rest_val, delta[:, j])

    # Coordinates not in monkey data -- hold at rest
    for coord in COORD_ORDER:
        if coord not in traj:
            traj[coord] = np.full(N_TOTAL, REST.get(coord, 0.0))

    in_bounds = check_bounds(traj, name)

    outpath = os.path.join(CENTEROUT_DIR, f"center_out_{name}.mot")
    write_mot(outpath, traj)

    elbow_range = f"{traj['elbow_flexion'].min():.1f}->{traj['elbow_flexion'].max():.1f}"
    sh_elv_range = f"{traj['shoulder_elv'].min():.1f}->{traj['shoulder_elv'].max():.1f}"
    source = "monkey" if deg in [0,90,180,270] else "interpolated"
    print(f"  {name:<18} ({source:>12}): "
          f"elbow=[{elbow_range}] sh_elv=[{sh_elv_range}] "
          f"{'✓' if in_bounds else '⚠ OOB'}")

print()
print("Done. Run extractcenterout.py next.")