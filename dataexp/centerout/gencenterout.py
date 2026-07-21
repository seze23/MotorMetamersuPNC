"""
Generate 8 center-out reach .mot files with EF3D-matched velocity.

Rest posture (confirmed in OpenSim):
  elv_angle=0, shoulder_elv=30, shoulder_rot=0, elbow_flexion=90

Timing (1152 timepoints at 240Hz = 4.8s):
  Hold at rest:   1.65s  (396 frames)
  Reach out:      0.50s  (120 frames) -- sinusoidal, ~150 deg/s peak
  Hold at target: 0.50s  (120 frames)
  Return:         0.50s  (120 frames) -- sinusoidal
  Hold at rest:   1.65s  (396 frames)
  Total:          4.80s  (1152 frames)

Peak angular velocity ~150 deg/s -- well within EF3D range (3-666 deg/s).
Sinusoidal profile (min-jerk) ensures zero velocity at movement boundaries.

No OpenSim API calls.
Output: dataexp/centerout/center_out_<name>.mot
"""

import numpy as np
import os

OUTPUT_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 240
N_TOTAL     = 1152
DURATION    = N_TOTAL / SAMPLE_RATE  # 4.8s

# Timing
N_HOLD_PRE   = 396   # 1.65s
N_REACH      = 120   # 0.50s
N_HOLD_MID   = 120   # 0.50s
N_RETURN     = 120   # 0.50s
N_HOLD_POST  = 396   # 1.65s

assert N_HOLD_PRE + N_REACH + N_HOLD_MID + N_RETURN + N_HOLD_POST == N_TOTAL

times = np.linspace(0, DURATION, N_TOTAL)

# Rest posture (confirmed in OpenSim GUI)
REST = dict(
    elv_angle     = 0.0,
    shoulder_elv  = 30.0,
    shoulder_rot  = 0.0,
    elbow_flexion = 90.0,
    pro_sup       = 0.0,
    deviation     = 0.0,
    flexion       = 0.0,
)

# 8 reach directions -- (name, elv, sh_elv, sh_rot, elbow) at target
# Same directions as before; velocity now properly matched to EF3D
TARGETS = [
    ("12_forward",   0,    45,    0,    60),
    ("1_fwd_right",  0,    39,   -6,    70),
    ("2_right_fwd",  0,    39,   -6,    85),
    ("3_right",      0,    39,    0,    95),
    ("10_fwd_left",  0,    45,   25,    60),
    ("11_left_fwd",  0,    45,   40,    65),
    ("9_left",       0,    45,   54,    75),
    ("6_backward",   0,    39,   20,   115),
]

def min_jerk(n):
    """Minimum-jerk profile 0->1 over n frames. Zero velocity at endpoints."""
    t = np.linspace(0, 1, n)
    return 10*t**3 - 15*t**4 + 6*t**5

def make_trajectory(rest_val, target_val):
    """
    Full 1152-frame trajectory for one coordinate:
    pre-hold -> sinusoidal reach -> mid-hold -> sinusoidal return -> post-hold
    """
    mj = min_jerk(N_REACH)
    return np.concatenate([
        np.full(N_HOLD_PRE,  rest_val),
        rest_val + (target_val - rest_val) * mj,
        np.full(N_HOLD_MID,  target_val),
        target_val + (rest_val - target_val) * mj,
        np.full(N_HOLD_POST, rest_val),
    ])

def peak_velocity_deg_per_s(rest_val, target_val):
    """
    Peak velocity of a min-jerk profile in deg/s.
    For min-jerk over duration T: peak vel = 1.875 * amplitude / T
    """
    amplitude = abs(target_val - rest_val)
    T = N_REACH / SAMPLE_RATE
    return 1.875 * amplitude / T

def write_mot(filepath, trajectories_deg):
    column_labels = ["elv_angle", "shoulder_elv", "shoulder_rot",
                     "elbow_flexion", "pro_sup", "deviation", "flexion"]
    lines = [
        "inDegrees=no\n",
        "DataType=double\n",
        "version=3\n",
        "OpenSimVersion=4.4-2022-10-11-798caa840\n",
        "endheader\n",
        "\t".join(["time"] + column_labels) + "\n",
    ]
    for i in range(N_TOTAL):
        row = [times[i]] + [np.radians(trajectories_deg[c][i]) for c in column_labels]
        lines.append("\t".join(f"{v:.10f}" for v in row) + "\n")
    with open(filepath, "w") as f:
        f.writelines(lines)

print(f"Generating {len(TARGETS)} .mot files -> {OUTPUT_DIR}")
print(f"Timing: {N_HOLD_PRE} hold + {N_REACH} reach + {N_HOLD_MID} hold + "
      f"{N_RETURN} return + {N_HOLD_POST} hold = {N_TOTAL} frames")
print()

for name, elv_t, sh_elv_t, sh_rot_t, elbow_t in TARGETS:
    traj = {
        "elv_angle":     make_trajectory(REST["elv_angle"],     elv_t),
        "shoulder_elv":  make_trajectory(REST["shoulder_elv"],  sh_elv_t),
        "shoulder_rot":  make_trajectory(REST["shoulder_rot"],  sh_rot_t),
        "elbow_flexion": make_trajectory(REST["elbow_flexion"], elbow_t),
        "pro_sup":       np.full(N_TOTAL, REST["pro_sup"]),
        "deviation":     np.full(N_TOTAL, REST["deviation"]),
        "flexion":       np.full(N_TOTAL, REST["flexion"]),
    }

    # Peak velocity for elbow (dominant DOF for most directions)
    elbow_peak_v = peak_velocity_deg_per_s(REST["elbow_flexion"], elbow_t)
    sh_elv_peak_v = peak_velocity_deg_per_s(REST["shoulder_elv"], sh_elv_t)

    outpath = os.path.join(OUTPUT_DIR, f"center_out_{name}.mot")
    write_mot(outpath, traj)

    print(f"  {name:>15}: elv={elv_t:>3} sh_elv={sh_elv_t:>3} "
          f"sh_rot={sh_rot_t:>3} elbow={elbow_t:>3} | "
          f"peak elbow vel={elbow_peak_v:.0f} deg/s -> {outpath.split('/')[-1]}")

print()
print("Done. Next steps:")
print("  1. Inspect each .mot in OpenSim GUI to verify directions visually")
print("  2. Run: python3 extractdata_centerout.py")
print("  3. Run: python3 compute_spindles_centerout.py")
print("  4. Run: python3 run_inference_centerout.py")