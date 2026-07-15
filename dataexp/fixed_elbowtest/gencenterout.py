"""
Generate 8 center-out reach .mot files for MoBL-ARMS upper limb model.
NO OpenSim API calls -- pure numpy/math only.

Rest posture (confirmed in OpenSim GUI):
  elv_angle=0, shoulder_elv=30, shoulder_rot=0, elbow_flexion=90

Each trial:
  - Hold at rest   (0.5s)
  - Reach to target (2.0s, minimum-jerk)
  - Hold at target  (0.5s)
  - Return to rest  (1.8s, minimum-jerk)
  Total: 4.8s at 240Hz = 1152 timepoints

The 8 directions are defined by varying shoulder_rot and elbow_flexion
from the rest posture. Each direction is a different combination that
moves the hand to a different azimuthal direction in the horizontal plane.
All values are within or near EF3D training distribution.

Coordinates (all in degrees, written as radians to .mot file):
  elv_angle    -- plane of elevation
  shoulder_elv -- shoulder elevation
  shoulder_rot -- shoulder internal/external rotation
  elbow_flexion -- elbow flexion/extension
  pro_sup      -- forearm pronation/supination (held at 0)
  deviation    -- wrist deviation (held at 0)
  flexion      -- wrist flexion (held at 0)
"""

import numpy as np
import os

OUTPUT_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp"

# --- Timing ---
SAMPLE_RATE  = 240
N_TOTAL      = 1152
DURATION     = N_TOTAL / SAMPLE_RATE  # 4.8s

N_HOLD_START = 120   # 0.5s
N_REACH      = 480   # 2.0s
N_HOLD_END   = 120   # 0.5s
N_RETURN     = 432   # 1.8s

assert N_HOLD_START + N_REACH + N_HOLD_END + N_RETURN == N_TOTAL

times = np.linspace(0, DURATION, N_TOTAL)

# --- Rest posture (confirmed in OpenSim) ---
REST = dict(
    elv_angle    = 0.0,
    shoulder_elv = 30.0,
    shoulder_rot = 0.0,
    elbow_flexion = 90.0,
    pro_sup      = 0.0,
    deviation    = 0.0,
    flexion      = 0.0,
)

# --- 8 reach targets ---
# Each target specifies the joint angles at the endpoint of the reach.
# elv_angle and shoulder_elv held near rest for horizontal plane movement.
# shoulder_rot and elbow_flexion varied to produce different reach directions.
# All values chosen to stay within EF3D training bounds:
#   elv_angle: 19-79, shoulder_elv: 39-99, shoulder_rot: -6-54, elbow_flexion: 45-130
#
# Directions named by clock position as seen from above the arm:
#   12 o'clock = straight forward (away from body)
#   3 o'clock  = to the right
#   etc.

TARGETS = [
    # name,          elv,  sh_elv, sh_rot, elbow
    ("12_forward",    19,    45,     0,     60),   # straight forward -- elbow extends
    ("1_fwd_right",   19,    39,    -6,     70),   # forward-right
    ("2_right_fwd",   19,    39,    -6,     85),   # right-forward
    ("3_right",       19,    39,     0,     95),   # to the right -- elbow flexes
    ("10_fwd_left",   19,    45,    25,     60),   # forward-left
    ("11_left_fwd",   19,    45,    40,     65),   # left-forward
    ("9_left",        19,    45,    54,     75),   # to the left
    ("6_backward",    19,    39,    20,    115),   # slightly backward/inward
]

def min_jerk(n_frames):
    """Minimum-jerk profile 0->1. Zero velocity at start and end."""
    t = np.linspace(0, 1, n_frames)
    return 10*t**3 - 15*t**4 + 6*t**5

def make_trajectory(rest_val, target_val):
    """Build a single coordinate trajectory: hold, reach, hold, return."""
    mj_out = min_jerk(N_REACH)
    mj_back = min_jerk(N_RETURN)
    return np.concatenate([
        np.full(N_HOLD_START, rest_val),
        rest_val + (target_val - rest_val) * mj_out,
        np.full(N_HOLD_END, target_val),
        target_val + (rest_val - target_val) * mj_back,
    ])

def write_mot(filepath, trajectories_deg):
    """
    Write a .mot file with 7 coordinate columns.
    trajectories_deg: dict with keys matching column_labels, values are (N,) arrays in degrees.
    Values are converted to radians before writing.
    """
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
        row = [times[i]] + [
            np.radians(trajectories_deg[c][i]) for c in column_labels
        ]
        lines.append("\t".join(f"{v:.10f}" for v in row) + "\n")

    with open(filepath, "w") as f:
        f.writelines(lines)

# --- Generate one .mot file per target ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Generating {len(TARGETS)} .mot files -> {OUTPUT_DIR}")
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

    # Sanity check: confirm movement actually happens
    elbow_std = traj["elbow_flexion"].std()
    sh_rot_std = traj["shoulder_rot"].std()

    outpath = os.path.join(OUTPUT_DIR, f"center_out_{name}.mot")
    write_mot(outpath, traj)

    print(f"  {name:>15}: elv={elv_t} sh_elv={sh_elv_t} sh_rot={sh_rot_t} elbow={elbow_t} "
          f"| elbow_std={elbow_std:.2f} sh_rot_std={sh_rot_std:.2f} -> {os.path.basename(outpath)}")

print()
print("Done. Load each .mot in OpenSim to verify visually before running Analyze.")
print("Checklist per file:")
print("  1. File -> Load Motion -> select .mot")
print("  2. Scrub timeline -- confirm arm moves from rest to target and back")
print("  3. Check arm stays in roughly horizontal plane throughout")
print("  4. If it looks right: Tools -> Analyze (Fiber Length + Kinematics, time 0-4.8)")