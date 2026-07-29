"""
Generate desired XYZ end-effector positions for 8
center-out directions at 10cm amplitude using minimum-jerk profiles.

Center position derived from FK at human KINARM rest posture:
  elv_angle=45, shoulder_elv=70, shoulder_rot=25, elbow_flexion=90
  -> wrist position in lab world frame: (41.3, -6.8, -2.9) cm
     (~shoulder height: KINARM horizontal-plane configuration)

The 8 reach targets are placed 10cm around this center in their XY plane
(their horizontal plane: X=lateral, Y=anterior/posterior).
Z is held constant.

Timing (1152 frames at 240Hz = 4.8s):
  Hold at center: 1.65s (396 frames)
  Reach out:      0.50s (120 frames) - minimum-jerk
  Hold at target: 0.50s (120 frames)
  Return:         0.50s (120 frames) - minimum-jerk reversed
  Hold at center: 1.65s (396 frames)

Output: dataexp/centerout/desired_xyz_<name>.npz
  xyz: (1152, 3) in lab world frame (cm), shoulder-centered
"""

import numpy as np
import os
import sys

from paths import REPO_DIR, CENTEROUT_DIR  # local path config
from fk_helper import get_shoulder_elbow_wrist_loc

SAMPLE_RATE = 240
N_TOTAL     = 1152
DURATION    = N_TOTAL / SAMPLE_RATE  # 4.8s

N_HOLD_PRE  = 396   # 1.65s
N_REACH     = 120   # 0.50s
N_HOLD_MID  = 120   # 0.50s
N_RETURN    = 120   # 0.50s
N_HOLD_POST = 396   # 1.65s
assert N_HOLD_PRE + N_REACH + N_HOLD_MID + N_RETURN + N_HOLD_POST == N_TOTAL

times = np.linspace(0, DURATION, N_TOTAL)

# Compute center from FK at KINARM horizontal-plane rest posture.
# Recentered 2026-07-29 off the EF3D floors (elv_angle>=19, shoulder_elv>=39)
# so reaches stay in the CNN training distribution; wrist sits ~shoulder height
# (Z ~ -3 cm), the true KINARM horizontal-plane configuration. Grounded in the
# Scott-lab center-out literature (shoulder ~30-35 deg plane, elbow ~75-90 deg;
# Cluff & Scott 2013, Maeda/Scott PMC6596259). See DRIFT_EXPLAINER.md.
REST = dict(elv_angle=45.0, shoulder_elv=70.0, shoulder_rot=25.0,
            elbow_flexion=90.0)

labels_rest = np.zeros((1, 7), dtype=np.float32)
labels_rest[0, 3] = REST['elv_angle']
labels_rest[0, 4] = REST['shoulder_elv']
labels_rest[0, 5] = REST['shoulder_rot']
labels_rest[0, 6] = REST['elbow_flexion']

_, _, wrist_rest = get_shoulder_elbow_wrist_loc(labels_rest)
wrist_rest = wrist_rest[0]  # (3,) X, Y, Z in cm, lab world frame

CENTER_XYZ = wrist_rest.copy()
CENTER_XY  = wrist_rest[:2]   # horizontal plane: X=lateral, Y=anterior

print(f"Rest posture: elv={REST['elv_angle']} sh_elv={REST['shoulder_elv']} "
      f"sh_rot={REST['shoulder_rot']} elbow={REST['elbow_flexion']}")
print(f"Center wrist position (lab world frame):")
print(f"  X={CENTER_XYZ[0]:.2f} cm (lateral)")
print(f"  Y={CENTER_XYZ[1]:.2f} cm (anterior/posterior)")
print(f"  Z={CENTER_XYZ[2]:.2f} cm (vertical)")
print()

REACH_CM = 10.0  # standard KINARM VGR reach amplitude

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
    """Minimum-jerk profile 0->1, zero velocity at endpoints."""
    t = np.linspace(0, 1, n)
    return 10*t**3 - 15*t**4 + 6*t**5

mj = min_jerk(N_REACH)

print(f"Generating {len(DIRECTION_NAMES)} XYZ trajectory files "
      f"({REACH_CM}cm reach, {N_TOTAL} frames at {SAMPLE_RATE}Hz)...")
print()

for deg, name in DIRECTION_NAMES.items():
    angle_rad = np.radians(deg)

    # Target in their XY plane, Z held at rest height
    target_xy  = CENTER_XY + REACH_CM * np.array([np.cos(angle_rad),
                                                    np.sin(angle_rad)])
    target_xyz = np.array([target_xy[0], target_xy[1], CENTER_XYZ[2]])

    # Build full 3D trajectory
    xyz = np.zeros((N_TOTAL, 3), dtype=np.float32)

    # Hold at center
    xyz[:N_HOLD_PRE] = CENTER_XYZ

    # Reach: center -> target via min-jerk
    for dim in range(3):
        xyz[N_HOLD_PRE:N_HOLD_PRE+N_REACH, dim] = (
            CENTER_XYZ[dim] + (target_xyz[dim] - CENTER_XYZ[dim]) * mj
        )

    # Hold at target
    xyz[N_HOLD_PRE+N_REACH:
        N_HOLD_PRE+N_REACH+N_HOLD_MID] = target_xyz

    # Return
    for dim in range(3):
        xyz[N_HOLD_PRE+N_REACH+N_HOLD_MID:
            N_HOLD_PRE+N_REACH+N_HOLD_MID+N_RETURN, dim] = (
            target_xyz[dim] + (CENTER_XYZ[dim] - target_xyz[dim]) * mj
        )

    # Hold at center
    xyz[N_HOLD_PRE+N_REACH+N_HOLD_MID+N_RETURN:] = CENTER_XYZ

    # Sanity checks
    assert np.allclose(xyz[0], xyz[-1], atol=1e-5), \
        "Start and end should match (hold at center)"
    assert np.allclose(xyz[0], CENTER_XYZ, atol=1e-5), \
        "Should start at center"
    reach_dist = np.linalg.norm(
        xyz[N_HOLD_PRE+N_REACH-1] - CENTER_XYZ
    )

    print(f"  {name:<18}: target XY=({target_xy[0]:+.1f},{target_xy[1]:+.1f}) cm  "
          f"Z={CENTER_XYZ[2]:.1f} cm  dist={reach_dist:.1f}cm")

    out_path = os.path.join(CENTEROUT_DIR, f"desired_xyz_{name}.npz")
    np.savez(out_path,
             xyz=xyz,                    # (1152, 3) lab world frame, cm
             times=times,
             center_xyz=CENTER_XYZ,
             target_xyz=target_xyz,
             rest_posture=np.array([REST['elv_angle'], REST['shoulder_elv'],
                                    REST['shoulder_rot'], REST['elbow_flexion']]),
             direction_deg=deg,
             direction_name=name)

print()
print(f"Saved {len(DIRECTION_NAMES)} files to {CENTEROUT_DIR}")
print("XYZ positions are in lab world frame (shoulder-centered, cm)")
print("Next: run ikcenterout.py")