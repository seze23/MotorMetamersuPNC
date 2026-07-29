"""
Generate desired_xyz_<name>.npz target files for the 24-direction x 3-radius
distortion-heatmap grid. Drop-in compatible with the existing glob-based
pipeline (ikcenterout.py, gencenterout.py, extractcenterout.py,
computefrcenterout.py, centeroutinference.py all discover files via
glob("desired_xyz_*.npz") / etc. and don't hardcode direction names), so no
other pipeline script needs modification.

Radii (4, 8, 12 cm) were chosen from the measured EF3D-bounded ceiling per
direction (dataexp/centerout/experiments/20260729_direction_radius_ceiling.csv,
global floor 12.68cm at 345deg) -- see EXPERIMENT_DESIGN_distortion_heatmap.md
section 3. Direction count (24, 15deg spacing) subsumes the original 8
canonical directions as every 3rd point -- see section 2 of the same doc.

Output naming: desired_xyz_g<angle_deg:03d>_r<radius_cm:02d>.npz
  e.g. desired_xyz_g045_r08.npz  (45 degrees, 8cm radius)
"""

import os
import numpy as np

from paths import CENTEROUT_DIR
from fk_helper import get_shoulder_elbow_wrist_loc

SAMPLE_RATE = 240
N_TOTAL = 1152
N_HOLD_PRE = 396
N_REACH = 120
N_HOLD_MID = 120
N_RETURN = 120
N_HOLD_POST = 396
assert N_HOLD_PRE + N_REACH + N_HOLD_MID + N_RETURN + N_HOLD_POST == N_TOTAL

times = np.linspace(0, N_TOTAL / SAMPLE_RATE, N_TOTAL)

REST = dict(elv_angle=45.0, shoulder_elv=70.0, shoulder_rot=25.0, elbow_flexion=90.0)
labels_rest = np.zeros((1, 7), dtype=np.float32)
labels_rest[0, 3:7] = [REST['elv_angle'], REST['shoulder_elv'], REST['shoulder_rot'], REST['elbow_flexion']]
_, _, wrist_rest = get_shoulder_elbow_wrist_loc(labels_rest)
CENTER_XYZ = wrist_rest[0]

N_DIRECTIONS = 24
RADII_CM = [4, 8, 12]  # from measured EF3D floor 12.68cm at 345deg


def min_jerk(n):
    t = np.linspace(0, 1, n)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


mj = min_jerk(N_REACH)
count = 0
for i in range(N_DIRECTIONS):
    deg = i * (360.0 / N_DIRECTIONS)
    angle_rad = np.radians(deg)
    for r in RADII_CM:
        name = f"g{round(deg):03d}_r{r:02d}"
        target_xy = CENTER_XYZ[:2] + r * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        target_xyz = np.array([target_xy[0], target_xy[1], CENTER_XYZ[2]])

        xyz = np.zeros((N_TOTAL, 3), dtype=np.float32)
        xyz[:N_HOLD_PRE] = CENTER_XYZ
        for dim in range(3):
            xyz[N_HOLD_PRE:N_HOLD_PRE + N_REACH, dim] = (
                CENTER_XYZ[dim] + (target_xyz[dim] - CENTER_XYZ[dim]) * mj)
        xyz[N_HOLD_PRE + N_REACH: N_HOLD_PRE + N_REACH + N_HOLD_MID] = target_xyz
        for dim in range(3):
            xyz[N_HOLD_PRE + N_REACH + N_HOLD_MID:
                N_HOLD_PRE + N_REACH + N_HOLD_MID + N_RETURN, dim] = (
                target_xyz[dim] + (CENTER_XYZ[dim] - target_xyz[dim]) * mj)
        xyz[N_HOLD_PRE + N_REACH + N_HOLD_MID + N_RETURN:] = CENTER_XYZ

        out_path = os.path.join(CENTEROUT_DIR, f"desired_xyz_{name}.npz")
        np.savez(out_path, xyz=xyz, times=times, center_xyz=CENTER_XYZ,
                 target_xyz=target_xyz, direction_deg=deg, radius_cm=r,
                 direction_name=name)
        count += 1

print(f"Saved {count} grid target files ({N_DIRECTIONS} directions x {len(RADII_CM)} radii) to {CENTEROUT_DIR}")
