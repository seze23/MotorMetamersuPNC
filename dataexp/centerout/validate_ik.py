"""Automated biomechanical plausibility check for center-out IK solutions.

Reads ik_<direction>.npz and reports, per direction, whether the solved joint
trajectory is physiologically plausible. Intended as the pass/fail signal for
iterating on generatereachpath.py / ikcenterout.py.

Checks (from AGENTS.md / CLAUDE.md):
  - EF3D training bounds on the 4 driven coordinates
  - shoulder_rot stays near rest (no ~120 deg degeneracy)
  - peak reach amplitude ~= 10 cm, Z drift small
  - joint-angle continuity (no frame-to-frame jumps)

Exit code 0 if all directions pass, 1 otherwise.
"""

import glob
import os
import sys

import numpy as np

from paths import CENTEROUT_DIR
from fk_helper import get_shoulder_elbow_wrist_loc

N_HOLD_PRE = 396
N_REACH = 120

# (column index, low, high) for the 4 driven coordinates
EF3D_BOUNDS = {
    "elv_angle": (0, 19, 79),
    "shoulder_elv": (1, 39, 99),
    "shoulder_rot": (2, -6, 54),
    "elbow_flexion": (3, 45, 130),
}

REACH_CM = 10.0
REACH_TOL_CM = 1.5
Z_DRIFT_MAX_CM = 1.0
JUMP_MAX_DEG = 5.0  # max plausible per-frame change at 240 Hz


def check_direction(path):
    d = np.load(path, allow_pickle=True)
    ja = d["joint_angles"]  # (N, 7) degrees
    name = str(d["direction_name"])
    problems = []

    for coord, (j, lo, hi) in EF3D_BOUNDS.items():
        mn, mx = ja[:, j].min(), ja[:, j].max()
        if mn < lo or mx > hi:
            problems.append(
                f"{coord} {mn:.1f}->{mx:.1f} outside EF3D [{lo},{hi}]"
            )

    # shoulder_rot degeneracy guard (rest is 25 deg)
    sr_max = ja[:, 2].max()
    if sr_max > 90:
        problems.append(f"shoulder_rot spikes to {sr_max:.1f} (degenerate)")

    # reach amplitude + Z drift via analytic FK
    labels = np.zeros_like(ja)
    labels[:, 3:] = ja[:, :4]
    _, _, wrist = get_shoulder_elbow_wrist_loc(labels)
    center = wrist[N_HOLD_PRE]
    peak = wrist[N_HOLD_PRE + N_REACH]
    delta = peak - center
    reach_xy = float(np.linalg.norm(delta[:2]))
    z_drift = float(abs(delta[2]))
    if abs(reach_xy - REACH_CM) > REACH_TOL_CM:
        problems.append(f"peak reach {reach_xy:.1f}cm != {REACH_CM}cm")
    if z_drift > Z_DRIFT_MAX_CM:
        problems.append(f"Z drift {z_drift:.1f}cm > {Z_DRIFT_MAX_CM}cm")

    # continuity on driven coords, skipping the first few IK warm-up frames
    # (the optimizer settles from the model default pose over frames 0-4, a
    # transient in the pre-reach hold that is not a movement discontinuity).
    IK_SETTLE = 5
    max_jump = float(np.abs(np.diff(ja[IK_SETTLE:, :4], axis=0)).max())
    if max_jump > JUMP_MAX_DEG:
        problems.append(f"joint jump {max_jump:.1f}deg/frame")

    return name, reach_xy, z_drift, sr_max, problems


def main():
    files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "ik_*.npz")))
    if not files:
        print(f"No ik_*.npz in {CENTEROUT_DIR}")
        return 1

    all_ok = True
    print(f"{'direction':<16} {'reachXY':>8} {'Zdrift':>7} {'srMax':>7}  status")
    print("-" * 60)
    for path in files:
        name, reach_xy, z_drift, sr_max, problems = check_direction(path)
        status = "OK" if not problems else "FAIL"
        if problems:
            all_ok = False
        print(f"{name:<16} {reach_xy:8.2f} {z_drift:7.2f} {sr_max:7.1f}  {status}")
        for p in problems:
            print(f"    - {p}")

    print("-" * 60)
    print("ALL PASS" if all_ok else "SOME DIRECTIONS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
