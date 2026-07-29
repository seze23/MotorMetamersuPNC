"""
Script 2 of 3: Inverse kinematics in analytic-FK space.

Researcher decision (2026-07-29): the CNN's ground-truth labels are built from
the analytic 2-segment forward-kinematics model (fk_helper), so the reach
targets are defined in that SAME space and this stage solves the analytic FK
directly. This guarantees zero divergence between targets and labels: each 10 cm
in-plane target is hit exactly, so there is no overshoot and no vertical drift.

Why not OpenSim marker-IK here (as CLAUDE.md's "IK solver: OpenSim IK" note
prefers): the analytic-FK wrist and the MoBL-ARMS `Handle` marker do not
coincide (~44 cm apart in direction, configuration-dependent). Tracking the
Handle therefore lands the analytic wrist off-target, worst toward forward-left
(see DRIFT_EXPLAINER.md). Driving OpenSim IK into analytic-FK space with an
outer Handle-correction loop was tried and DIVERGES numerically (the
Handle->analytic-wrist Jacobian is ill-conditioned; error grows even damped).
Analytic IK is the stable way to satisfy the chosen 0-divergence requirement.
OpenSim is still used downstream (extractdata_centerout) for muscle fiber
lengths from these joint angles.

Solve: for each frame find [elv_angle, shoulder_elv, elbow_flexion] (with
shoulder_rot fixed at its rest value, matching the previous soft IK task) such
that the analytic-FK wrist equals the desired world-frame target. Bounded to the
EF3D training distribution so solutions stay in the CNN's trained domain.

Output: dataexp/centerout/ik_<direction>.npz
  - joint_angles: (1152, 7) degrees -- all coordinates (shoulder_rot=rest,
    pro_sup/deviation/flexion=0)
  - times: (1152,)
"""

import os
import glob

import numpy as np
from scipy.optimize import least_squares

from paths import CENTEROUT_DIR
from fk_helper import get_shoulder_elbow_wrist_loc

N_TOTAL = 1152
N_HOLD_PRE = 396
N_REACH = 120

# Rest posture -- must match generatereachpath.py.
REST = {
    'elv_angle':     45.0,
    'shoulder_elv':  70.0,
    'shoulder_rot':  25.0,
    'elbow_flexion': 90.0,
}

# EF3D training-distribution bounds on the 3 solved coordinates (shoulder_rot is
# held fixed at rest). Solutions are constrained here so joints stay in the
# CNN's trained domain.
EF3D = {
    'elv_angle':     (19.0, 79.0),
    'shoulder_elv':  (39.0, 99.0),
    'elbow_flexion': (45.0, 130.0),
}

# All 7 MoBL-ARMS coordinates in order.
COORD_NAMES = ["elv_angle", "shoulder_elv", "shoulder_rot",
               "elbow_flexion", "pro_sup", "deviation", "flexion"]


def analytic_wrist(elv, shelv, rot, elbow):
    """Analytic-FK wrist position (3,) world frame, cm, for one posture (deg)."""
    lab = np.zeros((1, 7), dtype=np.float64)
    lab[0, 3], lab[0, 4], lab[0, 5], lab[0, 6] = elv, shelv, rot, elbow
    _, _, wrist = get_shoulder_elbow_wrist_loc(lab)
    return wrist[0]


def solve_frame(target_xyz, seed):
    """Solve [elv, shelv, elbow] so analytic wrist hits target_xyz (rot fixed)."""
    rot = REST['shoulder_rot']
    lo = np.array([EF3D['elv_angle'][0], EF3D['shoulder_elv'][0],
                   EF3D['elbow_flexion'][0]])
    hi = np.array([EF3D['elv_angle'][1], EF3D['shoulder_elv'][1],
                   EF3D['elbow_flexion'][1]])

    def residual(q):
        return analytic_wrist(q[0], q[1], rot, q[2]) - target_xyz

    sol = least_squares(residual, seed, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
    return sol.x, float(np.linalg.norm(sol.fun))


xyz_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "desired_xyz_*.npz")))
if not xyz_files:
    raise FileNotFoundError(f"No desired_xyz_*.npz in {CENTEROUT_DIR}")

print(f"Found {len(xyz_files)} directions. Solving analytic IK "
      f"(shoulder_rot fixed at {REST['shoulder_rot']} deg)...\n")

for xyz_path in xyz_files:
    name    = os.path.basename(xyz_path).replace("desired_xyz_", "").replace(".npz", "")
    out_npz = os.path.join(CENTEROUT_DIR, f"ik_{name}.npz")
    print(f"Solving: {name}")

    d         = np.load(xyz_path, allow_pickle=True)
    xyz_world = d['xyz'].astype(np.float64)     # (1152, 3) world frame, cm
    times     = d['times'].astype(np.float64)

    joint_angles = np.zeros((N_TOTAL, 7), dtype=np.float32)
    joint_angles[:, 2] = REST['shoulder_rot']   # shoulder_rot fixed

    seed = np.array([REST['elv_angle'], REST['shoulder_elv'],
                     REST['elbow_flexion']])
    worst_res = 0.0
    for i in range(N_TOTAL):
        q, res = solve_frame(xyz_world[i], seed)
        joint_angles[i, 0] = q[0]   # elv_angle
        joint_angles[i, 1] = q[1]   # shoulder_elv
        joint_angles[i, 3] = q[2]   # elbow_flexion
        seed = q                     # warm-start next frame for continuity
        worst_res = max(worst_res, res)

    print(f"  Joint angle ranges:")
    for j, cname in enumerate(COORD_NAMES[:4]):
        print(f"    {cname}: {joint_angles[:,j].min():.1f} -> "
              f"{joint_angles[:,j].max():.1f} deg")

    # Verify reach against the analytic FK (should be exact by construction).
    lab = np.zeros((N_TOTAL, 7), dtype=np.float32)
    lab[:, 3:] = joint_angles[:, :4]
    _, _, wrists = get_shoulder_elbow_wrist_loc(lab)
    delta    = wrists[N_HOLD_PRE + N_REACH] - wrists[N_HOLD_PRE]
    reach_xy = np.linalg.norm(delta[:2])
    z_drift  = abs(delta[2])
    print(f"  Peak reach: XY={reach_xy:.2f}cm  Z_drift={z_drift:.2f}cm  "
          f"max solve residual={worst_res:.4f}cm")

    np.savez(out_npz,
             joint_angles=joint_angles,   # (N, 7) degrees
             times=times,
             direction_name=name)
    print(f"  Saved {os.path.basename(out_npz)}\n")

print("All done. Next: run gencenterout.py")
