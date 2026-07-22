"""
Inverse kinematics: convert desired XYZ wrist positions
to MoBL-ARMS joint angles at each timepoint.

Uses get_shoulder_elbow_wrist_loc() (lab's FK) inside the IK cost function
so the IK and labels use exactly the same coordinate convention.

Strong Z penalty (weight=10) prevents vertical drift during horizontal reaches.
Warm-starts from previous frame's solution for smooth joint angle trajectories.

Output: dataexp/centerout/ik_<direction>.npz
  - joint_angles: (1152, 4) degrees [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion]
  - times: (1152,)
"""

import os
import sys
import glob
import numpy as np
from scipy.optimize import minimize

REPO_DIR      = "/home/sydneyez/sydneyez/ProprioceptiveIllusions"
CENTEROUT_DIR = os.path.join(REPO_DIR, "dataexp/centerout")
sys.path.insert(0, REPO_DIR)

from utils.visualize_sample import get_shoulder_elbow_wrist_loc

# Rest posture
REST_ANGLES = np.array([20.0, 35.0, 25.0, 85.0])  # elv, sh_elv, sh_rot, elbow

# EF3D training bounds
BOUNDS = [(19, 79), (39, 99), (-6, 54), (45, 130)]

xyz_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "desired_xyz_*.npz")))
if not xyz_files:
    raise FileNotFoundError(
        f"No desired_xyz_*.npz files in {CENTEROUT_DIR} -- "
        "run generate_centerout_xyz.py first"
    )

print(f"Found {len(xyz_files)} direction files:")
for f in xyz_files:
    print(f"  {os.path.basename(f)}")
print()

def fk_wrist(angles):
    """Get wrist position using lab's FK function."""
    labels = np.zeros((1, 7), dtype=np.float32)
    labels[0, 3] = angles[0]  # elv_angle
    labels[0, 4] = angles[1]  # shoulder_elv
    labels[0, 5] = angles[2]  # shoulder_rot
    labels[0, 6] = angles[3]  # elbow_flexion
    _, _, wrist = get_shoulder_elbow_wrist_loc(labels)
    return wrist[0]  # (3,) X, Y, Z in cm

for xyz_path in xyz_files:
    name     = os.path.basename(xyz_path).replace("desired_xyz_","").replace(".npz","")
    out_path = os.path.join(CENTEROUT_DIR, f"ik_{name}.npz")

    print(f"Running IK: {name}")

    d           = np.load(xyz_path, allow_pickle=True)
    xyz_desired = d['xyz']         # (1152, 3) desired wrist in world frame, cm
    times       = d['times']
    center_xyz  = d['center_xyz']  # (3,) rest wrist position
    rest_z      = float(center_xyz[2])
    N           = len(times)

    joint_angles = np.zeros((N, 4), dtype=np.float32)
    prev         = REST_ANGLES.copy()

    # Verify FK at rest matches center
    w_rest = fk_wrist(REST_ANGLES)
    print(f"  FK at rest: ({w_rest[0]:.2f}, {w_rest[1]:.2f}, {w_rest[2]:.2f}) cm")
    print(f"  Center XYZ: ({center_xyz[0]:.2f}, {center_xyz[1]:.2f}, {center_xyz[2]:.2f}) cm")
    print(f"  Rest Z for penalty: {rest_z:.2f} cm")

    for i in range(N):
        target = xyz_desired[i]  # (3,) desired wrist position

        def cost(p, target=target, rest_z=rest_z, prev=prev):
            w      = fk_wrist(p)
            xy_err = (w[0] - target[0])**2 + (w[1] - target[1])**2
            z_pen  = 10.0 * (w[2] - rest_z)**2   # strong Z constraint
            reg    = 0.01 * np.sum((p - prev)**2) # smooth transition
            return xy_err + z_pen + reg

        # Multiple starts at peak reach frames to avoid local minima
        if i % N == 396 + 60:  # midpoint of reach phase
            starts = [prev,
                      REST_ANGLES,
                      np.array([19., 39., -6., 45.]),
                      np.array([79., 99., 54., 130.]),
                      np.array([45., 69., 24., 87.])]
            best, best_cost = None, np.inf
            for s0 in starts:
                r = minimize(cost, s0, bounds=BOUNDS, method='L-BFGS-B')
                if r.fun < best_cost:
                    best_cost = r.fun
                    best = r
            res = best
        else:
            res = minimize(cost, prev, bounds=BOUNDS, method='L-BFGS-B')

        joint_angles[i] = res.x
        prev = res.x

        if i % 200 == 0:
            w_sol  = fk_wrist(res.x)
            xy_err = np.sqrt((w_sol[0]-target[0])**2 + (w_sol[1]-target[1])**2)
            z_err  = abs(w_sol[2] - rest_z)
            print(f"  frame {i:4d}/{N} | "
                  f"target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f}) | "
                  f"actual=({w_sol[0]:.1f},{w_sol[1]:.1f},{w_sol[2]:.1f}) | "
                  f"XY_err={xy_err:.2f}cm Z_err={z_err:.2f}cm | "
                  f"elbow={res.x[3]:.1f}deg")

    # Summary at peak reach frame
    peak = 396 + 120
    w_peak    = fk_wrist(joint_angles[peak])
    w_center  = fk_wrist(joint_angles[396])
    delta     = w_peak - w_center
    reach_xy  = np.linalg.norm(delta[:2])
    z_drift   = abs(delta[2])
    print(f"  Peak reach: delta_XY={reach_xy:.2f}cm  Z_drift={z_drift:.2f}cm")
    print(f"  Joint ranges:")
    jnames = ['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion']
    for j, jn in enumerate(jnames):
        print(f"    {jn}: {joint_angles[:,j].min():.1f} -> {joint_angles[:,j].max():.1f}")

    np.savez(out_path,
             joint_angles=joint_angles,
             times=times,
             direction_name=name)
    print(f"  Saved {os.path.basename(out_path)}")
    print()

print("Next: run gencenterout.py")