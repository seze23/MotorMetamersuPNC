"""
Convert desired XYZ wrist positions to MoBL-ARMS joint angles at 
each timepoint.

Uses convert_to_joint_angles() function from
data_generation/extract_flag3d_data_utils.py as the primary IK solver.

The monkey pos frame (X, Y horizontal) needs to be mapped to the
lab's world frame before passing to their IK:
  their world frame: X=lateral, Y=anterior, Z=vertical (up)
  monkey pos frame: col0=X (rightward), col1=Y (forward/backward)

The IK outputs 4 MoBL-ARMS joint angles:
  [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion]

Output: dataexp/centerout/ik_<direction>.npz
  - joint_angles: (1152, 4) joint angles in degrees
  - times: (1152,)
"""

import numpy as np
import os
import glob
import sys

REPO_DIR      = "/home/sydneyez/sydneyez/ProprioceptiveIllusions"
CENTEROUT_DIR = os.path.join(REPO_DIR, "dataexp/centerout")
sys.path.insert(0, REPO_DIR)

from data_generation.extract_flag3d_data_utils import convert_to_joint_angles

# Used as initial guess and reference for IK
REST_ANGLES = np.array([19.0, 39.0, 9.9, 84.3]) # elv, sh_elv, sh_rot, elbow

# The monkey workspace center in monkey pos coords
# (from generate_centerout_xyz.py output)
# The IK needs positions in the lab's shoulder-centered world frame

# Mapping from monkey pos frame to lab's world frame:
# monkey col0 (rightward +X) lab X (lateral)
# monkey col1 (forward +Y) lab Y (anterior/posterior)
# Z = 0 (horizontal plane) lab Z = height of arm above ground
# We need to set a reasonable arm height (Z in lab frame = Y in OpenSim)
# From the rest posture FK: wrist Y (their frame) ~ -16.5cm = arm height
ARM_HEIGHT_CM = -16.5   # their Y at rest = wrist height, stays constant (horizontal reach)

xyz_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "desired_xyz_*.npz")))
if not xyz_files:
    raise FileNotFoundError(
        f"No desired_xyz_*.npz files in {CENTEROUT_DIR}"
    )

print(f"Found {len(xyz_files)} direction files to process:")
for f in xyz_files:
    print(f"  {os.path.basename(f)}")
print()

for xyz_path in xyz_files:
    name = os.path.basename(xyz_path).replace("desired_xyz_","").replace(".npz","")
    out_path = os.path.join(CENTEROUT_DIR, f"ik_{name}.npz")

    print(f"Running IK: {name}")

    d    = np.load(xyz_path, allow_pickle=True)
    xyz  = d['xyz']           # (1152, 3) monkey pos frame, cm
    times = d['times']
    N    = len(times)

    # Convert monkey pos frame to lab shoulder-centered world frame
    # monkey: (X_right, Y_forward, Z=0)
    # lab world: (X_lateral, Y_anterior, Z_vertical)
    # Mapping: lab_X = monkey_X, lab_Y = monkey_Y, lab_Z = ARM_HEIGHT_CM
    # Then center on shoulder: subtract shoulder position in lab frame
    # (shoulder is at origin in their shoulder-centered frame)
    # We need to add back the shoulder offset to get absolute lab coords
    # From FK: rest wrist lab XY = (26.0, -16.5), shoulder at (0, 0)
    # Monkey center pos ~ (-1.5, -33.1) absolute -> maps to wrist at rest

    # The convert_to_joint_angles function expects:
    # shoulder (3,T), elbow (3,T), hand (3,T) in their world frame, shoulder-centered

    shoulder = np.zeros((N, 3))  # (N, 3) shoulder at origin

    # Hand position in lab world frame, shoulder-centered
    # monkey X (rightward) -> lab X (lateral)
    # monkey Y (forward) -> lab Y (anterior/posterior, note sign)
    # ARM_HEIGHT_CM -> lab Z (vertical)
    hand = np.zeros((N, 3))
    hand[:, 0] = xyz[:, 0]       # lateral (same direction)
    hand[:, 1] = xyz[:, 1]       # anterior (same direction)
    hand[:, 2] = ARM_HEIGHT_CM   # vertical

    # Estimate elbow position using simple geometry
    # Elbow lies along upper arm, roughly halfway between shoulder and hand
    # More accurate: use fixed upper arm direction from rest posture
    # For IK input, elbow estimate just needs to be in the right ballpark
    UPPER_ARM = 33.0  # cm

    # Simple elbow estimate: project along shoulder-to-hand direction, UPPER_ARM length
    hand_dist = np.linalg.norm(hand, axis=1, keepdims=True)
    hand_dir  = hand / np.maximum(hand_dist, 1e-6)
    elbow     = hand_dir * UPPER_ARM  # (N, 3) rough elbow position

    # Stack for convert_to_joint_angles: expects (3, T) arrays
    shoulder_T = shoulder.T    # (3, N)
    elbow_T    = elbow.T       # (3, N)
    hand_T     = hand.T        # (3, N)

    print(f" convert_to_joint_angles on {N} timepoints...")
    try:
        joint_coords, _, _ = convert_to_joint_angles(
            np.stack([shoulder_T, elbow_T, hand_T], axis=0)  # (3, 3, N)
        )
        # joint_coords: (4, N) -> [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion]
        joint_angles_deg = joint_coords.T  # (N, 4) degrees
        print(f" Joint angle ranges:")
        jnames = ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion']
        for j, jn in enumerate(jnames):
            print(f"    {jn}: {joint_angles_deg[:,j].min():.1f}"
                  f"{joint_angles_deg[:,j].max():.1f} deg")

    except Exception as e:
        print(f"  convert_to_joint_angles failed: {e}")
        print("  Falling back to numerical IK")

        from scipy.optimize import minimize
        import numpy as np_

        def roty(a):
            a = np.radians(a)
            return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        def rotz(a):
            a = np.radians(a)
            return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        def rotx(a):
            a = np.radians(a)
            return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        s2w = np.array([[0,0,-1],[-1,0,0],[0,1,0]])
        FOREARM = 26.0

        def fk_wrist(elv, sh_elv, sh_rot, elbow):
            def sr(ea,se,sr_): return roty(ea).dot(rotz(se)).dot(roty(-ea)).dot(roty(sr_))
            ev = np.array([0,-UPPER_ARM,0])
            hv = np.array([0,-FOREARM,0])
            Rs = sr(elv,sh_elv,sh_rot)
            return s2w.dot(Rs.dot(rotx(elbow).dot(hv)+ev))

        joint_angles_deg = np.zeros((N, 4), dtype=np.float32)
        prev = REST_ANGLES.copy()

        for i in range(N):
            target = hand[i]  # (3,) in lab world frame

            def cost(p):
                w = fk_wrist(*p)
                return np.sum((w - target)**2) + 0.01*np.sum((p - prev)**2)

            res = minimize(cost, prev,
                          bounds=[(19,79),(39,99),(-6,54),(45,130)],
                          method='L-BFGS-B')
            joint_angles_deg[i] = res.x
            prev = res.x

            if i % 200 == 0:
                w_sol = fk_wrist(*res.x)
                err   = np.linalg.norm(w_sol - target)
                print(f"  frame {i:4d}/{N} | err={err:.2f}cm | "
                      f"elbow={res.x[3]:.1f}deg")

    np.savez(out_path,
             joint_angles=joint_angles_deg,   # (N, 4) degrees
             times=times,
             direction_name=name)
    print(f"  Saved {os.path.basename(out_path)}")
    print()

print("All done. Next: run generatecenterout.py")