"""
Extract muscle fiber lengths, joint angles, and marker positions
for all center-out reach .mot files.

Uses equilibrateMuscles() per timepoint (lab's exact method from
extract_flag3d_data_utils.py: convert_to_muscle_lengths()).

Saves one .npz per direction in dataexp/centerout/.

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions
  conda activate proprioception
  python3 extractcenterout.py
"""

import opensim as osm
import numpy as np
import os
import glob

MODEL_PATH    = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
CENTEROUT_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout"

MUSCLE_NAMES = [
    'CORB',    'DELT1',   'DELT2',    'DELT3',  'INFSP',
    'LAT1',    'LAT2',    'LAT3',     'PECM1',  'PECM2',
    'PECM3',   'SUBSC',   'SUPSP',    'TMAJ',   'TMIN',
    'ANC',     'BIClong', 'BICshort', 'BRA',    'BRD',
    'ECRL',    'PT',      'TRIlat',   'TRIlong','TRImed'
]

# 4 coordinates driven for muscle equilibration
COORD_ORDER       = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion"]
# All 7 stored in joint_angles array for labels
COORD_LABEL_ORDER = ["elv_angle", "shoulder_elv", "shoulder_rot",
                     "elbow_flexion", "pro_sup", "deviation", "flexion"]

# Markers for trajectory visualization and end-effector labels
SHOULDER_MARKER = "R.Shoulder"
ELBOW_MARKER    = "R.Elbow.Lateral"
WRIST_MARKER    = "Handle"

# shoulder_to_world rotation (lab convention from get_shoulder_elbow_wrist_loc)
#   their X = -OpenSim Z (lateral)
#   their Y = -OpenSim X (anterior/posterior)
#   their Z =  OpenSim Y (vertical/up)
S2W = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

# --- Load model once ---
print("Loading model...")
model      = osm.Model(MODEL_PATH)
init_state = model.initSystem()
model.equilibrateMuscles(init_state)
muscle_set = model.getMuscles()
coord_set  = model.getCoordinateSet()
marker_set = model.getMarkerSet()
print(f"Model loaded. {muscle_set.getSize()} muscles, "
      f"{marker_set.getSize()} markers available.")

# Verify required markers exist
for mname in [SHOULDER_MARKER, ELBOW_MARKER, WRIST_MARKER]:
    try:
        marker_set.get(mname)
        print(f"  Marker found: {mname}")
    except Exception:
        raise KeyError(f"Marker '{mname}' not found in model")
print()

# --- Find all center-out .mot files ---
mot_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "center_out_*.mot")))
if not mot_files:
    raise FileNotFoundError(f"No center_out_*.mot files found in {CENTEROUT_DIR}")

print(f"Found {len(mot_files)} .mot files to process:")
for f in mot_files:
    print(f"  {os.path.basename(f)}")
print()

# --- Process each direction ---
for mot_path in mot_files:
    direction = os.path.basename(mot_path).replace("center_out_","").replace(".mot","")
    out_npz   = os.path.join(CENTEROUT_DIR, f"center_out_{direction}.npz")

    print(f"Processing: {direction}")

    motion      = osm.TimeSeriesTable(mot_path)
    coord_names = list(motion.getColumnLabels())
    times       = np.array(motion.getIndependentColumn())
    N           = motion.getNumRows()

    fiber_lengths = np.zeros((N, len(MUSCLE_NAMES)), dtype=np.float32)
    joint_angles  = np.zeros((N, 7),                 dtype=np.float32)
    # Marker positions in OpenSim ground frame (meters), pre-rotation
    shoulder_xyz  = np.zeros((N, 3),                 dtype=np.float32)
    elbow_xyz     = np.zeros((N, 3),                 dtype=np.float32)
    wrist_xyz     = np.zeros((N, 3),                 dtype=np.float32)

    for i in range(N):
        row = motion.getRowAtIndex(i)

        # Joint angles -> degrees (all 7, for labels)
        for j, cname in enumerate(COORD_LABEL_ORDER):
            joint_angles[i, j] = np.degrees(row[coord_names.index(cname)])

        # Set 4 driven coordinates
        for cname in COORD_ORDER:
            coord_set.get(cname).setValue(init_state, row[coord_names.index(cname)])

        # Solve fiber-tendon equilibrium at this pose
        model.equilibrateMuscles(init_state)

        # Fiber lengths in mm
        for k, mname in enumerate(MUSCLE_NAMES):
            fiber_lengths[i, k] = (
                muscle_set.get(mname).getFiberLength(init_state) * 1000
            )

        # Marker positions in OpenSim ground frame (meters)
        def get_marker(name):
            p = marker_set.get(name).getLocationInGround(init_state)
            return np.array([p.get(0), p.get(1), p.get(2)])

        shoulder_xyz[i] = get_marker(SHOULDER_MARKER)
        elbow_xyz[i]    = get_marker(ELBOW_MARKER)
        wrist_xyz[i]    = get_marker(WRIST_MARKER)

        if i % 300 == 0:
            bic   = fiber_lengths[i, MUSCLE_NAMES.index('BIClong')]
            elbow = joint_angles[i, 3]
            wrist_w = (S2W @ (wrist_xyz[i] - shoulder_xyz[i])) * 100
            print(f"  frame {i:4d}/{N} | elbow={elbow:.1f}deg | "
                  f"BIClong={bic:.2f}mm | "
                  f"Wrist XY=({wrist_w[0]:.1f},{wrist_w[1]:.1f})cm")

    # Sanity check
    bic_std = fiber_lengths[:, MUSCLE_NAMES.index('BIClong')].std()
    print(f"  BIClong std: {bic_std:.3f}mm  (should be > 0)")

    # Apply shoulder_to_world and center on shoulder, convert m -> cm
    # Store in their world frame so downstream code doesn't need to transform
    shoulder_w = (S2W @ shoulder_xyz.T).T * 100   # (N,3) their frame, cm
    elbow_w    = (S2W @ elbow_xyz.T).T    * 100
    wrist_w    = (S2W @ wrist_xyz.T).T    * 100

    wrist_centered  = wrist_w - shoulder_w   # shoulder at origin
    elbow_centered  = elbow_w - shoulder_w

    np.savez(out_npz,
             times=times,
             fiber_lengths=fiber_lengths,          # (N,25) mm
             joint_angles=joint_angles,            # (N,7) degrees
             wrist_xyz_world=wrist_centered,       # (N,3) cm, their frame, shoulder-centered
             elbow_xyz_world=elbow_centered,       # (N,3) cm, their frame, shoulder-centered
             coord_names=np.array(COORD_LABEL_ORDER),
             muscle_names=np.array(MUSCLE_NAMES))

    print(f"  Saved {os.path.basename(out_npz)}")
    reach_dist = np.linalg.norm(wrist_centered[-N//2] - wrist_centered[0])
    print(f"  Reach distance (approx): {reach_dist:.1f}cm")
    print()

print(f"All done. {len(mot_files)} directions extracted.")