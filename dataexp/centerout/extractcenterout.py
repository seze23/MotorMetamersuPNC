"""
Extract muscle fiber lengths and joint angles for all center-out reach .mot files.
Uses equilibrateMuscles() per timepoint.

Iterates over all center_out_*.mot files in dataexp/centerout/
Saves one .npz per direction in the same directory.

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions
  conda activate proprioception
  python3 extractcenterout.py
"""

import opensim as osm
import numpy as np
import os
import glob

MODEL_PATH  = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
CENTEROUT_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout"

MUSCLE_NAMES = [
    'CORB',    'DELT1',   'DELT2',    'DELT3',  'INFSP',
    'LAT1',    'LAT2',    'LAT3',     'PECM1',  'PECM2',
    'PECM3',   'SUBSC',   'SUPSP',    'TMAJ',   'TMIN',
    'ANC',     'BIClong', 'BICshort', 'BRA',    'BRD',
    'ECRL',    'PT',      'TRIlat',   'TRIlong','TRImed'
]

COORD_ORDER       = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion"]
COORD_LABEL_ORDER = ["elv_angle", "shoulder_elv", "shoulder_rot",
                     "elbow_flexion", "pro_sup", "deviation", "flexion"]

# Load model once -- reuse across all directions
print("Loading model...")
model      = osm.Model(MODEL_PATH)
init_state = model.initSystem()
model.equilibrateMuscles(init_state)
muscle_set = model.getMuscles()
coord_set  = model.getCoordinateSet()
print(f"Model loaded. {muscle_set.getSize()} muscles available.")
print()

# Find all center-out mot files
mot_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "center_out_*.mot")))
if not mot_files:
    raise FileNotFoundError(f"No center_out_*.mot files found in {CENTEROUT_DIR}")
print(f"Found {len(mot_files)} .mot files to process:")
for f in mot_files:
    print(f"  {os.path.basename(f)}")
print()

for mot_path in mot_files:
    direction = os.path.basename(mot_path).replace("center_out_", "").replace(".mot", "")
    out_npz   = os.path.join(CENTEROUT_DIR, f"center_out_{direction}.npz")

    print(f"Processing: {direction}")

    motion      = osm.TimeSeriesTable(mot_path)
    coord_names = list(motion.getColumnLabels())
    times       = np.array(motion.getIndependentColumn())
    N           = motion.getNumRows()

    fiber_lengths = np.zeros((N, len(MUSCLE_NAMES)), dtype=np.float32)
    joint_angles  = np.zeros((N, 7), dtype=np.float32)

    for i in range(N):
        row = motion.getRowAtIndex(i)

        # Joint angles -> degrees
        for j, cname in enumerate(COORD_LABEL_ORDER):
            joint_angles[i, j] = np.degrees(row[coord_names.index(cname)])

        # Set 4 coordinates and solve equilibrium
        for cname in COORD_ORDER:
            coord_set.get(cname).setValue(init_state, row[coord_names.index(cname)])
        model.equilibrateMuscles(init_state)

        # Fiber lengths in mm
        for k, mname in enumerate(MUSCLE_NAMES):
            fiber_lengths[i, k] = muscle_set.get(mname).getFiberLength(init_state) * 1000

        if i % 300 == 0:
            bic   = fiber_lengths[i, MUSCLE_NAMES.index('BIClong')]
            elbow = joint_angles[i, 3]
            print(f"  frame {i:4d}/{N} | elbow={elbow:.1f}deg | BIClong={bic:.2f}mm")

    bic_std = fiber_lengths[:, MUSCLE_NAMES.index('BIClong')].std()
    print(f"  BIClong std: {bic_std:.3f}mm  (should be > 0)")

    np.savez(out_npz,
             times=times,
             fiber_lengths=fiber_lengths,
             joint_angles=joint_angles,
             coord_names=np.array(COORD_LABEL_ORDER),
             muscle_names=np.array(MUSCLE_NAMES))

    print(f"  Saved {os.path.basename(out_npz)}")
    print()

print(f"All done. {len(mot_files)} directions extracted.")