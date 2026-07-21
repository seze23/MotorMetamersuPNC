"""
Extract muscle fiber lengths AND joint angles from the horizontal elbow
sweep motion file using the lab's exact method (equilibrateMuscles per
timepoint + getFiberLength). Joint angles are read directly from the
.mot file - no OpenSim Analyze Tool needed.

Overwrites:
  dataexp/elbow_fiber_lengths.npz  - fiber lengths (N,25) mm + joint angles (N,7) deg

Old TestFull_*.sto files in newelbowtest/ are no longer used.

Run on the cluster:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions
  conda activate proprioception
  python3 extractdata.py
"""

import opensim as osm
import numpy as np

MODEL_PATH = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
MOT_PATH   = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/elbow_sweep_horizontal_1152.mot"
OUTPUT_NPZ = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/elbow_fiber_lengths.npz"

MUSCLE_NAMES = [
    'CORB',    'DELT1',   'DELT2',    'DELT3',  'INFSP',
    'LAT1',    'LAT2',    'LAT3',     'PECM1',  'PECM2',
    'PECM3',   'SUBSC',   'SUPSP',    'TMAJ',   'TMIN',
    'ANC',     'BIClong', 'BICshort', 'BRA',    'BRD',
    'ECRL',    'PT',      'TRIlat',   'TRIlong','TRImed'
]

# 4 coordinates driven for muscle extraction (matches lab's convert_to_muscle_lengths)
COORD_ORDER = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion"]

# All 7 coordinates stored in labels (for FK + model input)
COORD_LABEL_ORDER = ["elv_angle", "shoulder_elv", "shoulder_rot",
                     "elbow_flexion", "pro_sup", "deviation", "flexion"]

# --- Load model ---
model      = osm.Model(MODEL_PATH)
init_state = model.initSystem()
model.equilibrateMuscles(init_state)

muscle_set = model.getMuscles()
coord_set  = model.getCoordinateSet()

print("Verifying muscles...")
for mname in MUSCLE_NAMES:
    try:
        muscle_set.get(mname)
    except Exception:
        raise KeyError(f"Muscle '{mname}' not found")
print(f"All {len(MUSCLE_NAMES)} muscles found.")

# --- Load motion ---
motion      = osm.TimeSeriesTable(MOT_PATH)
coord_names = list(motion.getColumnLabels())
times       = np.array(motion.getIndependentColumn())
N           = motion.getNumRows()
print(f"Loaded motion: {N} timepoints over {times[-1]:.2f}s")
print(f"Motion coordinates: {coord_names}")

# --- Storage arrays ---
fiber_lengths = np.zeros((N, len(MUSCLE_NAMES)), dtype=np.float32)
joint_angles  = np.zeros((N, 7), dtype=np.float32)  # degrees

# --- Extraction loop ---
print("Extracting fiber lengths (equilibrateMuscles per frame, ~5 min)...")
for i in range(N):
    row = motion.getRowAtIndex(i)

    # Read joint angles directly from .mot (radians) -> convert to degrees
    for j, cname in enumerate(COORD_LABEL_ORDER):
        joint_angles[i, j] = np.degrees(row[coord_names.index(cname)])

    # Set 4 driven coordinates and solve equilibrium
    for cname in COORD_ORDER:
        coord_set.get(cname).setValue(init_state, row[coord_names.index(cname)])
    model.equilibrateMuscles(init_state)

    # Extract fiber lengths (mm)
    for k, mname in enumerate(MUSCLE_NAMES):
        fiber_lengths[i, k] = muscle_set.get(mname).getFiberLength(init_state) * 1000

    if i % 200 == 0:
        bic      = fiber_lengths[i, MUSCLE_NAMES.index('BIClong')]
        tri      = fiber_lengths[i, MUSCLE_NAMES.index('TRIlat')]
        elbow_deg = joint_angles[i, 3]  # elbow_flexion in degrees
        print(f"  Frame {i:4d}/{N} | elbow={elbow_deg:.1f}deg | "
              f"BIClong={bic:.2f}mm TRIlat={tri:.2f}mm")

print()
print("Sanity check - fiber lengths:")
for mname in ['BIClong', 'BICshort', 'TRIlat', 'TRIlong']:
    idx  = MUSCLE_NAMES.index(mname)
    vals = fiber_lengths[:, idx]
    print(f"  {mname}: first={vals[0]:.3f}mm last={vals[-1]:.3f}mm std={vals.std():.4f}mm")

print()
print("Sanity check - joint angles (degrees):")
for j, cname in enumerate(COORD_LABEL_ORDER):
    vals = joint_angles[:, j]
    print(f"  {cname}: first={vals[0]:.2f} last={vals[-1]:.2f} "
          f"min={vals.min():.2f} max={vals.max():.2f}")

# --- Save ---
np.savez(OUTPUT_NPZ,
         times=times,
         fiber_lengths=fiber_lengths,          # (N, 25) mm
         joint_angles=joint_angles,            # (N, 7) degrees
         coord_names=np.array(COORD_LABEL_ORDER),
         muscle_names=np.array(MUSCLE_NAMES))

print(f"\nSaved {OUTPUT_NPZ}")
print("joint_angles columns:", COORD_LABEL_ORDER)
print("No .sto files needed everything is in this .npz")