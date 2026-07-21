"""
Extract muscle fiber lengths using the exact method from the lab's own
extract_flag3d_data_utils.py: equilibrateMuscles() per timepoint,
then getFiberLength() * 1000 for mm output.

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

# Only 4 coordinates driven -- matches lab's convert_to_muscle_lengths exactly
COORD_ORDER = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion"]

model      = osm.Model(MODEL_PATH)
init_state = model.initSystem()
model.equilibrateMuscles(init_state)  # initial equilibrium -- same as lab code

muscle_set = model.getMuscles()
coord_set  = model.getCoordinateSet()

# Verify all 25 muscles exist
print("Verifying muscles...")
for mname in MUSCLE_NAMES:
    try:
        muscle_set.get(mname)
    except Exception:
        raise KeyError(f"Muscle '{mname}' not found")
print(f"All {len(MUSCLE_NAMES)} muscles found.")

motion      = osm.TimeSeriesTable(MOT_PATH)
coord_names = list(motion.getColumnLabels())
times       = np.array(motion.getIndependentColumn())
N           = motion.getNumRows()
print(f"Loaded motion: {N} timepoints over {times[-1]:.2f}s")

fiber_lengths = np.zeros((N, len(MUSCLE_NAMES)), dtype=np.float32)

print("Extracting fiber lengths (equilibrateMuscles per frame -- may take ~5 min)...")
for i in range(N):
    row = motion.getRowAtIndex(i)

    # Set only the 4 driven coordinates -- exact match to lab's approach
    for cname in COORD_ORDER:
        j = coord_names.index(cname)
        # lab converts degrees to radians: np.pi * (deg / 180)
        # our .mot file is already in radians (inDegrees=no), so use directly
        coord_set.get(cname).setValue(init_state, row[j])

    # This is the critical call -- solves fiber-tendon equilibrium at this pose
    model.equilibrateMuscles(init_state)

    for k, mname in enumerate(MUSCLE_NAMES):
        fiber_lengths[i, k] = muscle_set.get(mname).getFiberLength(init_state) * 1000  # m -> mm

    if i % 200 == 0:
        bic = fiber_lengths[i, MUSCLE_NAMES.index('BIClong')]
        tri = fiber_lengths[i, MUSCLE_NAMES.index('TRIlat')]
        elbow_rad = row[coord_names.index('elbow_flexion')]
        print(f"  Frame {i:4d}/{N} | elbow={np.degrees(elbow_rad):.1f}deg | "
              f"BIClong={bic:.2f}mm TRIlat={tri:.2f}mm")

print()
print("Sanity check:")
for mname in ['BIClong', 'BICshort', 'TRIlat', 'TRIlong']:
    idx  = MUSCLE_NAMES.index(mname)
    vals = fiber_lengths[:, idx]
    print(f"  {mname}: first={vals[0]:.3f}mm last={vals[-1]:.3f}mm std={vals.std():.4f}mm")

np.savez(OUTPUT_NPZ,
         times=times,
         fiber_lengths=fiber_lengths,   # (N, 25) mm, getFiberLength * 1000
         muscle_names=np.array(MUSCLE_NAMES))

print(f"\nSaved {OUTPUT_NPZ}")