import opensim as osm
import numpy as np
import pandas as pd

MODEL_PATH = "/Users/sydeze/Downloads/MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
MOT_PATH = "/Users/sydeze/Downloads/elbow_sweep_horizontal_1152.mot"
OUTPUT_NPZ = "/Users/sydeze/Downloads/elbow_fiber_lengths.npz"

MUSCLE_NAMES = [
    'CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP',
    'LAT1', 'LAT2', 'LAT3', 'PECM1', 'PECM2',
    'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN',
    'ANC', 'BIClong', 'BICshort', 'BRA', 'BRD',
    'ECRL', 'PT', 'TRIlat', 'TRIlong', 'TRImed'
]

model = osm.Model(MODEL_PATH)
state = model.initSystem()
coord_set = model.getCoordinateSet()
muscle_set = model.getMuscles()

motion = osm.TimeSeriesTable(MOT_PATH)
coord_names = motion.getColumnLabels()
times = np.array(motion.getIndependentColumn())
N = motion.getNumRows()

fiber_lengths = np.zeros((N, len(MUSCLE_NAMES)), dtype=np.float32)

for i in range(N):
    row = motion.getRowAtIndex(i)
    for j, cname in enumerate(coord_names):
        coord_set.get(cname).setValue(state, row[j])
    model.realizeDynamics(state)  # key: realizeDynamics not realizePosition

    for k, mname in enumerate(MUSCLE_NAMES):
        muscle = muscle_set.get(mname)
        fiber_lengths[i, k] = muscle.getFiberLength(state)

    if i % 200 == 0:
        print(f"Frame {i}/{N}, BIClong={fiber_lengths[i, MUSCLE_NAMES.index('BIClong')]*1000:.2f}mm")

np.savez(OUTPUT_NPZ, times=times, fiber_lengths=fiber_lengths,
         muscle_names=np.array(MUSCLE_NAMES))
print(f"Saved {OUTPUT_NPZ}")
print("BIClong std (mm):", fiber_lengths[:, MUSCLE_NAMES.index('BIClong')].std() * 1000)