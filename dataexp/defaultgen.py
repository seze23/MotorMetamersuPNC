import opensim as osm
import numpy as np
from scipy.signal import medfilt, savgol_filter

MODEL_PATH = "/home/sydneyez/MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"

# Fixed posture 
SHOULDER_ELV_DEG = 70
SHOULDER_ROT_DEG = 20
ELBOW_START_DEG  = 45
ELBOW_END_DEG    = 90

# Other coordinates held at neutral default 
ELV_ANGLE_DEG = 0
PRO_SUP_DEG   = 0
DEVIATION_DEG = 0
FLEXION_DEG   = 0

N_TIMEPOINTS = 67    # ~66.7 Hz over 1 second, matching Ef3D sampling rate
DURATION = 1.0
times = np.linspace(0, DURATION, N_TIMEPOINTS)

elbow_flexion_deg = ELBOW_START_DEG + (times / DURATION) * (ELBOW_END_DEG - ELBOW_START_DEG)

# Load model
model = osm.Model(MODEL_PATH)
state = model.initSystem()
coord_set = model.getCoordinateSet()

# Confirm these against your model's actual body names
print("Available bodies:", [model.getBodySet().get(i).getName()
                              for i in range(model.getBodySet().getSize())])
SHOULDER_BODY, ELBOW_BODY, WRIST_BODY = "humerus", "ulna", "hand"

def get_pos(body_name, state):
    body = model.getBodySet().get(body_name)
    p = body.getPositionInGround(state)
    return np.array([p.get(0), p.get(1), p.get(2)])

shoulder_xyz, elbow_xyz, wrist_xyz = [], [], []

for t_idx in range(N_TIMEPOINTS):
    coord_set.get("elv_angle").setValue(state, np.radians(ELV_ANGLE_DEG))
    coord_set.get("shoulder_elv").setValue(state, np.radians(SHOULDER_ELV_DEG))
    coord_set.get("shoulder_rot").setValue(state, np.radians(SHOULDER_ROT_DEG))
    coord_set.get("elbow_flexion").setValue(state, np.radians(elbow_flexion_deg[t_idx]))
    coord_set.get("pro_sup").setValue(state, np.radians(PRO_SUP_DEG))
    coord_set.get("deviation").setValue(state, np.radians(DEVIATION_DEG))
    coord_set.get("flexion").setValue(state, np.radians(FLEXION_DEG))
    model.realizePosition(state)

    shoulder_xyz.append(get_pos(SHOULDER_BODY, state))
    elbow_xyz.append(get_pos(ELBOW_BODY, state))
    wrist_xyz.append(get_pos(WRIST_BODY, state))

shoulder_xyz = np.array(shoulder_xyz)
elbow_xyz    = np.array(elbow_xyz)
wrist_xyz    = np.array(wrist_xyz)

# Center relative to shoulder, matching Ef3D convention
elbow_xyz_centered = elbow_xyz - shoulder_xyz
wrist_xyz_centered = wrist_xyz - shoulder_xyz

# Save results
np.savez("elbow_sweep_45to90_1s.npz",
         times=times,
         elbow_flexion_deg=elbow_flexion_deg,
         shoulder_xyz=shoulder_xyz,
         elbow_xyz_centered=elbow_xyz_centered,
         wrist_xyz_centered=wrist_xyz_centered)

print("Saved elbow_sweep_45to90_1s.npz")
print("Wrist trajectory shape:", wrist_xyz_centered.shape)

# to load this into the OpenSim visualizer
column_labels = ["elv_angle", "shoulder_elv", "shoulder_rot",
                  "elbow_flexion", "pro_sup", "deviation", "flexion"]
table = osm.TimeSeriesTable()
labels = osm.StdVectorString()
for c in column_labels:
    labels.append(c)
table.setColumnLabels(labels)

for t_idx, t in enumerate(times):
    row_deg = [ELV_ANGLE_DEG, SHOULDER_ELV_DEG, SHOULDER_ROT_DEG,
               elbow_flexion_deg[t_idx], PRO_SUP_DEG, DEVIATION_DEG, FLEXION_DEG]
    table.appendRow(t, osm.RowVector(np.radians(row_deg).tolist()))

table.addTableMetaDataString("inDegrees", "no")
osm.STOFileAdapter.write(table, "elbow_sweep_45to90_1s.mot")
print("Wrote elbow_sweep_45to90_1s.mot - load this via File > Preview Experimental Data in OpenSim")