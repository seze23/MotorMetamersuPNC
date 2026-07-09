import opensim as osm
import numpy as np

MODEL_PATH = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
OUTPUT_MOT = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/elbow_sweep_45to90_1152.mot"

# --- Fixed posture (same convention confirmed correct earlier) ---
SHOULDER_ELV_DEG = 70
SHOULDER_ROT_DEG = 20
ELBOW_START_DEG  = 45
ELBOW_END_DEG    = 90

ELV_ANGLE_DEG = 0
PRO_SUP_DEG   = 0
DEVIATION_DEG = 0
FLEXION_DEG   = 0

# --- Match the pretrained model's expected input shape ---
N_TIMEPOINTS = 1152     # was 67
SAMPLE_RATE = 240       # Hz
DURATION = N_TIMEPOINTS / SAMPLE_RATE  # = 4.8s

times = np.linspace(0, DURATION, N_TIMEPOINTS)

elbow_flexion_deg = ELBOW_START_DEG + (times / DURATION) * (ELBOW_END_DEG - ELBOW_START_DEG)

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
osm.STOFileAdapter.write(table, OUTPUT_MOT)

print(f"Wrote {OUTPUT_MOT}")
print(f"Duration: {DURATION}s, {N_TIMEPOINTS} timepoints at {SAMPLE_RATE}Hz")
print(f"Elbow flexion: {elbow_flexion_deg[0]:.2f} -> {elbow_flexion_deg[-1]:.2f} deg")
print()
print("Next step: load this .mot in OpenSim, load the model, and run")
print("Tools > Analyze with Muscle Analysis (Length + Fiber Velocity)")
print("and Body Kinematics, over the full 0-4.8s time range.")