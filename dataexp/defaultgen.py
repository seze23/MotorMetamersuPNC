import opensim as osm
import numpy as np

MODEL_PATH = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
OUTPUT_MOT = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/elbow_sweep_45to90_1152.mot"

# --- Posture: midpoints of EF3D training distribution ---
# EF3D ranges: elv_angle (19-79), shoulder_elv (39-99), shoulder_rot (-6-54)
# Using midpoints so the model sees a familiar posture
ELV_ANGLE_DEG    = 49   # was 0 -- now inside EF3D distribution
SHOULDER_ELV_DEG = 69   # was 70 -- midpoint of 39-99
SHOULDER_ROT_DEG = 24   # was 20 -- midpoint of -6-54
ELBOW_START_DEG  = 45
ELBOW_END_DEG    = 90

# Wrist coordinates held at neutral
PRO_SUP_DEG   = 0
DEVIATION_DEG = 0
FLEXION_DEG   = 0

# --- Match the pretrained model's expected input shape ---
N_TIMEPOINTS = 1152
SAMPLE_RATE  = 240        # Hz
DURATION     = N_TIMEPOINTS / SAMPLE_RATE  # = 4.8s

times = np.linspace(0, DURATION, N_TIMEPOINTS)

# Linear sweep from 45 to 90 degrees over 4.8 seconds
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
print(f"Posture: elv_angle={ELV_ANGLE_DEG}, shoulder_elv={SHOULDER_ELV_DEG}, shoulder_rot={SHOULDER_ROT_DEG}")
print(f"Elbow flexion: {elbow_flexion_deg[0]:.2f} -> {elbow_flexion_deg[-1]:.2f} deg")
print()
print("Next steps:")
print("1. Transfer this .mot file to your Mac (scp or rsync)")
print("2. Load DefaultMOBL_ARMS_fixed_41.osim in OpenSim")
print("3. Load this .mot via File -> Load Motion")
print("4. Scrub through timeline -- confirm arm actually moves (elbow 45->90)")
print("5. Tools -> Analyze with:")
print("   - Muscle Analysis: Fiber Length checked")
print("   - Body Kinematics checked")
print("   - Kinematics checked")
print("   - Time range: 0.0 to 4.8")
print("6. Transfer new .sto files back to cluster -> newelbowtest/")