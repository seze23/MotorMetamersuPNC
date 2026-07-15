import opensim as osm
import numpy as np

MODEL_PATH = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
OUTPUT_MOT = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/elbow_sweep_horizontal_1152.mot"

# --- KINARM-like horizontal posture ---
# Confirmed visually in OpenSim: upper arm horizontal, forearm forward,
# elbow flexion sweeps hand across horizontal plane
ELV_ANGLE_DEG    = 0    # abduction plane -- arm out to the side
SHOULDER_ELV_DEG = 85   # arm nearly horizontal
SHOULDER_ROT_DEG = 0    # neutral rotation
ELBOW_START_DEG  = 60   # starting flexion
ELBOW_END_DEG    = 100  # ending flexion -- hand moves outward/forward
PRO_SUP_DEG      = 0
DEVIATION_DEG    = 0
FLEXION_DEG      = 0

# --- Timing structure ---
# Total: 1152 timepoints at 240Hz = 4.8s
# Short holds at start/end to establish posture without breaking movement flow
SAMPLE_RATE  = 240
N_TOTAL      = 1152
DURATION     = N_TOTAL / SAMPLE_RATE  # 4.8s

# Phase durations in seconds
T_HOLD_START = 0.5   # 120 frames -- short enough not to break flow
T_REACH      = 2.0   # 480 frames -- minimum-jerk outward reach
T_HOLD_END   = 0.5   # 120 frames
T_RETURN     = 1.8   # 432 frames -- return to start

assert abs((T_HOLD_START + T_REACH + T_HOLD_END + T_RETURN) - DURATION) < 0.01, \
    "Phase durations must sum to 4.8s"

# Frame counts
N_HOLD_START = int(T_HOLD_START * SAMPLE_RATE)   # 120
N_REACH      = int(T_REACH      * SAMPLE_RATE)   # 480
N_HOLD_END   = int(T_HOLD_END   * SAMPLE_RATE)   # 120
N_RETURN     = N_TOTAL - N_HOLD_START - N_REACH - N_HOLD_END  # 432

def min_jerk(n_frames):
    """
    Minimum-jerk trajectory from 0 to 1.
    Velocity is zero at start and end -- no discontinuities.
    s(t) = 10t^3 - 15t^4 + 6t^5
    """
    t = np.linspace(0, 1, n_frames)
    return 10*t**3 - 15*t**4 + 6*t**5

# Build elbow angle trajectory
elbow = np.concatenate([
    np.full(N_HOLD_START, ELBOW_START_DEG),
    ELBOW_START_DEG + (ELBOW_END_DEG - ELBOW_START_DEG) * min_jerk(N_REACH),
    np.full(N_HOLD_END,   ELBOW_END_DEG),
    ELBOW_END_DEG   + (ELBOW_START_DEG - ELBOW_END_DEG) * min_jerk(N_RETURN),
])

assert len(elbow) == N_TOTAL, f"Expected {N_TOTAL} frames, got {len(elbow)}"
times = np.linspace(0, DURATION, N_TOTAL)

# --- Write .mot file ---
column_labels = ["elv_angle", "shoulder_elv", "shoulder_rot",
                  "elbow_flexion", "pro_sup", "deviation", "flexion"]

table = osm.TimeSeriesTable()
labels = osm.StdVectorString()
for c in column_labels:
    labels.append(c)
table.setColumnLabels(labels)

for t_idx, t in enumerate(times):
    row_deg = [ELV_ANGLE_DEG, SHOULDER_ELV_DEG, SHOULDER_ROT_DEG,
               elbow[t_idx], PRO_SUP_DEG, DEVIATION_DEG, FLEXION_DEG]
    table.appendRow(t, osm.RowVector(np.radians(row_deg).tolist()))

table.addTableMetaDataString("inDegrees", "no")
osm.STOFileAdapter.write(table, OUTPUT_MOT)

print(f"Wrote {OUTPUT_MOT}")
print(f"Duration: {DURATION}s, {N_TOTAL} timepoints at {SAMPLE_RATE}Hz")
print(f"Posture: elv_angle={ELV_ANGLE_DEG}, shoulder_elv={SHOULDER_ELV_DEG}, shoulder_rot={SHOULDER_ROT_DEG}")
print(f"Elbow: {ELBOW_START_DEG} -> {ELBOW_END_DEG} -> {ELBOW_START_DEG} deg")
print(f"Phase frames: hold={N_HOLD_START}, reach={N_REACH}, hold={N_HOLD_END}, return={N_RETURN}")
print()
print("Next steps:")
print("1. Transfer this .mot to your Mac")
print("2. Load DefaultMOBL_ARMS_fixed_41.osim in OpenSim")
print("3. Load this .mot via File -> Load Motion")
print("4. Scrub through -- confirm arm moves horizontally, velocity smooth at start/end")
print("5. Tools -> Analyze with:")
print("   - Muscle Analysis: Fiber Length checked")
print("   - Body Kinematics checked")
print("   - Kinematics checked")
print("   - Time range: 0.0 to 4.8")
print("6. Transfer new .sto files to cluster -> newelbowtest/")