"""
Run a custom OpenSim-generated elbow-flexion trial through the pretrained
ProprioceptiveIllusions model (coef_seed=0, train_seed=9).

CONFIRMED from HDF5 inspection and source reading:
- muscle input: FiberLength (not MuscleLength), in mm (x1000 from OpenSim meters)
- velocity/acceleration: derived from FiberLength via np.gradient (FiberVelocity.sto
  from OpenSim outputs a degenerate constant -- not usable)
- firing rates: Ia range 50-180 Hz -- verified against EF3D processed file
- labels XYZ: computed via get_shoulder_elbow_wrist_loc() forward kinematics,
  NOT raw OpenSim marker positions. Output is cm in their world frame.
- labels angles: degrees, column order [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion]
- elbow angle: no 180-degree swap needed (EF3D range 45-130 matches raw flexion)
- MARKERS_NPZ not needed -- FK from joint angles is authoritative

KNOWN POSTURE CAVEAT:
- elv_angle=0 is outside EF3D training distribution (19-79 deg)
- predictions may be worse than at a central posture (try elv_angle=49 to compare)
"""

import os
import sys
import h5py
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

REPO_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions"
sys.path.insert(0, REPO_DIR)

from extract_data.generate_train_test_data import process_chunk
from utils.spindle_FR_helper import normalize, load_coefficients, get_sampled_coefficients
from utils.visualize_sample import get_shoulder_elbow_wrist_loc
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset

# ------------------------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------------------------
STO_DIR     = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/newelbowtest"
CONFIG_PATH = os.path.join(REPO_DIR, "extract_data/configs/train_test_data_spindles_extended.yaml")
OUTPUT_HDF5 = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/my_trial_fixed.hdf5"
SAVE_DIR    = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp"

COEF_SEED  = 0
TRAIN_SEED = 9
MODEL_PATH = os.path.join(
    REPO_DIR,
    "trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints",
    f"spatiotemporal_4_8-8-32-64_7171_{COEF_SEED}_{TRAIN_SEED}",
)

# Confirmed from muscle_names.py and generate_spindle_coefficients.py
MUSCLE_NAMES = [
    'CORB',    'DELT1',   'DELT2',    'DELT3',  'INFSP',
    'LAT1',    'LAT2',    'LAT3',     'PECM1',  'PECM2',
    'PECM3',   'SUBSC',   'SUPSP',    'TMAJ',   'TMIN',
    'ANC',     'BIClong', 'BICshort', 'BRA',    'BRD',
    'ECRL',    'PT',      'TRIlat',   'TRIlong','TRImed'
]

TIME_STEPS = 1152
dt = 1.0 / 240.0

# ------------------------------------------------------------------------------
# 2. Load .sto files
# ------------------------------------------------------------------------------
def load_sto(path):
    with open(path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if l.strip() == "endheader")
    df = pd.read_csv(path, sep="\t", skiprows=header_end + 1)
    df.columns = [c.strip() for c in df.columns]
    return df

# FiberVelocity.sto from OpenSim outputs a degenerate constant -- not usable.
# Derive velocity and acceleration from FiberLength via np.gradient instead,
# consistent with EF3D data_generation pipeline.
fiber_length_path = os.path.join(STO_DIR, "TestFull_MuscleAnalysis_FiberLength.sto")
kinematics_path   = os.path.join(STO_DIR, "TestFull_Kinematics_q.sto")

for p in [fiber_length_path, kinematics_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

length_df = load_sto(fiber_length_path)
q_df      = load_sto(kinematics_path)

assert len(length_df) == TIME_STEPS, \
    f"Expected {TIME_STEPS} rows, got {len(length_df)} -- regenerate at 1152 timepoints"

print(f"Loaded {TIME_STEPS} timepoints from {STO_DIR}")

# ------------------------------------------------------------------------------
# 3. Build muscle arrays: (1, 25, 1152) in mm
#    velocity and acceleration derived from length via np.gradient
# ------------------------------------------------------------------------------
muscle_lengths = np.zeros((1, 25, TIME_STEPS), dtype=np.float32)

for i, muscle in enumerate(MUSCLE_NAMES):
    if muscle not in length_df.columns:
        raise KeyError(
            f"Muscle '{muscle}' not found in FiberLength.sto -- "
            f"available: {list(length_df.columns)[:10]}..."
        )
    muscle_lengths[0, i, :] = length_df[muscle].to_numpy() * 1000  # m -> mm

# Derive velocity and acceleration from length
muscle_velocities    = np.gradient(muscle_lengths,    dt, axis=2).astype(np.float32)
muscle_accelerations = np.gradient(muscle_velocities, dt, axis=2).astype(np.float32)

print(f"Muscle lengths (mm):      {muscle_lengths.min():.1f} -> {muscle_lengths.max():.1f}")
print(f"Muscle velocities (mm/s): {muscle_velocities.min():.1f} -> {muscle_velocities.max():.1f}")
print(f"Muscle accels (mm/s^2):   {muscle_accelerations.min():.1f} -> {muscle_accelerations.max():.1f}")

# ------------------------------------------------------------------------------
# 4. Build labels: (1, 1152, 7) using their FK function
#    [wrist_X, wrist_Y, wrist_Z (cm, world frame), elv, sh_elv, sh_rot, elbow]
# ------------------------------------------------------------------------------
assert len(q_df) == TIME_STEPS, \
    f"Kinematics_q has {len(q_df)} rows, expected {TIME_STEPS}"

# get_shoulder_elbow_wrist_loc indexes columns 3,4,5,6 -- pass full 7-col array
labels_for_fk = np.zeros((TIME_STEPS, 7), dtype=np.float32)
labels_for_fk[:, 3] = q_df["elv_angle"].to_numpy()
labels_for_fk[:, 4] = q_df["shoulder_elv"].to_numpy()
labels_for_fk[:, 5] = q_df["shoulder_rot"].to_numpy()
labels_for_fk[:, 6] = q_df["elbow_flexion"].to_numpy()

# Returns (shoulder_loc, elbow_loc, wrist_loc) each (time, 3) in cm, world frame
shoulder_loc, elbow_loc, wrist_loc = get_shoulder_elbow_wrist_loc(labels_for_fk)

labels = np.zeros((1, TIME_STEPS, 7), dtype=np.float32)
labels[0, :, 0:3] = wrist_loc
labels[0, :, 3]   = labels_for_fk[:, 3]
labels[0, :, 4]   = labels_for_fk[:, 4]
labels[0, :, 5]   = labels_for_fk[:, 5]
labels[0, :, 6]   = labels_for_fk[:, 6]

print(f"Wrist XYZ (cm): {wrist_loc.min(axis=0)} -> {wrist_loc.max(axis=0)}")
print(f"Elbow angle (deg): {labels[0,:,6].min():.1f} -> {labels[0,:,6].max():.1f}")

# ------------------------------------------------------------------------------
# 5. Normalize + compute spindle firing rates
# ------------------------------------------------------------------------------
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Absolute paths so get_sampled_coefficients writes cache correctly regardless of cwd
config["i_a_coeff_path"] = os.path.join(REPO_DIR, config["i_a_coeff_path"])
config["ii_coeff_path"]  = os.path.join(REPO_DIR, config["ii_coeff_path"])

muscles          = config["muscles"]
num_coefficients = [config["num_i_a"], config["num_ii"]]
coefficients     = {
    key: load_coefficients(config[key + "_coeff_path"])
    for key in ["i_a", "ii"]
}
sampled_coefficients = get_sampled_coefficients(config, num_coefficients, muscles, coefficients)

data       = normalize(muscle_lengths, muscle_velocities, muscle_accelerations, config["optimal_lengths"])
chunk_data = process_chunk(
    data, coefficients, num_coefficients, muscles,
    chunk_size=1, sampled_coefficients=sampled_coefficients
)
chunk_data = chunk_data.astype(np.float32)  # process_chunk defaults float64; model is float32

print(f"Spindle firing rates shape: {chunk_data.shape}")
print(f"Spindle firing rates range: {chunk_data.min():.1f} -> {chunk_data.max():.1f}")
# Should be ~50-180 Hz matching EF3D processed file

# ------------------------------------------------------------------------------
# 6. Write HDF5 in SpindleDataset format
# ------------------------------------------------------------------------------
with h5py.File(OUTPUT_HDF5, "w") as f:
    f.create_dataset("data",   data=chunk_data)
    f.create_dataset("labels", data=labels)
print(f"Wrote {OUTPUT_HDF5}")

# ------------------------------------------------------------------------------
# 7. Load checkpoint and run inference
# ------------------------------------------------------------------------------
with open(os.path.join(MODEL_PATH, "config.yaml"), "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
model_config = {k: parse_config_value(v) for k, v in model_config.items()}

test_data = SpindleDataset(
    OUTPUT_HDF5,
    dataset_type="test",
    key="spindle_info",
    task="letter_reconstruction_joints",
    aclass=None,
    need_muscles=False,
    new_size=model_config["input_shape"][-1],
)

device = torch.device("cpu")  # forced: sm_120 (Blackwell) not supported by this PyTorch build
tester = load_model(
    model_config, MODEL_PATH, "letter_reconstruction_joints",
    device, test_data, causal=True, save_dir=REPO_DIR,
)

predictions, _ = tester.get_predictions()
print(f"Predictions shape: {predictions.shape}")

# ------------------------------------------------------------------------------
# 8. Compare predictions to ground truth
# ------------------------------------------------------------------------------
pred  = predictions[0].cpu().detach().numpy()  # (1152, 7)
true  = labels[0]                               # (1152, 7)
times = np.arange(TIME_STEPS) / 240.0

col_names = [
    "Wrist X (cm)", "Wrist Y (cm)", "Wrist Z (cm)",
    "Elevation (deg)", "Shoulder Elev (deg)",
    "Shoulder Rot (deg)", "Elbow Flexion (deg)"
]

fig, axes = plt.subplots(7, 1, figsize=(10, 18), sharex=True)
for i, (ax, name) in enumerate(zip(axes, col_names)):
    ax.plot(times, true[:, i], c="black", label="ground truth", linewidth=1.5)
    ax.plot(times, pred[:, i], c="red",   label="predicted",    linewidth=1.5, linestyle="--")
    ax.set_ylabel(name, fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
axes[-1].set_xlabel("Time (s)")
plt.suptitle("Predicted vs Ground Truth — Elbow Sweep 45deg to 90deg", fontsize=12, y=1.01)
plt.tight_layout()

out_fig = os.path.join(SAVE_DIR, "prediction_vs_truth_fixed.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"Saved {out_fig}")

print("\nPer-column RMSE:")
for i, name in enumerate(col_names):
    rmse = np.sqrt(np.mean((pred[:, i] - true[:, i])**2))
    print(f"  {name:>24}: {rmse:.3f}")

wrist_rmse = np.sqrt(np.mean((pred[:, :3] - true[:, :3])**2))
elbow_rmse = np.sqrt(np.mean((pred[:,  6] - true[:,  6])**2))
print(f"\nWrist XYZ RMSE: {wrist_rmse:.3f} cm")
print(f"Elbow angle RMSE: {elbow_rmse:.3f} deg")