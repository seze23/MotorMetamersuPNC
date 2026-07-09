"""
Feed a custom OpenSim-generated trial through the pretrained ProprioceptiveIllusions
model (coef_seed=0, train_seed=9) and compare its predicted arm state to your
actual ground-truth kinematics.

CONFIRMED from reading the actual repo source (not guessed):
- 25 muscles required, exact order: MUSCLE_NAMES_FOR_ELBOW
- optimal_lengths are in mm; your .sto muscle lengths are in meters -> x1000
- data shape into the model: (N_trials, 10, 25, 1152)  [10 = 5 Ia + 5 II afferent channels]
- labels shape: (N_trials, 1152, 7) = [wrist_X, wrist_Y, wrist_Z, elv_angle, shoulder_elv, shoulder_rot, elbow_flexion]
  (last 4 columns = "joint_coords"; label_dims: 7 in the yaml confirms this)
- SpindleDataset just reads "data"/"labels" from your HDF5 directly -- no special
  file naming needed for a custom trial
- next_valbatch does plain slicing, so a single trial (N=1) works fine even
  though batch_size defaults to 128

ASSUMPTIONS STILL TO VERIFY once you have the actual checkpoint downloaded:
- The exact order of the 4 joint_coords columns (elv_angle, shoulder_elv,
  shoulder_rot, elbow_flexion) -- inferred from ELBOW_ANGLE_INDEX = 6 and
  label_dims = 7, but not seen explicitly written out anywhere in the code.
  Worth confirming against the checkpoint's config.yaml or a real EF3D sample
  once you have Zenodo data downloaded.
- task = "letter_reconstruction_joints" -- this is the general model tested
  against FLAG_PCR/EF3D/ES3D in test_model.py's defaults. Confirm this matches
  the checkpoint you actually have.
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
from inference.test_model_utils_new import load_model, Tester

# ------------------------------------------------------------------------------
# 1. Paths -- adjust these to match your actual file locations
# ------------------------------------------------------------------------------
STO_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/newelbowtest"
MARKERS_NPZ = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/elbow_sweep_markers_1152.npz"  # regenerate at 1152 timepoints, same as .sto
CONFIG_PATH = os.path.join(REPO_DIR, "extract_data/configs/train_test_data_spindles_extended.yaml")

OUTPUT_HDF5 = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/my_trial.hdf5"

# From test_model.py's model_path construction, seed=0 (coef), train_seed=9
BASE_DIR = REPO_DIR
MODEL_PATH_PREFIX = "optimized_linear_extended"
N_AFF = 5
TASK = "letter_reconstruction_joints"
COEF_SEED = 0
TRAIN_SEED = 9
MODEL_PATH = os.path.join(
    BASE_DIR,
    f"trained_models/experiment_causal_flag-pcr_{MODEL_PATH_PREFIX}_{N_AFF}_{N_AFF}_{TASK}",
    f"spatiotemporal_4_8-8-32-64_7171_{COEF_SEED}_{TRAIN_SEED}",
)

MUSCLE_NAMES_FOR_ELBOW = ['CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP', 'LAT1', 'LAT2', 'LAT3', 'PECM1',
                          'PECM2', 'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN', 'ANC', 'BIClong',
                          'BICshort', 'BRA', 'BRD', 'ECRL', 'PT', 'TRIlat', 'TRIlong', 'TRImed']

TIME_STEPS = 1152

# ------------------------------------------------------------------------------
# 2. Load your regenerated 1152-timepoint muscle length / velocity data
# ------------------------------------------------------------------------------
def load_sto(path):
    with open(path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if l.strip() == "endheader")
    df = pd.read_csv(path, sep="\t", skiprows=header_end + 1)
    df.columns = [c.strip() for c in df.columns]
    return df

length_df = load_sto(os.path.join(STO_DIR, "TestFull_MuscleAnalysis_Length.sto"))
velocity_path = os.path.join(STO_DIR, "TestFull_MuscleAnalysis_FiberVelocity.sto")
if not os.path.exists(velocity_path):
    raise FileNotFoundError(
        f"{velocity_path} not found -- rerun the Analyze Tool with "
        "Muscle Analysis including Fiber Velocity before running this script."
    )
velocity_df = load_sto(velocity_path)

assert len(length_df) == TIME_STEPS, f"Expected {TIME_STEPS} rows, got {len(length_df)} -- did you regenerate at 1152 timepoints?"

# Subset + reorder to the 25 muscles the model expects, convert m -> mm
muscle_lengths = np.zeros((1, 25, TIME_STEPS), dtype=np.float32)
muscle_velocities = np.zeros((1, 25, TIME_STEPS), dtype=np.float32)

for i, muscle in enumerate(MUSCLE_NAMES_FOR_ELBOW):
    if muscle not in length_df.columns:
        raise KeyError(f"Muscle '{muscle}' not found in Right_MuscleAnalysis_Length.sto columns")
    muscle_lengths[0, i, :] = length_df[muscle].to_numpy() * 1000  # m -> mm
    muscle_velocities[0, i, :] = velocity_df[muscle].to_numpy() * 1000  # m/s -> mm/s

# Acceleration is not a native OpenSim output -- derive via gradient
dt = 1 / 240.0
muscle_accelerations = np.gradient(muscle_velocities, dt, axis=2).astype(np.float32)

# ------------------------------------------------------------------------------
# 3. Build labels: [wrist_X, wrist_Y, wrist_Z, elv_angle, shoulder_elv, shoulder_rot, elbow_flexion]
# ------------------------------------------------------------------------------
q_df = load_sto(os.path.join(STO_DIR, "TestFull_Kinematics_q.sto"))
assert len(q_df) == TIME_STEPS

d = np.load(MARKERS_NPZ)
wrist_xyz = d["wrist_xyz"]  # shape (TIME_STEPS, 3), meters, ground frame
assert wrist_xyz.shape[0] == TIME_STEPS, "marker npz must also be regenerated at 1152 timepoints"

labels = np.zeros((1, TIME_STEPS, 7), dtype=np.float32)
labels[0, :, 0:3] = wrist_xyz  # NOTE: verify units/frame match what model expects (likely mm, ground frame)
labels[0, :, 3] = q_df["elv_angle"].to_numpy()
labels[0, :, 4] = q_df["shoulder_elv"].to_numpy()
labels[0, :, 5] = q_df["shoulder_rot"].to_numpy()
labels[0, :, 6] = q_df["elbow_flexion"].to_numpy()

# ------------------------------------------------------------------------------
# 4. Normalize muscle data + compute spindle firing rates (reusing their exact code)
# ------------------------------------------------------------------------------
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Make coefficient paths absolute -- get_sampled_coefficients() reads these
# directly from config and writes a cache file next to them, so a relative
# path here breaks depending on the script's current working directory.
config["i_a_coeff_path"] = os.path.join(REPO_DIR, config["i_a_coeff_path"])
config["ii_coeff_path"] = os.path.join(REPO_DIR, config["ii_coeff_path"])

muscles = config["muscles"]
num_coefficients = [config["num_i_a"], config["num_ii"]]
coefficients = {
    key: load_coefficients(config[key + "_coeff_path"])
    for key in ["i_a", "ii"]
}
sampled_coefficients = get_sampled_coefficients(config, num_coefficients, muscles, coefficients)

data = {"lengths": muscle_lengths, "velocities": muscle_velocities, "accelerations": muscle_accelerations}
data = normalize(data["lengths"], data["velocities"], data["accelerations"], config["optimal_lengths"])

chunk_data = process_chunk(data, coefficients, num_coefficients, muscles, chunk_size=1, sampled_coefficients=sampled_coefficients)
chunk_data = chunk_data.astype(np.float32)  # process_chunk defaults to float64; model weights are float32
print("Spindle firing rate data shape:", chunk_data.shape)  # expect (1, 10, 25, 1152)

# ------------------------------------------------------------------------------
# 5. Write to HDF5 in the exact format SpindleDataset expects
# ------------------------------------------------------------------------------
with h5py.File(OUTPUT_HDF5, "w") as f:
    f.create_dataset("data", data=chunk_data)
    f.create_dataset("labels", data=labels)
print(f"Wrote {OUTPUT_HDF5}")

# ------------------------------------------------------------------------------
# 6. Load the pretrained model and run inference
# ------------------------------------------------------------------------------
from train.new_spindle_dataset import SpindleDataset

with open(os.path.join(MODEL_PATH, "config.yaml"), "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
from inference.test_model_utils_new import parse_config_value
model_config = {k: parse_config_value(v) for k, v in model_config.items()}

test_data = SpindleDataset(
    OUTPUT_HDF5,
    dataset_type="test",
    key="spindle_info",
    task=TASK,
    aclass=None,
    need_muscles=False,
    new_size=model_config["input_shape"][-1],
)

device = torch.device("cpu")  # forced -- this node's GPU (sm_120) isn't supported by the installed PyTorch build

tester = load_model(
    model_config, MODEL_PATH, TASK, device, test_data,
    causal=True, save_dir=REPO_DIR,
)

predictions, batch_X_used = tester.get_predictions()
print("Predictions shape:", predictions.shape)  # expect (1, 1152, 7)

# ------------------------------------------------------------------------------
# 7. Plot predicted vs ground-truth
# ------------------------------------------------------------------------------
pred = predictions[0].cpu().detach().numpy()
true = labels[0]
times = np.arange(TIME_STEPS) / 240.0

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
axes[0].plot(times, true[:, 0], label="true wrist X", c="black")
axes[0].plot(times, pred[:, 0], label="predicted wrist X", c="red", linestyle="--")
axes[0].legend(); axes[0].set_ylabel("Wrist X")

axes[1].plot(times, true[:, 1], label="true wrist Y", c="black")
axes[1].plot(times, pred[:, 1], label="predicted wrist Y", c="red", linestyle="--")
axes[1].legend(); axes[1].set_ylabel("Wrist Y")

axes[2].plot(times, true[:, 2], label="true wrist Z", c="black")
axes[2].plot(times, pred[:, 2], label="predicted wrist Z", c="red", linestyle="--")
axes[2].legend(); axes[2].set_ylabel("Wrist Z")

axes[3].plot(times, true[:, 6], label="true elbow flexion", c="black")
axes[3].plot(times, pred[:, 6], label="predicted elbow flexion", c="red", linestyle="--")
axes[3].legend(); axes[3].set_ylabel("Elbow flexion (deg)")
axes[3].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/prediction_vs_truth.png", dpi=150)
print("Saved prediction_vs_truth.png")

wrist_rmse = np.sqrt(np.mean((pred[:, :3] - true[:, :3])**2))
elbow_rmse = np.sqrt(np.mean((pred[:, 6] - true[:, 6])**2))
print(f"Wrist XYZ RMSE: {wrist_rmse:.3f}")
print(f"Elbow angle RMSE: {elbow_rmse:.3f} deg")