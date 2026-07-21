"""
Final inference script for center-out reach pipeline.

Outputs:
  1. CSV: per-timepoint predicted vs ground truth for shoulder_rot,
     shoulder_elv, elbow_flexion, and XYZ end-effector coordinates,
     plus L2 distance from true trajectory at each timepoint
  2. Terminal: RMSE (deg) for each angle, wrist XYZ RMSE (cm),
     mean L2 distance (cm)
  3. PNG: 2D XZ trajectory comparison (flattened along Y)
  4. PNG: 7-panel time series predicted vs ground truth

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions
  conda activate proprioception
  python3 newinference.py
"""

import os
import sys
import h5py
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions"
sys.path.insert(0, REPO_DIR)

from utils.visualize_sample import get_shoulder_elbow_wrist_loc
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset

# --- Paths ---
FR_NPZ      = os.path.join(REPO_DIR, "dataexp/elbow_spindle_firing_rates.npz")
KIN_STO     = os.path.join(REPO_DIR, "dataexp/newelbowtest/TestFull_Kinematics_q.sto")
OUTPUT_HDF5 = os.path.join(REPO_DIR, "dataexp/my_trial_final.hdf5")
OUTPUT_CSV  = os.path.join(REPO_DIR, "dataexp/inference_results.csv")
SAVE_DIR    = os.path.join(REPO_DIR, "dataexp")

COEF_SEED  = 0
TRAIN_SEED = 9
MODEL_PATH = os.path.join(
    REPO_DIR,
    "trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints",
    f"spatiotemporal_4_8-8-32-64_7171_{COEF_SEED}_{TRAIN_SEED}",
)

TIME_STEPS  = 1152
SAMPLE_RATE = 240

# --- Helpers ---
def load_sto(path):
    with open(path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if l.strip() == "endheader")
    df = pd.read_csv(path, sep="\t", skiprows=header_end + 1)
    df.columns = [c.strip() for c in df.columns]
    return df

# ------------------------------------------------------------------------------
# 1. Load spindle firing rates
# ------------------------------------------------------------------------------
print("Loading spindle firing rates...")
fr_data    = np.load(FR_NPZ, allow_pickle=True)
chunk_data = fr_data['firing_rates'].astype(np.float32)
times      = fr_data['times']
assert chunk_data.shape == (1, 10, 25, TIME_STEPS)
print(f"  shape: {chunk_data.shape}, range: {chunk_data.min():.2f}->{chunk_data.max():.2f} Hz")

# ------------------------------------------------------------------------------
# 2. Build labels
# ------------------------------------------------------------------------------
print("Building labels...")
q_df = load_sto(KIN_STO)
assert len(q_df) == TIME_STEPS

labels_for_fk = np.zeros((TIME_STEPS, 7), dtype=np.float32)
labels_for_fk[:, 3] = q_df["elv_angle"].to_numpy()
labels_for_fk[:, 4] = q_df["shoulder_elv"].to_numpy()
labels_for_fk[:, 5] = q_df["shoulder_rot"].to_numpy()
labels_for_fk[:, 6] = q_df["elbow_flexion"].to_numpy()

_, _, wrist_loc = get_shoulder_elbow_wrist_loc(labels_for_fk)

labels = np.zeros((1, TIME_STEPS, 7), dtype=np.float32)
labels[0, :, 0:3] = wrist_loc          # XYZ cm
labels[0, :, 3]   = labels_for_fk[:, 3]
labels[0, :, 4]   = labels_for_fk[:, 4]
labels[0, :, 5]   = labels_for_fk[:, 5]
labels[0, :, 6]   = labels_for_fk[:, 6]

# ------------------------------------------------------------------------------
# 3. Write HDF5 and run inference
# ------------------------------------------------------------------------------
with h5py.File(OUTPUT_HDF5, "w") as f:
    f.create_dataset("data",   data=chunk_data)
    f.create_dataset("labels", data=labels)

print("Loading model and running inference...")
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

device = torch.device("cpu")
tester = load_model(
    model_config, MODEL_PATH, "letter_reconstruction_joints",
    device, test_data, causal=True, save_dir=REPO_DIR,
)

predictions, _ = tester.get_predictions()
pred = predictions[0].cpu().detach().numpy()  # (1152, 7)
true = labels[0]                               # (1152, 7)

# col order: [wrist_X, wrist_Y, wrist_Z, elv_angle, shoulder_elv, shoulder_rot, elbow_flexion]

# ------------------------------------------------------------------------------
# 4. Compute per-timepoint metrics
# ------------------------------------------------------------------------------
t = np.arange(TIME_STEPS) / SAMPLE_RATE

# L2 distance between predicted and true wrist XYZ at each timepoint (cm)
l2_per_timepoint = np.sqrt(np.sum((pred[:, :3] - true[:, :3])**2, axis=1))

# Build output dataframe
results_df = pd.DataFrame({
    "time_s":                   t,
    # Ground truth
    "true_wrist_X_cm":          true[:, 0],
    "true_wrist_Y_cm":          true[:, 1],
    "true_wrist_Z_cm":          true[:, 2],
    "true_shoulder_elv_deg":    true[:, 4],
    "true_shoulder_rot_deg":    true[:, 5],
    "true_elbow_flexion_deg":   true[:, 6],
    # Predictions
    "pred_wrist_X_cm":          pred[:, 0],
    "pred_wrist_Y_cm":          pred[:, 1],
    "pred_wrist_Z_cm":          pred[:, 2],
    "pred_shoulder_elv_deg":    pred[:, 4],
    "pred_shoulder_rot_deg":    pred[:, 5],
    "pred_elbow_flexion_deg":   pred[:, 6],
    # Error
    "l2_distance_cm":           l2_per_timepoint,
})

results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved per-timepoint results -> {OUTPUT_CSV}")

# ------------------------------------------------------------------------------
# 5. Terminal summary
# ------------------------------------------------------------------------------
sh_elv_rmse  = np.sqrt(np.mean((pred[:, 4] - true[:, 4])**2))
sh_rot_rmse  = np.sqrt(np.mean((pred[:, 5] - true[:, 5])**2))
elbow_rmse   = np.sqrt(np.mean((pred[:, 6] - true[:, 6])**2))
wrist_rmse   = np.sqrt(np.mean((pred[:, :3] - true[:, :3])**2))
mean_l2      = l2_per_timepoint.mean()
max_l2       = l2_per_timepoint.max()

print("\n" + "="*50)
print("INFERENCE SUMMARY")
print("="*50)
print(f"  Shoulder elevation RMSE:  {sh_elv_rmse:.3f} deg")
print(f"  Shoulder rotation RMSE:   {sh_rot_rmse:.3f} deg")
print(f"  Elbow flexion RMSE:       {elbow_rmse:.3f} deg")
print(f"  Wrist XYZ RMSE:           {wrist_rmse:.3f} cm")
print(f"  Mean L2 distance:         {mean_l2:.3f} cm")
print(f"  Max L2 distance:          {max_l2:.3f} cm")
print("="*50)

# ------------------------------------------------------------------------------
# 6. Figure A: 2D XZ trajectory comparison (flattened along Y)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

from matplotlib.colors import Normalize
cmap   = plt.cm.plasma
norm   = Normalize(vmin=0, vmax=TIME_STEPS)
alphas = np.linspace(0.2, 1, TIME_STEPS)
colors = [(cmap(norm(i))[0], cmap(norm(i))[1], cmap(norm(i))[2], alphas[i])
          for i in range(TIME_STEPS)]

# Ground truth -- XZ plane (Y flattened)
ax.scatter(true[:, 0], true[:, 2], c=colors, s=12, label="Ground truth", zorder=3)
ax.scatter(true[0, 0],  true[0, 2],  c='black', s=50, zorder=4, marker='o')   # start
ax.scatter(true[-1, 0], true[-1, 2], c='black', s=50, zorder=4, marker='s')   # end

# Predicted -- XZ plane
ax.scatter(pred[:, 0], pred[:, 2], c='crimson', s=5, alpha=0.4,
           label="Predicted", zorder=2)

ax.set_xlabel("X (cm) — lateral")
ax.set_ylabel("Z (cm) — anterior/posterior")
ax.set_title("Wrist trajectory: XZ plane\n(flattened along Y/vertical)", fontsize=11)
ax.legend(fontsize=9)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Colorbar for time
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label("Timepoint", fontsize=8)
cbar.set_ticks([0, TIME_STEPS//2, TIME_STEPS])
cbar.set_ticklabels(["0s", "2.4s", "4.8s"])

plt.tight_layout()
fig_xz = os.path.join(SAVE_DIR, "trajectory_xz_comparison.png")
plt.savefig(fig_xz, dpi=150, bbox_inches="tight")
print(f"Saved XZ trajectory plot -> {fig_xz}")

# ------------------------------------------------------------------------------
# 7. Figure B: 6-panel time series for the variables of interest
# ------------------------------------------------------------------------------
plot_cols = [
    (4, "Shoulder elevation (deg)"),
    (5, "Shoulder rotation (deg)"),
    (6, "Elbow flexion (deg)"),
    (0, "Wrist X (cm)"),
    (1, "Wrist Y (cm)"),
    (2, "Wrist Z (cm)"),
]

fig2, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)
for ax, (col_idx, col_name) in zip(axes, plot_cols):
    rmse = np.sqrt(np.mean((pred[:, col_idx] - true[:, col_idx])**2))
    ax.plot(t, true[:, col_idx], c="black",  label="Ground truth", linewidth=1.5)
    ax.plot(t, pred[:, col_idx], c="crimson", label="Predicted",   linewidth=1.5, linestyle="--")
    ax.set_ylabel(col_name, fontsize=8)
    ax.set_title(f"RMSE: {rmse:.3f}", fontsize=7, loc="left", pad=2)
    ax.legend(fontsize=7, loc="upper right")

# L2 distance on a 7th panel
ax7 = fig2.add_subplot(7, 1, 7)
ax7.plot(t, l2_per_timepoint, c="darkorange", linewidth=1.5)
ax7.axhline(mean_l2, c="darkorange", linewidth=1, linestyle="--", alpha=0.6)
ax7.set_ylabel("L2 dist (cm)", fontsize=8)
ax7.set_xlabel("Time (s)")
ax7.set_title(f"Mean L2: {mean_l2:.3f} cm", fontsize=7, loc="left", pad=2)

fig2.suptitle(
    f"Predicted vs Ground Truth — Horizontal Elbow Sweep 60°→100°→60°\n"
    f"Shoulder elv RMSE: {sh_elv_rmse:.2f}°  |  "
    f"Shoulder rot RMSE: {sh_rot_rmse:.2f}°  |  "
    f"Elbow RMSE: {elbow_rmse:.2f}°  |  "
    f"Wrist RMSE: {wrist_rmse:.2f} cm",
    fontsize=10, y=1.01
)
plt.tight_layout()

fig2_path = os.path.join(SAVE_DIR, "prediction_vs_truth_final.png")
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"Saved time series plot -> {fig2_path}")