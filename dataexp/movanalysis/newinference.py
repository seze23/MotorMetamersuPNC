"""
Final inference script. Loads everything from:
  - dataexp/elbow_fiber_lengths.npz   (fiber lengths + joint angles)
  - dataexp/elbow_spindle_firing_rates.npz  (precomputed firing rates)

No .sto files needed.

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions
  conda activate proprioception
  python3 run_inference_final.py
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
FL_NPZ      = os.path.join(REPO_DIR, "dataexp/elbow_fiber_lengths.npz")
FR_NPZ      = os.path.join(REPO_DIR, "dataexp/elbow_spindle_firing_rates.npz")
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

# COORD_LABEL_ORDER matches extractdata.py
# joint_angles columns: [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion, pro_sup, deviation, flexion]
ELV_IDX    = 0
SH_ELV_IDX = 1
SH_ROT_IDX = 2
ELBOW_IDX  = 3

# ------------------------------------------------------------------------------
# 1. Load fiber lengths + joint angles from npz
# ------------------------------------------------------------------------------
print("Loading fiber lengths and joint angles...")
fl_data      = np.load(FL_NPZ, allow_pickle=True)
fiber_lengths = fl_data['fiber_lengths']   # (1152, 25) mm
joint_angles  = fl_data['joint_angles']    # (1152, 7) degrees
times         = fl_data['times']           # (1152,)

assert fiber_lengths.shape == (TIME_STEPS, 25)
assert joint_angles.shape  == (TIME_STEPS, 7)

print(f"  elbow_flexion: {joint_angles[:, ELBOW_IDX].min():.1f} -> "
      f"{joint_angles[:, ELBOW_IDX].max():.1f} -> "
      f"{joint_angles[-1, ELBOW_IDX]:.1f} deg  (should be 60->100->60)")

# ------------------------------------------------------------------------------
# 2. Load precomputed spindle firing rates
# ------------------------------------------------------------------------------
print("Loading spindle firing rates...")
fr_data    = np.load(FR_NPZ, allow_pickle=True)
chunk_data = fr_data['firing_rates'].astype(np.float32)  # (1, 10, 25, 1152)
assert chunk_data.shape == (1, 10, 25, TIME_STEPS)
print(f"  range: {chunk_data.min():.2f} -> {chunk_data.max():.2f} Hz")

# ------------------------------------------------------------------------------
# 3. Build labels (1, 1152, 7)
#    XYZ from FK using joint angles, angles directly from joint_angles
# ------------------------------------------------------------------------------
print("Building labels...")

# get_shoulder_elbow_wrist_loc expects (time, 7) with cols at indices 3,4,5,6
labels_for_fk = np.zeros((TIME_STEPS, 7), dtype=np.float32)
labels_for_fk[:, 3] = joint_angles[:, ELV_IDX]
labels_for_fk[:, 4] = joint_angles[:, SH_ELV_IDX]
labels_for_fk[:, 5] = joint_angles[:, SH_ROT_IDX]
labels_for_fk[:, 6] = joint_angles[:, ELBOW_IDX]

_, _, wrist_loc = get_shoulder_elbow_wrist_loc(labels_for_fk)  # (1152, 3) cm

labels = np.zeros((1, TIME_STEPS, 7), dtype=np.float32)
labels[0, :, 0:3] = wrist_loc
labels[0, :, 3]   = joint_angles[:, ELV_IDX]
labels[0, :, 4]   = joint_angles[:, SH_ELV_IDX]
labels[0, :, 5]   = joint_angles[:, SH_ROT_IDX]
labels[0, :, 6]   = joint_angles[:, ELBOW_IDX]

print(f"  wrist XYZ range (cm): X={wrist_loc[:,0].min():.1f}->{wrist_loc[:,0].max():.1f}  "
      f"Z={wrist_loc[:,2].min():.1f}->{wrist_loc[:,2].max():.1f}")

# ------------------------------------------------------------------------------
# 4. Write HDF5 and run inference
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
# 5. Compute metrics
# ------------------------------------------------------------------------------
t = np.arange(TIME_STEPS) / SAMPLE_RATE

l2_per_timepoint = np.sqrt(np.sum((pred[:, :3] - true[:, :3])**2, axis=1))

sh_elv_rmse = np.sqrt(np.mean((pred[:, 4] - true[:, 4])**2))
sh_rot_rmse = np.sqrt(np.mean((pred[:, 5] - true[:, 5])**2))
elbow_rmse  = np.sqrt(np.mean((pred[:, 6] - true[:, 6])**2))
wrist_rmse  = np.sqrt(np.mean((pred[:, :3] - true[:, :3])**2))
mean_l2     = l2_per_timepoint.mean()
max_l2      = l2_per_timepoint.max()

# --- CSV ---
results_df = pd.DataFrame({
    "time_s":                   t,
    "true_wrist_X_cm":          true[:, 0],
    "true_wrist_Y_cm":          true[:, 1],
    "true_wrist_Z_cm":          true[:, 2],
    "true_shoulder_elv_deg":    true[:, 4],
    "true_shoulder_rot_deg":    true[:, 5],
    "true_elbow_flexion_deg":   true[:, 6],
    "pred_wrist_X_cm":          pred[:, 0],
    "pred_wrist_Y_cm":          pred[:, 1],
    "pred_wrist_Z_cm":          pred[:, 2],
    "pred_shoulder_elv_deg":    pred[:, 4],
    "pred_shoulder_rot_deg":    pred[:, 5],
    "pred_elbow_flexion_deg":   pred[:, 6],
    "l2_distance_cm":           l2_per_timepoint,
})
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {OUTPUT_CSV}")

# --- Terminal summary ---
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
# 6. Figure A: 2D XZ trajectory (Y flattened)
# ------------------------------------------------------------------------------
from matplotlib.colors import Normalize

fig, ax = plt.subplots(figsize=(7, 6))
cmap   = plt.cm.plasma
norm   = Normalize(vmin=0, vmax=TIME_STEPS)
alphas = np.linspace(0.3, 1, TIME_STEPS)
colors = [(cmap(norm(i))[0], cmap(norm(i))[1], cmap(norm(i))[2], alphas[i])
          for i in range(TIME_STEPS)]

ax.scatter(true[:, 0], true[:, 2], c=colors, s=15, label="Ground truth", zorder=3)
ax.scatter(true[0, 0],  true[0, 2],  c='black', s=80, zorder=5, marker='o', label="Start")
ax.scatter(true[-1, 0], true[-1, 2], c='black', s=80, zorder=5, marker='s', label="End")
ax.scatter(pred[:, 0], pred[:, 2], c='crimson', s=8, alpha=0.5, label="Predicted", zorder=2)

ax.set_xlabel("X (cm) — lateral", fontsize=11)
ax.set_ylabel("Z (cm) — anterior/posterior", fontsize=11)
ax.set_title("Wrist XZ plane", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Time", fontsize=8)
cbar.set_ticks([0, TIME_STEPS//2, TIME_STEPS])
cbar.set_ticklabels(["0s", "2.4s", "4.8s"])

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "traj_xz_comparison.png"), dpi=150)
print(f"Saved traj_xz_comparison.png")

# ------------------------------------------------------------------------------
# 7. Figure B: 7-panel time series (clean single figure)
# ------------------------------------------------------------------------------
plot_cols = [
    (4, "Shoulder elevation (deg)"),
    (5, "Shoulder rotation (deg)"),
    (6, "Elbow flexion (deg)"),
    (0, "Wrist X (cm)"),
    (1, "Wrist Y (cm)"),
    (2, "Wrist Z (cm)"),
]

fig2, axes = plt.subplots(7, 1, figsize=(10, 18), sharex=True)

for i, (ax, (col_idx, col_name)) in enumerate(zip(axes[:6], plot_cols)):
    rmse = np.sqrt(np.mean((pred[:, col_idx] - true[:, col_idx])**2))
    ax.plot(t, true[:, col_idx], c="black",   label="Ground truth", linewidth=1.5)
    ax.plot(t, pred[:, col_idx], c="crimson", label="Predicted",    linewidth=1.5, linestyle="--")
    ax.set_ylabel(col_name, fontsize=9)
    ax.set_title(f"RMSE: {rmse:.3f}", fontsize=8, loc="left", pad=2)
    ax.legend(fontsize=8, loc="upper right")

# L2 distance panel
axes[6].plot(t, l2_per_timepoint, c="darkorange", linewidth=1.5)
axes[6].axhline(mean_l2, c="darkorange", linewidth=1, linestyle="--", alpha=0.7,
                label=f"Mean: {mean_l2:.2f} cm")
axes[6].set_ylabel("L2 distance (cm)", fontsize=9)
axes[6].set_xlabel("Time (s)", fontsize=9)
axes[6].set_title("Wrist spatial error per timepoint", fontsize=8, loc="left", pad=2)
axes[6].legend(fontsize=8)

fig2.suptitle(
    f"Horizontal elbow sweep prediction vs. truth\n"
    f"sh_elv RMSE: {sh_elv_rmse:.2f}°  |  sh_rot RMSE: {sh_rot_rmse:.2f}°  |  "
    f"elbow RMSE: {elbow_rmse:.2f}°  |  wrist RMSE: {wrist_rmse:.2f} cm  |  "
    f"mean L2: {mean_l2:.2f} cm",
    fontsize=10
)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "pred_vs_truth_final.png"), dpi=150, bbox_inches="tight")
print("Saved pred_vs_truth_final.png")