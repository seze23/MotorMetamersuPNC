"""
Run inference for all center-out reaches and produce:
  1. Per-direction CSV (timepoint-level predictions + L2)
  2. Per-direction pred_vs_truth time series PNG
  3. Panel B style figure: truth vs predicted in their XY horizontal plane
  4. Terminal summary table

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions
  conda activate proprioception
  python3 centeroutinference.py
"""

import os
import sys
import h5py
import yaml
import numpy as np
import pandas as pd
import torch
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

REPO_DIR      = "/home/sydneyez/sydneyez/ProprioceptiveIllusions"
CENTEROUT_DIR = os.path.join(REPO_DIR, "dataexp/centerout")
sys.path.insert(0, REPO_DIR)

from utils.visualize_sample import get_shoulder_elbow_wrist_loc
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset

COEF_SEED  = 0
TRAIN_SEED = 9
MODEL_PATH = os.path.join(
    REPO_DIR,
    "trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints",
    f"spatiotemporal_4_8-8-32-64_7171_{COEF_SEED}_{TRAIN_SEED}",
)

TIME_STEPS  = 1152
SAMPLE_RATE = 240

DIRECTIONS_ORDERED = [
    "0_right", "45_fwd_right", "90_forward", "135_fwd_left",
    "180_left", "225_back_left", "270_backward", "315_back_right",
]
DIR_COLORS = {
    "0_right":       "#e74c3c",
    "45_fwd_right":  "#e67e22",
    "90_forward":    "#f1c40f",
    "135_fwd_left":  "#2ecc71",
    "180_left":      "#1abc9c",
    "225_back_left": "#3498db",
    "270_backward":  "#9b59b6",
    "315_back_right":"#e91e63",
}

BLACK   = "#1a1a1a"
CRIMSON = "#c0392b"
ORANGE  = "#e67e22"

# Load model once
print("Loading pretrained model...")
with open(os.path.join(MODEL_PATH, "config.yaml"), "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
model_config = {k: parse_config_value(v) for k, v in model_config.items()}
print("Model config loaded.")

spindle_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "center_out_*_spindles.npz")))
if not spindle_files:
    raise FileNotFoundError(
        f"No *_spindles.npz files in {CENTEROUT_DIR}"
    )

print(f"Found {len(spindle_files)} directions:")
for f in spindle_files:
    print(f"  {os.path.basename(f)}")
print()

all_true_xyz = {}
all_pred_xyz = {}
summary_rows = []
t = np.arange(TIME_STEPS) / SAMPLE_RATE

for sp_path in spindle_files:
    direction = (os.path.basename(sp_path)
                 .replace("center_out_", "")
                 .replace("_spindles.npz", ""))
    print(f"Running inference: {direction}")

    # Load data
    sp_data      = np.load(sp_path, allow_pickle=True)
    chunk_data   = sp_data['firing_rates'].astype(np.float32)  # (1,10,25,1152)
    joint_angles = sp_data['joint_angles']                     # (1152,7) degrees

    # Build labels via FK
    # get_shoulder_elbow_wrist_loc indexes columns 3,4,5,6
    labels_for_fk = np.zeros((TIME_STEPS, 7), dtype=np.float32)
    labels_for_fk[:, 3] = joint_angles[:, 0]  # elv_angle
    labels_for_fk[:, 4] = joint_angles[:, 1]  # shoulder_elv
    labels_for_fk[:, 5] = joint_angles[:, 2]  # shoulder_rot
    labels_for_fk[:, 6] = joint_angles[:, 3]  # elbow_flexion

    _, _, wrist_loc = get_shoulder_elbow_wrist_loc(labels_for_fk)  # (1152,3) cm

    labels = np.zeros((1, TIME_STEPS, 7), dtype=np.float32)
    labels[0, :, 0:3] = wrist_loc
    labels[0, :, 3]   = joint_angles[:, 0]
    labels[0, :, 4]   = joint_angles[:, 1]
    labels[0, :, 5]   = joint_angles[:, 2]
    labels[0, :, 6]   = joint_angles[:, 3]

    # Write temp HDF5 and run inference
    tmp_hdf5 = os.path.join(CENTEROUT_DIR, f"_tmp_{direction}.hdf5")
    with h5py.File(tmp_hdf5, "w") as f:
        f.create_dataset("data",   data=chunk_data)
        f.create_dataset("labels", data=labels)

    test_data = SpindleDataset(
        tmp_hdf5,
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
    pred = predictions[0].cpu().detach().numpy()  # (1152,7)
    true = labels[0]                               # (1152,7)
    os.remove(tmp_hdf5)

    # Metrics
    l2          = np.sqrt(np.sum((pred[:, :3] - true[:, :3])**2, axis=1))
    sh_elv_rmse = np.sqrt(np.mean((pred[:, 4] - true[:, 4])**2))
    sh_rot_rmse = np.sqrt(np.mean((pred[:, 5] - true[:, 5])**2))
    elbow_rmse  = np.sqrt(np.mean((pred[:, 6] - true[:, 6])**2))
    wrist_rmse  = np.sqrt(np.mean((pred[:, :3] - true[:, :3])**2))
    mean_l2     = l2.mean()

    summary_rows.append({
        "direction":       direction,
        "sh_elv_rmse_deg": sh_elv_rmse,
        "sh_rot_rmse_deg": sh_rot_rmse,
        "elbow_rmse_deg":  elbow_rmse,
        "wrist_rmse_cm":   wrist_rmse,
        "mean_l2_cm":      mean_l2,
    })

    all_true_xyz[direction] = true[:, :3]
    all_pred_xyz[direction] = pred[:, :3]

    # Per-direction CSV
    pd.DataFrame({
        "time_s":                  t,
        "true_wrist_X_cm":         true[:, 0],
        "true_wrist_Y_cm":         true[:, 1],
        "true_wrist_Z_cm":         true[:, 2],
        "true_shoulder_elv_deg":   true[:, 4],
        "true_shoulder_rot_deg":   true[:, 5],
        "true_elbow_flexion_deg":  true[:, 6],
        "pred_wrist_X_cm":         pred[:, 0],
        "pred_wrist_Y_cm":         pred[:, 1],
        "pred_wrist_Z_cm":         pred[:, 2],
        "pred_shoulder_elv_deg":   pred[:, 4],
        "pred_shoulder_rot_deg":   pred[:, 5],
        "pred_elbow_flexion_deg":  pred[:, 6],
        "l2_distance_cm":          l2,
    }).to_csv(os.path.join(CENTEROUT_DIR, f"results_{direction}.csv"), index=False)

    # Per-direction 7-panel time series
    plot_cols = [
        (true[:, 4], pred[:, 4], "Shoulder elevation (deg)", sh_elv_rmse),
        (true[:, 5], pred[:, 5], "Shoulder rotation (deg)",  sh_rot_rmse),
        (true[:, 6], pred[:, 6], "Elbow flexion (deg)",      elbow_rmse),
        (true[:, 0], pred[:, 0], "Wrist X (cm)",
         np.sqrt(np.mean((pred[:, 0]-true[:, 0])**2))),
        (true[:, 1], pred[:, 1], "Wrist Y (cm)",
         np.sqrt(np.mean((pred[:, 1]-true[:, 1])**2))),
        (true[:, 2], pred[:, 2], "Wrist Z (cm)",
         np.sqrt(np.mean((pred[:, 2]-true[:, 2])**2))),
    ]

    fig, axes = plt.subplots(7, 1, figsize=(11, 20), sharex=True,
                             gridspec_kw={"hspace": 0.45})
    for ax, (tv, pv, ylabel, col_rmse) in zip(axes[:6], plot_cols):
        ax.plot(t, tv, c=BLACK,   linewidth=1.8, label="Ground truth", zorder=3)
        ax.plot(t, pv, c=CRIMSON, linewidth=1.5, linestyle="--",
                label="Predicted", zorder=2)
        ax.set_ylabel(ylabel, fontsize=9, labelpad=4)
        ax.text(0.01, 0.93, f"RMSE: {col_rmse:.3f}",
                transform=ax.transAxes, fontsize=8, va='top', color='dimgray',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)

    axes[6].fill_between(t, 0, l2, color=ORANGE, alpha=0.25)
    axes[6].plot(t, l2, c=ORANGE, linewidth=1.5)
    axes[6].axhline(mean_l2, c=ORANGE, linewidth=1.2, linestyle="--", alpha=0.9,
                    label=f"Mean: {mean_l2:.2f} cm")
    axes[6].set_ylabel("Wrist L2\ndistance (cm)", fontsize=9)
    axes[6].set_xlabel("Time (s)", fontsize=10)
    axes[6].legend(fontsize=8, loc="upper right", framealpha=0.7)
    axes[6].spines[['top', 'right']].set_visible(False)
    axes[6].tick_params(labelsize=8)

    fig.suptitle(
        f"Center-out: {direction.replace('_', ' ')}\n",
        fontsize=10, y=1.00
    )
    plt.tight_layout()
    plt.savefig(os.path.join(CENTEROUT_DIR, f"pred_vs_truth_{direction}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  elbow RMSE: {elbow_rmse:.2f}°  wrist RMSE: {wrist_rmse:.2f} cm  "
          f"mean L2: {mean_l2:.2f} cm")
    print()

# ============================================================
# PANEL B: side-by-side XY plane, reach phase only, centered
# ============================================================
REACH_START = 396
REACH_END   = 756   # 396 + 120 reach + 120 hold + 120 return

fig2, (ax_true, ax_pred) = plt.subplots(1, 2, figsize=(12, 6),
                                         sharey=True, sharex=True)

for direction in DIRECTIONS_ORDERED:
    if direction not in all_true_xyz:
        continue
    color = DIR_COLORS.get(direction, "gray")

    # Trim to reach phase
    true_xyz = all_true_xyz[direction][REACH_START:REACH_END]
    pred_xyz = all_pred_xyz[direction][REACH_START:REACH_END]

    # Center at start of reach (common origin for all directions)
    true_rel = true_xyz - all_true_xyz[direction][REACH_START]
    pred_rel = pred_xyz - all_pred_xyz[direction][REACH_START]

    ax_true.plot(true_rel[:, 0], true_rel[:, 1], c=color,
                 linewidth=2.0, alpha=0.85, label=direction.replace("_", " "))
    ax_true.scatter(0, 0, c=color, s=40, zorder=5,
                    marker='o', edgecolors='black', linewidth=0.5)

    ax_pred.plot(pred_rel[:, 0], pred_rel[:, 1], c=color,
                 linewidth=2.0, alpha=0.85, linestyle="--")
    ax_pred.scatter(pred_rel[0, 0], pred_rel[0, 1], c=color, s=40, zorder=5,
                    marker='o', edgecolors='black', linewidth=0.5)

# Draw origin cross on both panels
for ax in [ax_true, ax_pred]:
    ax.axhline(0, c='black', linewidth=0.5, alpha=0.3)
    ax.axvline(0, c='black', linewidth=0.5, alpha=0.3)

for ax, title in [(ax_true, "Truth"), (ax_pred, "Predicted")]:
    ax.set_xlabel("X (cm)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_aspect('equal')

ax_true.set_ylabel("Y (cm)", fontsize=11)
ax_true.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.7,
               title="Direction", title_fontsize=8)

fig2.suptitle(
    "Center-out reach trajectories\n",
    fontsize=10, y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(CENTEROUT_DIR, "panel_b_trajectories.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved panel_b_trajectories.png")

# ============================================================
# TERMINAL SUMMARY
# ============================================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(CENTEROUT_DIR, "summary_all_directions.csv"), index=False
)

print()
print("="*65)
print(f"{'Direction':<20} {'sh_elv':>7} {'sh_rot':>7} {'elbow':>7} "
      f"{'wrist':>8} {'L2':>8}")
print(f"{'':20} {'RMSE°':>7} {'RMSE°':>7} {'RMSE°':>7} "
      f"{'RMSE cm':>8} {'mean cm':>8}")
print("-"*65)
for row in summary_rows:
    print(f"{row['direction']:<20} "
          f"{row['sh_elv_rmse_deg']:>7.2f} "
          f"{row['sh_rot_rmse_deg']:>7.2f} "
          f"{row['elbow_rmse_deg']:>7.2f} "
          f"{row['wrist_rmse_cm']:>8.2f} "
          f"{row['mean_l2_cm']:>8.2f}")
print("-"*65)
means = summary_df.mean(numeric_only=True)
print(f"{'MEAN':<20} "
      f"{means['sh_elv_rmse_deg']:>7.2f} "
      f"{means['sh_rot_rmse_deg']:>7.2f} "
      f"{means['elbow_rmse_deg']:>7.2f} "
      f"{means['wrist_rmse_cm']:>8.2f} "
      f"{means['mean_l2_cm']:>8.2f}")
print("="*65)