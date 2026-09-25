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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from experiment import load_manifest, resolve_artifact, set_artifact
from paths import REPO_DIR, PREDICTIONS_DIR, FIGURES_DIR, EXPERIMENT_CONFIG
sys.path.insert(0, REPO_DIR)

from utils.visualize_sample import get_shoulder_elbow_wrist_loc
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset

PROPRIOCEPTION_CONFIG = EXPERIMENT_CONFIG.get("proprioception", {})
MODEL_FAMILY = str(PROPRIOCEPTION_CONFIG.get("model_family", "extended")).lower()
COEF_SEED = int(PROPRIOCEPTION_CONFIG.get("coefficient_seed", 0))
TRAIN_SEED = int(PROPRIOCEPTION_CONFIG.get("training_seed", 9))

configured_model_path = PROPRIOCEPTION_CONFIG.get("model_path")
if MODEL_FAMILY == "extended":
    experiment_dir = (
        "experiment_causal_flag-pcr_optimized_linear_extended_5_5_"
        "letter_reconstruction_joints"
    )
elif MODEL_FAMILY == "main":
    experiment_dir = (
        "experiment_causal_flag-pcr_optimized_linear_5_5_"
        "letter_reconstruction_joints"
    )
    if (COEF_SEED, TRAIN_SEED) != (0, 9):
        raise ValueError("The main pretrained model is available only as model 0_9")
else:
    raise ValueError("proprioception.model_family must be 'main' or 'extended'")

if configured_model_path:
    MODEL_PATH = os.path.expanduser(str(configured_model_path))
    if not os.path.isabs(MODEL_PATH):
        MODEL_PATH = os.path.join(REPO_DIR, MODEL_PATH)
    MODEL_PATH = os.path.abspath(MODEL_PATH)
else:
    MODEL_PATH = os.path.join(
        REPO_DIR, "trained_models", experiment_dir,
        f"spatiotemporal_4_8-8-32-64_7171_{COEF_SEED}_{TRAIN_SEED}",
    )
if not os.path.isfile(os.path.join(MODEL_PATH, "config.yaml")):
    raise FileNotFoundError(
        f"Selected pretrained model is not installed: {MODEL_PATH}"
    )

BLACK   = "#1a1a1a"
CRIMSON = "#c0392b"
ORANGE  = "#e67e22"

# Load model once
print("Loading pretrained model...")
with open(os.path.join(MODEL_PATH, "config.yaml"), "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
model_config = {k: parse_config_value(v) for k, v in model_config.items()}
print("Model config loaded.")

manifest = load_manifest()
spindle_files = [(item["id"], resolve_artifact(item["spindle_data"]))
                 for item in manifest["trajectories"]]
if not spindle_files:
    raise ValueError("The experiment manifest contains no trajectories.")
trajectory_ids = [item[0] for item in spindle_files]
colors = plt.cm.hsv(np.linspace(0, 0.9, len(trajectory_ids)))
direction_colors = dict(zip(trajectory_ids, colors))

print(f"Found {len(spindle_files)} directions:")
for trajectory_id, _ in spindle_files:
    print(f"  {trajectory_id}")
print()

all_true_xyz = {}
all_pred_xyz = {}
summary_rows = []

for direction, sp_path in spindle_files:
    print(f"Running inference: {direction}")

    # Load data
    sp_data      = np.load(sp_path, allow_pickle=True)
    chunk_data   = sp_data['firing_rates'].astype(np.float32)  # (1,10,25,1152)
    joint_angles = sp_data['joint_angles']                     # (1152,7) degrees
    t = sp_data['times'].astype(np.float64)
    time_steps = len(t)

    # Prefer marker positions extracted from the selected OpenSim model.
    # The analytic fallback supports older spindle files.
    if 'wrist_xyz_world' in sp_data:
        wrist_loc = sp_data['wrist_xyz_world']
    else:
        labels_for_fk = np.zeros((time_steps, 7), dtype=np.float32)
        labels_for_fk[:, 3:7] = joint_angles[:, :4]
        _, _, wrist_loc = get_shoulder_elbow_wrist_loc(labels_for_fk)

    labels = np.zeros((1, time_steps, 7), dtype=np.float32)
    labels[0, :, 0:3] = wrist_loc
    labels[0, :, 3]   = joint_angles[:, 0]
    labels[0, :, 4]   = joint_angles[:, 1]
    labels[0, :, 5]   = joint_angles[:, 2]
    labels[0, :, 6]   = joint_angles[:, 3]

    # Write temp HDF5 and run inference
    tmp_hdf5 = os.path.join(PREDICTIONS_DIR, f"_tmp_{direction}.hdf5")
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
    }).to_csv(os.path.join(PREDICTIONS_DIR, f"{direction}.csv"), index=False)
    set_artifact(direction, "predictions", os.path.join(PREDICTIONS_DIR, f"{direction}.csv"))

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
    prediction_figure = os.path.join(FIGURES_DIR, f"pred_vs_truth_{direction}.png")
    plt.savefig(prediction_figure,
                dpi=150, bbox_inches="tight")
    set_artifact(direction, "prediction_figure", prediction_figure)
    plt.close()

    print(f"  elbow RMSE: {elbow_rmse:.2f}°  wrist RMSE: {wrist_rmse:.2f} cm  "
          f"mean L2: {mean_l2:.2f} cm")
    print()

# ============================================================
# PANEL B: side-by-side XY plane, reach phase only, centered
# ============================================================
fig2, (ax_true, ax_pred) = plt.subplots(1, 2, figsize=(12, 6),
                                         sharey=True, sharex=True)

for direction in trajectory_ids:
    if direction not in all_true_xyz:
        continue
    color = direction_colors[direction]

    true_xyz = all_true_xyz[direction]
    pred_xyz = all_pred_xyz[direction]

    # Center at start of reach (common origin for all directions)
    true_rel = true_xyz - true_xyz[0]
    pred_rel = pred_xyz - pred_xyz[0]

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
plt.savefig(os.path.join(FIGURES_DIR, "trajectory_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved panel_b_trajectories.png")

# ============================================================
# TERMINAL SUMMARY
# ============================================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(PREDICTIONS_DIR, "summary.csv"), index=False
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
