"""
Multi-seed inference: run all 20 spatiotemporal checkpoints (COEF_SEED in
0-4, TRAIN_SEED in 0,1,2,9 -- same experiment config as the deployed 0_9
checkpoint) on the EXISTING grid spindle data. Does not rerun stages 1-5
(spindle firing rates don't depend on which CNN checkpoint reads them
later), so this is just 20x stage-6 inference over the already-computed
80 points (original 8 directions + 72-point grid).

Exploratory pass, per researcher: if results are noisy/confusing, fall back
to the existing single-checkpoint (0_9) figures already validated. Not
assumed to be the final poster data yet.

For each (seed, grid point) records:
  - true and predicted (x, y, z) at the peak-reach frame (515 =
    N_HOLD_PRE + N_REACH - 1, point of maximum displacement from center,
    same convention already used in heatmap_plot.py / ablation eval)
  - per-direction summary RMSE (sh_elv, sh_rot, elbow, wrist, L2)

Also ranks all 20 seeds by overall mean L2, to check whether 0_9 (the
checkpoint the original paper identified as best on ITS task) is also best,
or near-best, on THIS task.

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout
  conda activate proprioception
  python3 multi_seed_inference.py
"""

import os
import sys
import glob

import h5py
import yaml
import numpy as np
import pandas as pd
import torch

REPO_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions"
CENTEROUT_DIR = os.path.join(REPO_DIR, "dataexp/centerout")
sys.path.insert(0, REPO_DIR)

from utils.visualize_sample import get_shoulder_elbow_wrist_loc
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset

COEF_SEEDS = [0, 1, 2, 3, 4]
TRAIN_SEEDS = [0, 1, 2, 9]
EXPERIMENT_DIR = "experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints"

TIME_STEPS = 1152
SAMPLE_RATE = 240
N_HOLD_PRE = 396
N_REACH = 120
PEAK_FRAME = N_HOLD_PRE + N_REACH - 1  # 515

spindle_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "center_out_*_spindles.npz")))
if not spindle_files:
    raise FileNotFoundError(f"No *_spindles.npz files in {CENTEROUT_DIR}")
print(f"Found {len(spindle_files)} grid points (directions).")

# Pre-load all spindle data + build labels once (identical across seeds --
# analytic-FK labels don't depend on which checkpoint reads them)
grid_data = {}
for sp_path in spindle_files:
    direction = (os.path.basename(sp_path)
                 .replace("center_out_", "").replace("_spindles.npz", ""))
    sp_data = np.load(sp_path, allow_pickle=True)
    chunk_data = sp_data['firing_rates'].astype(np.float32)
    joint_angles = sp_data['joint_angles']

    labels_for_fk = np.zeros((TIME_STEPS, 7), dtype=np.float32)
    labels_for_fk[:, 3] = joint_angles[:, 0]
    labels_for_fk[:, 4] = joint_angles[:, 1]
    labels_for_fk[:, 5] = joint_angles[:, 2]
    labels_for_fk[:, 6] = joint_angles[:, 3]
    _, _, wrist_loc = get_shoulder_elbow_wrist_loc(labels_for_fk)

    labels = np.zeros((1, TIME_STEPS, 7), dtype=np.float32)
    labels[0, :, 0:3] = wrist_loc
    labels[0, :, 3] = joint_angles[:, 0]
    labels[0, :, 4] = joint_angles[:, 1]
    labels[0, :, 5] = joint_angles[:, 2]
    labels[0, :, 6] = joint_angles[:, 3]

    grid_data[direction] = dict(chunk_data=chunk_data, labels=labels)

print(f"Pre-built labels for {len(grid_data)} grid points.\n")

per_point_rows = []
per_seed_summary = []

for coef_seed in COEF_SEEDS:
    for train_seed in TRAIN_SEEDS:
        seed_name = f"{coef_seed}_{train_seed}"
        model_path = os.path.join(
            REPO_DIR, "trained_models", EXPERIMENT_DIR,
            f"spatiotemporal_4_8-8-32-64_7171_{coef_seed}_{train_seed}",
        )
        if not os.path.exists(model_path):
            print(f"SKIP {seed_name}: {model_path} not found")
            continue

        print(f"=== Seed {seed_name} ===")
        with open(os.path.join(model_path, "config.yaml"), "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        model_config = {k: parse_config_value(v) for k, v in model_config.items()}

        seed_l2_all = []
        for direction, gd in grid_data.items():
            chunk_data = gd["chunk_data"]
            labels = gd["labels"]
            true = labels[0]

            tmp_hdf5 = os.path.join(CENTEROUT_DIR, f"_tmp_multiseed_{seed_name}_{direction}.hdf5")
            with h5py.File(tmp_hdf5, "w") as f:
                f.create_dataset("data", data=chunk_data)
                f.create_dataset("labels", data=labels)

            test_data = SpindleDataset(
                tmp_hdf5, dataset_type="test", key="spindle_info",
                task="letter_reconstruction_joints", aclass=None,
                need_muscles=False, new_size=model_config["input_shape"][-1],
            )
            device = torch.device("cpu")
            tester = load_model(
                model_config, model_path, "letter_reconstruction_joints",
                device, test_data, causal=True, save_dir=REPO_DIR,
            )
            predictions, _ = tester.get_predictions()
            pred = predictions[0].cpu().detach().numpy()
            os.remove(tmp_hdf5)

            l2 = np.sqrt(np.sum((pred[:, :3] - true[:, :3]) ** 2, axis=1))
            sh_elv_rmse = np.sqrt(np.mean((pred[:, 4] - true[:, 4]) ** 2))
            sh_rot_rmse = np.sqrt(np.mean((pred[:, 5] - true[:, 5]) ** 2))
            elbow_rmse = np.sqrt(np.mean((pred[:, 6] - true[:, 6]) ** 2))
            wrist_rmse = np.sqrt(np.mean((pred[:, :3] - true[:, :3]) ** 2))
            mean_l2 = l2.mean()
            seed_l2_all.append(mean_l2)

            per_point_rows.append(dict(
                seed=seed_name, coef_seed=coef_seed, train_seed=train_seed,
                direction=direction,
                true_x=true[PEAK_FRAME, 0], true_y=true[PEAK_FRAME, 1], true_z=true[PEAK_FRAME, 2],
                pred_x=pred[PEAK_FRAME, 0], pred_y=pred[PEAK_FRAME, 1], pred_z=pred[PEAK_FRAME, 2],
                l2_at_peak=l2[PEAK_FRAME],
                sh_elv_rmse=sh_elv_rmse, sh_rot_rmse=sh_rot_rmse,
                elbow_rmse=elbow_rmse, wrist_rmse=wrist_rmse, mean_l2=mean_l2,
            ))

        overall_mean_l2 = float(np.mean(seed_l2_all))
        per_seed_summary.append(dict(seed=seed_name, coef_seed=coef_seed,
                                      train_seed=train_seed, overall_mean_l2=overall_mean_l2))
        print(f"  overall mean L2 across {len(seed_l2_all)} points: {overall_mean_l2:.4f} cm")

per_point_df = pd.DataFrame(per_point_rows)
per_point_out = os.path.join(CENTEROUT_DIR, "multi_seed_results.csv")
per_point_df.to_csv(per_point_out, index=False)
print(f"\nSaved {per_point_out} ({len(per_point_df)} rows)")

ranking_df = pd.DataFrame(per_seed_summary).sort_values("overall_mean_l2").reset_index(drop=True)
ranking_df["rank"] = ranking_df.index + 1
ranking_out = os.path.join(CENTEROUT_DIR, "multi_seed_ranking.csv")
ranking_df.to_csv(ranking_out, index=False)

print(f"\nSaved {ranking_out}")
print("\n" + "=" * 60)
print("SEED RANKING (best to worst, by overall mean L2)")
print("=" * 60)
for _, row in ranking_df.iterrows():
    marker = "  <-- deployed checkpoint" if row["seed"] == "0_9" else ""
    print(f"  #{int(row['rank']):2d}  seed {row['seed']:6s}  mean L2 = {row['overall_mean_l2']:.4f} cm{marker}")
