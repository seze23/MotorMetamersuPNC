"""
Corrected multi-seed inference: unlike multi_seed_inference.py (which fed
every checkpoint the same COEF_SEED=0 spindle data, confounding the
comparison for all 16 non-zero-COEF_SEED checkpoints), this version loads
the COEF_SEED-MATCHED spindle data for each checkpoint
(center_out_<direction>_spindles_coefseed<N>.npz, produced by
computefrcenterout_multiseed.py), giving a genuinely valid 20-checkpoint
comparison.

Run computefrcenterout_multiseed.py FIRST -- this script expects
center_out_<direction>_spindles_coefseed{0,1,2,3,4}.npz to already exist.

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout
  conda activate proprioception
  python3 multi_seed_inference_matched.py
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


def build_labels_for_direction(direction, coef_seed):
    """Load the COEF_SEED-matched spindle file and build (data, labels) for
    this direction. Labels are analytic-FK based on joint_angles, which are
    identical regardless of coef_seed (same underlying reach trajectory) --
    only the spindle firing-rate INPUT differs by coef_seed."""
    sp_path = os.path.join(CENTEROUT_DIR, f"center_out_{direction}_spindles_coefseed{coef_seed}.npz")
    if not os.path.exists(sp_path):
        raise FileNotFoundError(
            f"{sp_path} not found -- run computefrcenterout_multiseed.py first"
        )
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

    return chunk_data, labels


directions = sorted(set(
    os.path.basename(f).split("_spindles_coefseed")[0].replace("center_out_", "")
    for f in glob.glob(os.path.join(CENTEROUT_DIR, "center_out_*_spindles_coefseed0.npz"))
))
if not directions:
    raise FileNotFoundError(
        f"No center_out_*_spindles_coefseed0.npz found in {CENTEROUT_DIR} -- "
        "run computefrcenterout_multiseed.py first"
    )
print(f"Found {len(directions)} grid points.\n")

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

        print(f"=== Seed {seed_name} (matched to coefseed{coef_seed} spindle data) ===")
        with open(os.path.join(model_path, "config.yaml"), "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        model_config = {k: parse_config_value(v) for k, v in model_config.items()}

        seed_l2_all = []
        for direction in directions:
            chunk_data, labels = build_labels_for_direction(direction, coef_seed)
            true = labels[0]

            tmp_hdf5 = os.path.join(CENTEROUT_DIR, f"_tmp_matched_{seed_name}_{direction}.hdf5")
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
per_point_out = os.path.join(CENTEROUT_DIR, "multi_seed_results_matched.csv")
per_point_df.to_csv(per_point_out, index=False)
print(f"\nSaved {per_point_out} ({len(per_point_df)} rows)")

ranking_df = pd.DataFrame(per_seed_summary).sort_values("overall_mean_l2").reset_index(drop=True)
ranking_df["rank"] = ranking_df.index + 1
ranking_out = os.path.join(CENTEROUT_DIR, "multi_seed_ranking_matched.csv")
ranking_df.to_csv(ranking_out, index=False)
print(f"Saved {ranking_out}")

print("\n" + "=" * 60)
print("SEED RANKING, PROPERLY MATCHED (best to worst, mean L2)")
print("=" * 60)
for _, row in ranking_df.iterrows():
    marker = "  <-- deployed checkpoint" if row["seed"] == "0_9" else ""
    print(f"  #{int(row['rank']):2d}  seed {row['seed']:6s}  mean L2 = {row['overall_mean_l2']:.4f} cm{marker}")
