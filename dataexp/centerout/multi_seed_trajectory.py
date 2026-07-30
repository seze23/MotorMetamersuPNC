"""
Multi-seed reach-phase trajectory data, for the truth-vs-predicted trajectory
poster figure. Scoped to the 8 canonical directions only (not all 72 grid
points -- a trajectory plot with 72 overlapping paths would be unreadable).

Reach phase only: frames 396:516 (N_HOLD_PRE to N_HOLD_PRE + N_REACH),
centered at reach onset (frame 396) -- same convention as the existing
panel_b_reach_only_trajectories.png, and same COEF_SEED-matched data
loading as multi_seed_inference_matched.py.

Run (after computefrcenterout_multiseed.py and multi_seed_inference_matched.py):
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout
  conda activate proprioception
  python3 multi_seed_trajectory.py
"""

import os
import sys

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

CANONICAL_DIRECTIONS = [
    "0_right", "45_fwd_right", "90_forward", "135_fwd_left",
    "180_left", "225_back_left", "270_backward", "315_back_right",
]

TIME_STEPS = 1152
N_HOLD_PRE = 396
N_REACH = 120
WINDOW_START = N_HOLD_PRE          # 396
WINDOW_END = N_HOLD_PRE + N_REACH  # 516


def build_labels_for_direction(direction, coef_seed):
    sp_path = os.path.join(CENTEROUT_DIR, f"center_out_{direction}_spindles_coefseed{coef_seed}.npz")
    if not os.path.exists(sp_path):
        raise FileNotFoundError(f"{sp_path} not found -- run computefrcenterout_multiseed.py first")
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


rows = []
for coef_seed in COEF_SEEDS:
    for train_seed in TRAIN_SEEDS:
        seed_name = f"{coef_seed}_{train_seed}"
        model_path = os.path.join(
            REPO_DIR, "trained_models", EXPERIMENT_DIR,
            f"spatiotemporal_4_8-8-32-64_7171_{coef_seed}_{train_seed}",
        )
        if not os.path.exists(model_path):
            print(f"SKIP {seed_name}: not found")
            continue

        print(f"=== Seed {seed_name} ===")
        with open(os.path.join(model_path, "config.yaml"), "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        model_config = {k: parse_config_value(v) for k, v in model_config.items()}

        for direction in CANONICAL_DIRECTIONS:
            chunk_data, labels = build_labels_for_direction(direction, coef_seed)
            true = labels[0]

            tmp_hdf5 = os.path.join(CENTEROUT_DIR, f"_tmp_traj_{seed_name}_{direction}.hdf5")
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

            # Reach-phase window, centered at reach onset (frame WINDOW_START)
            true_win = true[WINDOW_START:WINDOW_END, :3] - true[WINDOW_START, :3]
            pred_win = pred[WINDOW_START:WINDOW_END, :3] - pred[WINDOW_START, :3]

            for frame_idx in range(WINDOW_END - WINDOW_START):
                rows.append(dict(
                    seed=seed_name, coef_seed=coef_seed, train_seed=train_seed,
                    direction=direction, frame_idx=frame_idx,
                    true_x=true_win[frame_idx, 0], true_y=true_win[frame_idx, 1],
                    pred_x=pred_win[frame_idx, 0], pred_y=pred_win[frame_idx, 1],
                ))

        print(f"  done: {len(CANONICAL_DIRECTIONS)} directions")

traj_df = pd.DataFrame(rows)
out_csv = os.path.join(CENTEROUT_DIR, "multi_seed_trajectory.csv")
traj_df.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv} ({len(traj_df)} rows = {traj_df['seed'].nunique()} seeds x "
      f"{traj_df['direction'].nunique()} directions x {WINDOW_END - WINDOW_START} frames)")
