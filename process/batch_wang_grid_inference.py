"""Batch all 100 Wang-grid targets through one checkpoint load."""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from experiment import load_manifest, resolve_artifact
from paths import EXPERIMENT_CONFIG, PREDICTIONS_DIR, REPO_DIR

sys.path.insert(0, REPO_DIR)
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset
from utils.visualize_sample import get_shoulder_elbow_wrist_loc


def model_path():
    configured = Path(os.path.expanduser(
        str(EXPERIMENT_CONFIG["proprioception"]["model_path"])
    ))
    return configured if configured.is_absolute() else Path(REPO_DIR) / configured


def main():
    selected_model = model_path().resolve()
    with (selected_model / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    config = {key: parse_config_value(value) for key, value in config.items()}
    config["padding_mode"] = str(
        EXPERIMENT_CONFIG["proprioception"].get("temporal_padding_mode", "zeros")
    ).lower()

    trajectories = load_manifest()["trajectories"]
    trajectory_ids = [item["id"] for item in trajectories]
    firing_rates, labels, times = [], [], []
    for item in trajectories:
        spindle = np.load(resolve_artifact(item["spindle_data"]))
        rates = spindle["firing_rates"].astype(np.float32)
        joint_angles = spindle["joint_angles"]
        count = len(joint_angles)
        fk_labels = np.zeros((count, 7), dtype=np.float32)
        fk_labels[:, 3:7] = joint_angles[:, :4]
        _, _, wrist = get_shoulder_elbow_wrist_loc(fk_labels)
        label = np.zeros((count, 7), dtype=np.float32)
        label[:, :3] = wrist
        label[:, 3:7] = joint_angles[:, :4]
        firing_rates.append(rates[0])
        labels.append(label)
        times.append(spindle["times"].astype(np.float64))

    data = np.stack(firing_rates)
    labels = np.stack(labels)
    temp = Path(PREDICTIONS_DIR) / "_tmp_wang_grid_batch.hdf5"
    with h5py.File(temp, "w") as stream:
        stream.create_dataset("data", data=data)
        stream.create_dataset("labels", data=labels)
    dataset = SpindleDataset(
        str(temp), dataset_type="test", key="spindle_info",
        task="letter_reconstruction_joints", aclass=None,
        need_muscles=False, new_size=config["input_shape"][-1],
    )
    tester = load_model(
        config, str(selected_model), "letter_reconstruction_joints",
        torch.device("cpu"), dataset, causal=True, save_dir=REPO_DIR,
    )
    tester.batch_size = 25  # exactly four batches; avoids dropped remainder
    predictions, _ = tester.get_predictions()
    predictions = predictions.cpu().detach().numpy()
    temp.unlink()
    if predictions.shape[:2] != labels.shape[:2]:
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match {labels.shape}"
        )

    for index, trajectory_id in enumerate(trajectory_ids):
        true = labels[index]
        pred = predictions[index]
        l2 = np.linalg.norm(pred[:, :3] - true[:, :3], axis=1)
        pd.DataFrame({
            "time_s": times[index],
            "true_wrist_X_cm": true[:, 0], "true_wrist_Y_cm": true[:, 1],
            "true_wrist_Z_cm": true[:, 2],
            "true_elv_angle_deg": true[:, 3],
            "true_shoulder_elv_deg": true[:, 4],
            "true_shoulder_rot_deg": true[:, 5],
            "true_elbow_flexion_deg": true[:, 6],
            "pred_wrist_X_cm": pred[:, 0], "pred_wrist_Y_cm": pred[:, 1],
            "pred_wrist_Z_cm": pred[:, 2],
            "pred_elv_angle_deg": pred[:, 3],
            "pred_shoulder_elv_deg": pred[:, 4],
            "pred_shoulder_rot_deg": pred[:, 5],
            "pred_elbow_flexion_deg": pred[:, 6],
            "l2_distance_cm": l2,
        }).to_csv(Path(PREDICTIONS_DIR) / f"{trajectory_id}.csv", index=False)
    print(f"Saved batched predictions for {len(trajectory_ids)} targets.")


if __name__ == "__main__":
    main()
