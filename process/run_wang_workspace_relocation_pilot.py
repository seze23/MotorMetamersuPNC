"""Matched Wang-grid pilot with original versus poster-like shoulder workspace."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import process.run_poster_to_wang_posture_bridge as bridge
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset

SOURCE = ROOT / "outputs/wang_grid_full_scale_refined/ik"
OUT = ROOT / "dataexp/poster_bias_investigation/wang_workspace_relocation_pilot"
TARGETS = ("r00_c00", "r00_c10", "r00_c19", "r02_c09", "r02_c10", "r04_c00", "r04_c10", "r04_c19")
POSTER_OFFSET = np.array([45.0-27.729233, 70.0-26.808939, 25.0-45.29328, 0, 0, 0, 0], dtype=np.float32)
LATE = np.arange(bridge.PEAK + round(.9*240), bridge.PEAK + round(1.15*240))


def make_trajectories():
    trajectories, identifiers = [], []
    for condition, offset in (("Original Wang workspace", np.zeros(7)), ("Poster-like shoulder workspace", POSTER_OFFSET)):
        for target in TARGETS:
            joints = np.load(SOURCE / f"{target}.npz")["joint_angles"].astype(np.float32) + offset
            trajectories.append(joints); identifiers.append((condition, target))
    return np.stack(trajectories), identifiers


def predict(rates, labels, coefficient_seed, training_seed):
    temp = OUT / f"_tmp_{coefficient_seed}_{training_seed}.hdf5"
    with h5py.File(temp, "w") as stream:
        stream.create_dataset("data", data=rates); stream.create_dataset("labels", data=labels)
    name = f"{coefficient_seed}_{training_seed}"
    model = bridge.MODELS / f"spatiotemporal_4_8-8-32-64_7171_{name}"
    config = yaml.safe_load((model / "config.yaml").read_text())
    config = {key: parse_config_value(value) for key, value in config.items()}; config["padding_mode"] = "zeros"
    dataset = SpindleDataset(str(temp), dataset_type="test", key="spindle_info", task="letter_reconstruction_joints", aclass=None, need_muscles=False, new_size=config["input_shape"][-1])
    tester = load_model(config, str(model), "letter_reconstruction_joints", torch.device("cpu"), dataset, causal=True, save_dir=str(ROOT)); tester.batch_size = min(25, len(labels))
    prediction, _ = tester.get_predictions(); temp.unlink()
    return name, prediction.detach().cpu().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trajectories, identifiers = make_trajectories(); labels = bridge.labels_from_trajectories(trajectories)
    cache = OUT / "corrected_muscle_lengths.npy"
    if cache.exists(): lengths = np.load(cache)
    else:
        lengths = bridge.extract_lengths(trajectories); np.save(cache, lengths)
    rows = []
    for coefficient_seed in range(5):
        rates = bridge.rates_from_lengths(lengths, coefficient_seed)
        for training_seed in (0, 1, 2, 9):
            model, prediction = predict(rates, labels, coefficient_seed, training_seed)
            for index, (condition, target) in enumerate(identifiers):
                raw = prediction[index, LATE, :3].mean(0) - labels[index, LATE, :3].mean(0)
                rows.append({"model": model, "coefficient_seed": coefficient_seed, "training_seed": training_seed,
                             "condition": condition, "target": target,
                             "true_x": labels[index, LATE, 0].mean(), "true_y": labels[index, LATE, 1].mean(), "true_z": labels[index, LATE, 2].mean(),
                             "outward_cm": -raw[0], "forward_cm": -raw[1], "up_cm": raw[2], "l2_cm": np.linalg.norm(raw)})
            print("completed", model, flush=True)
    result = pd.DataFrame(rows); result.to_csv(OUT / "model_target_results.csv", index=False)
    summary = result.groupby("condition", sort=False).agg(mean_outward_cm=("outward_cm", "mean"), fraction_outward=("outward_cm", lambda value: float((value>0).mean())), mean_l2_cm=("l2_cm", "mean")).reset_index()
    target = result.groupby(["condition", "target"], sort=False).agg(mean_outward_cm=("outward_cm", "mean"), fraction_outward=("outward_cm", lambda value: float((value>0).mean()))).reset_index()
    summary.to_csv(OUT / "condition_summary.csv", index=False); target.to_csv(OUT / "target_summary.csv", index=False)
    print(summary.to_string(index=False)); print(target.to_string(index=False))


if __name__ == "__main__": main()
