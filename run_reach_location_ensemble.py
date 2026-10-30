"""Run the 72-location reach grid through the 20 matched proprioception models.

Deterministic path/IK/muscle stages are run once. Spindle signals are computed
once per coefficient seed, and all 72 reaches are inferred as one batch for
each matching training-seed checkpoint. Results are averaged online.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent
PROCESS = ROOT / "process"
for _entry in (str(ROOT), str(PROCESS)):
    while _entry in sys.path:
        sys.path.remove(_entry)
sys.path.insert(0, str(PROCESS))
sys.path.insert(0, str(ROOT))

# process.paths binds its configuration during import. Resolve the CLI config
# before importing any process module so subprocesses and this process agree.
_config_argument = "experiments/reach_location_grid.yaml"
if "--config" in sys.argv:
    _config_argument = sys.argv[sys.argv.index("--config") + 1]
_early_config = Path(_config_argument)
if not _early_config.is_absolute():
    _early_config = (ROOT / _early_config).resolve()
with _early_config.open(encoding="utf-8") as _handle:
    _early_experiment = yaml.safe_load(_handle)["experiment"]
os.environ["MOTOR_META_CONFIG"] = str(_early_config)
os.environ["MOTOR_META_EXPERIMENT"] = _early_experiment

from experiment import load_manifest, resolve_artifact
from inference.test_model_utils_new import load_model, parse_config_value
from mujoco_pipeline.spindles import (
    MOBL_OPTIMAL_FIBER_LENGTH_MM,
    firing_rates,
    spindle_coefficients,
)
from run_pipeline import Pipeline
from train.new_spindle_dataset import SpindleDataset
from utils.visualize_sample import get_shoulder_elbow_wrist_loc

COEFFICIENT_SEEDS = (0, 1, 2, 3, 4)
TRAINING_SEEDS = (0, 1, 2, 9)
PREDICTION_NAMES = (
    "wrist_X_cm", "wrist_Y_cm", "wrist_Z_cm", "elv_angle_deg",
    "shoulder_elv_deg", "shoulder_rot_deg", "elbow_flexion_deg",
)


def main_checkout():
    pointer = ROOT / ".git"
    if not pointer.is_file():
        return ROOT
    text = pointer.read_text(encoding="utf-8").strip()
    if not text.startswith("gitdir:"):
        return ROOT
    return Path(text.split(":", 1)[1].strip()).resolve().parents[2]


def checkpoint_path(coefficient_seed, training_seed):
    relative = Path(
        "trained_models/"
        "experiment_causal_flag-pcr_optimized_linear_extended_5_5_"
        "letter_reconstruction_joints"
    ) / f"spatiotemporal_4_8-8-32-64_7171_{coefficient_seed}_{training_seed}"
    for base in (ROOT, main_checkout()):
        candidate = base / relative
        if (candidate / "config.yaml").is_file() and (candidate / "model.ckpt").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing checkpoint {relative}")


def labels_from_joints(joint_angles):
    count = len(joint_angles)
    labels_for_fk = np.zeros((count, 7), dtype=np.float32)
    labels_for_fk[:, 3:7] = joint_angles[:, :4]
    _, _, wrist = get_shoulder_elbow_wrist_loc(labels_for_fk)
    labels = np.zeros((count, 7), dtype=np.float32)
    labels[:, :3] = wrist
    labels[:, 3:7] = joint_angles[:, :4]
    return labels


def build_seed_batch(trajectories, coefficient_seed, cache_dir):
    cache = cache_dir / f"spindle_seed_{coefficient_seed}.hdf5"
    if cache.is_file():
        return cache
    config, coefficients, sampled, muscles = spindle_coefficients(coefficient_seed, 5)
    all_rates, all_labels = [], []
    for index, trajectory in enumerate(trajectories, 1):
        muscle = np.load(resolve_artifact(trajectory["muscle_data"]), allow_pickle=True)
        rates, _, _ = firing_rates(
            muscle["fiber_lengths"], muscle["times"],
            MOBL_OPTIMAL_FIBER_LENGTH_MM, config, coefficients, sampled, muscles,
        )
        all_rates.append(rates[0])
        all_labels.append(labels_from_joints(muscle["joint_angles"]))
        print(f"  spindle seed {coefficient_seed}: {index}/{len(trajectories)}", flush=True)
    with h5py.File(cache, "w") as handle:
        handle.create_dataset("data", data=np.stack(all_rates), compression="gzip")
        handle.create_dataset("labels", data=np.stack(all_labels), compression="gzip")
    return cache


def infer_batch(batch_path, model_path, padding_mode):
    with (model_path / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = {key: parse_config_value(value) for key, value in config.items()}
    config["padding_mode"] = padding_mode
    dataset = SpindleDataset(
        str(batch_path), dataset_type="test", key="spindle_info",
        task="letter_reconstruction_joints", need_muscles=False,
        new_size=config["input_shape"][-1],
    )
    tester = load_model(
        config, str(model_path), "letter_reconstruction_joints",
        torch.device("cpu"), dataset, causal=True, save_dir=str(ROOT),
    )
    predictions, _ = tester.get_predictions()
    return predictions.detach().cpu().numpy()


def save_results(output_dir, trajectories, labels, mean, std, run_count):
    prediction_dir = output_dir / "ensemble_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    peak_rows = []
    for index, trajectory in enumerate(trajectories):
        muscle = np.load(resolve_artifact(trajectory["muscle_data"]), allow_pickle=True)
        times = muscle["times"]
        frame = pd.DataFrame({"time_s": times})
        for column, name in enumerate(PREDICTION_NAMES):
            frame[f"true_{name}"] = labels[index, :, column]
            frame[f"pred_{name}_mean"] = mean[index, :, column]
            frame[f"pred_{name}_std"] = std[index, :, column]
        error = np.linalg.norm(mean[index, :, :3] - labels[index, :, :3], axis=1)
        frame["ensemble_mean_l2_cm"] = error
        frame.to_csv(prediction_dir / f"{trajectory['id']}.csv", index=False)
        hold_before = round(1.65 * float(trajectory["sample_rate_hz"]))
        reach_count = round(0.50 * float(trajectory["sample_rate_hz"]))
        peak = hold_before + reach_count - 1
        onset = hold_before
        peak_rows.append({
            "trajectory": trajectory["id"],
            "direction_degrees": trajectory["direction_degrees"],
            "radius_cm": trajectory["radius_cm"],
            "true_x_rel_cm": labels[index, peak, 0] - labels[index, onset, 0],
            "true_y_rel_cm": labels[index, peak, 1] - labels[index, onset, 1],
            "pred_x_rel_cm": mean[index, peak, 0] - mean[index, onset, 0],
            "pred_y_rel_cm": mean[index, peak, 1] - mean[index, onset, 1],
            "peak_l2_cm": error[peak],
            "mean_l2_cm": error.mean(),
            "peak_prediction_std_cm": np.linalg.norm(std[index, peak, :3]),
            "runs_averaged": run_count,
        })
    peaks = pd.DataFrame(peak_rows)
    peaks.to_csv(prediction_dir / "peak_summary.csv", index=False)
    plot_heatmap(peaks, prediction_dir / "ensemble_distortion_heatmap.png", run_count)


def plot_heatmap(frame, output, run_count):
    fig, axis = plt.subplots(figsize=(8, 7))
    scatter = axis.scatter(
        frame["true_x_rel_cm"], frame["true_y_rel_cm"],
        c=frame["peak_l2_cm"], cmap="inferno", s=85,
        edgecolors="black", linewidths=0.4,
    )
    axis.set(xlabel="True X displacement (cm)", ylabel="True Y displacement (cm)")
    axis.set_title(f"Ensemble distortion ({run_count} matched models)", fontsize=13)
    axis.set_aspect("equal")
    axis.axhline(0, color="gray", linewidth=0.5)
    axis.axvline(0, color="gray", linewidth=0.5)
    fig.colorbar(scatter, ax=axis, label="Peak L2 error of mean prediction (cm)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path, skip_deterministic=False):
    pipeline = Pipeline(config_path)
    if not skip_deterministic:
        pipeline.run_through(through="muscle-lengths", start="path", display=False)
    os.environ["MOTOR_META_CONFIG"] = str(pipeline.config_path)
    os.environ["MOTOR_META_EXPERIMENT"] = pipeline.experiment
    # paths/experiment modules were imported above, so ensure they represent this config.
    import paths
    if Path(paths.CENTEROUT_DIR).resolve() != pipeline.output_dir.resolve():
        raise RuntimeError("Run this script in a fresh Python process for the selected experiment")
    trajectories = load_manifest()["trajectories"]
    if len(trajectories) != 72:
        raise ValueError(f"Expected 72 reach locations, found {len(trajectories)}")
    cache_dir = pipeline.output_dir / "ensemble_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    running_mean = running_m2 = labels = None
    run_count = 0
    for coefficient_seed in COEFFICIENT_SEEDS:
        batch = build_seed_batch(trajectories, coefficient_seed, cache_dir)
        if labels is None:
            with h5py.File(batch, "r") as handle:
                labels = handle["labels"][()]
        for training_seed in TRAINING_SEEDS:
            model_path = checkpoint_path(coefficient_seed, training_seed)
            print(f"Inferring matched model {coefficient_seed}_{training_seed}", flush=True)
            prediction = infer_batch(batch, model_path, "replicate").astype(np.float64)
            run_count += 1
            if running_mean is None:
                running_mean = prediction
                running_m2 = np.zeros_like(prediction)
            else:
                delta = prediction - running_mean
                running_mean += delta / run_count
                running_m2 += delta * (prediction - running_mean)
    std = np.sqrt(running_m2 / run_count)
    save_results(pipeline.output_dir, trajectories, labels, running_mean, std, run_count)
    metadata = {
        "coefficient_seeds": list(COEFFICIENT_SEEDS),
        "training_seeds": list(TRAINING_SEEDS),
        "matched_models": run_count,
        "locations": len(trajectories),
        "padding": "replicate",
    }
    with (pipeline.output_dir / "ensemble_predictions" / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved averaged results under {pipeline.output_dir / 'ensemble_predictions'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/reach_location_grid.yaml")
    parser.add_argument("--skip-deterministic", action="store_true")
    arguments = parser.parse_args()
    # Set these before importing process modules that bind configuration at import time.
    config = (ROOT / arguments.config).resolve() if not Path(arguments.config).is_absolute() else Path(arguments.config)
    with config.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)["experiment"]
    os.environ["MOTOR_META_CONFIG"] = str(config)
    os.environ["MOTOR_META_EXPERIMENT"] = experiment
    run(config, arguments.skip_deterministic)
