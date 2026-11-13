"""Exact 20-model poster replay with a matched TMIN-only correction."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import io
import subprocess
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import opensim as osm
import pandas as pd
from scipy.signal import savgol_filter
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1e1dd51"
PREFIX = "dataexp/centerout/"
MODEL_OSIM = ROOT / "MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
MODELS = ROOT / "trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints"
OUT = ROOT / "dataexp" / "old_centerout_corrected_tmin_comparison"
COORDS = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]
TMIN = 14
PEAK = 515
TRAINING_SEEDS = (0, 1, 2, 9)

sys.path.insert(0, str(ROOT))
from extract_data.generate_train_test_data import process_chunk
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset
from utils.spindle_FR_helper import normalize, load_coefficients, get_sampled_coefficients
from utils.visualize_sample import get_shoulder_elbow_wrist_loc


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


def git_npz(path):
    return np.load(io.BytesIO(git_bytes(path)))


def coefficient_files():
    listing = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", COMMIT, "--", "dataexp/centerout"],
        cwd=ROOT, text=True,
    ).splitlines()
    result = {}
    for path in listing:
        if "_spindles_coefseed" not in path or not path.endswith(".npz"):
            continue
        seed = int(path.rsplit("coefseed", 1)[1].split(".", 1)[0])
        result.setdefault(seed, []).append(path)
    for seed in result:
        result[seed] = sorted(result[seed])
    if set(result) != set(range(5)) or any(len(paths) != 80 for paths in result.values()):
        raise ValueError({seed: len(paths) for seed, paths in result.items()})
    return result


def corrected_lengths_from_bytes(payload):
    name, raw = payload
    source = np.load(io.BytesIO(raw))
    joints = source["joint_angles"]
    model = osm.Model(str(MODEL_OSIM))
    state = model.initSystem()
    muscle = model.updMuscles().get("TMIN")
    muscle.updGeometryPath().moveUpPathWrap(state, 2)
    coordinates = [model.getCoordinateSet().get(coord) for coord in COORDS]
    result = np.empty(len(joints), dtype=np.float32)
    cache = {}
    for index, angles in enumerate(joints):
        key = angles.tobytes()
        if key not in cache:
            for coordinate, degrees in zip(coordinates, angles):
                coordinate.setValue(state, np.deg2rad(float(degrees)))
            model.equilibrateMuscles(state)
            cache[key] = muscle.getFiberLength(state) * 1000
        result[index] = cache[key]
    return name, result


def corrected_length_cache(base_paths):
    cache_path = OUT / "multiring_corrected_tmin_lengths.npz"
    if cache_path.exists():
        saved = np.load(cache_path)
        return {key: saved[key] for key in saved.files}
    payloads = []
    for path in base_paths:
        name = Path(path).name.split("_spindles_coefseed", 1)[0].replace("center_out_", "")
        payloads.append((name, git_bytes(path)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        pairs = list(pool.map(corrected_lengths_from_bytes, payloads))
    result = dict(pairs)
    np.savez_compressed(cache_path, **result)
    return result


def tmin_rates_batch(lengths, times, coefficient_seed):
    config_path = ROOT / "extract_data/configs/train_test_data_spindles_extended.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["seed"] = coefficient_seed
    for key in ("i_a_coeff_path", "ii_coeff_path", "i_a_sampled_coeff_path", "ii_sampled_coeff_path"):
        if config.get(key) is not None:
            config[key] = str(ROOT / config[key])
    coefficients = {key: load_coefficients(config[key + "_coeff_path"]) for key in ("i_a", "ii")}
    counts = [config["num_i_a"], config["num_ii"]]
    sampled = get_sampled_coefficients(config, counts, config["muscles"], coefficients)
    full = np.zeros((len(lengths), 25, lengths.shape[1]), dtype=np.float32)
    full[:, TMIN, :] = lengths
    dt = float(np.median(np.diff(times)))
    velocity = np.zeros_like(full)
    raw = np.gradient(full, dt, axis=2)
    velocity[:, TMIN, :] = savgol_filter(raw[:, TMIN, :], 31, 1, axis=1)
    acceleration = np.gradient(velocity, dt, axis=2).astype(np.float32)
    normalized = normalize(full, velocity, acceleration, config["optimal_lengths"])
    return process_chunk(normalized, coefficients, counts, config["muscles"], len(lengths), sampled).astype(np.float32)


def infer_pair(coefficient_seed, training_seed, original, corrected, labels, directions):
    rates = np.concatenate([original, corrected], axis=0)
    paired_labels = np.concatenate([labels, labels], axis=0)
    temp = OUT / f"_tmp_multiring_{coefficient_seed}_{training_seed}.hdf5"
    with h5py.File(temp, "w") as stream:
        stream.create_dataset("data", data=rates)
        stream.create_dataset("labels", data=paired_labels)
    name = f"{coefficient_seed}_{training_seed}"
    model = MODELS / f"spatiotemporal_4_8-8-32-64_7171_{name}"
    config = yaml.safe_load((model / "config.yaml").read_text(encoding="utf-8"))
    config = {key: parse_config_value(value) for key, value in config.items()}
    config["padding_mode"] = "zeros"
    dataset = SpindleDataset(str(temp), dataset_type="test", key="spindle_info",
                             task="letter_reconstruction_joints", aclass=None,
                             need_muscles=False, new_size=config["input_shape"][-1])
    tester = load_model(config, str(model), "letter_reconstruction_joints",
                        torch.device("cpu"), dataset, causal=True, save_dir=str(ROOT))
    tester.batch_size = 40
    prediction, _ = tester.get_predictions()
    prediction = prediction.detach().cpu().numpy()
    temp.unlink()
    rows = []
    count = len(directions)
    for condition, offset in (("Archived poster TMIN", 0), ("Corrected TMIN only", count)):
        for index, direction in enumerate(directions):
            pred = prediction[offset + index, PEAK, :3]
            true = labels[index, PEAK, :3]
            raw = pred - true
            rows.append({
                "seed": name, "coef_seed": coefficient_seed, "train_seed": training_seed,
                "condition": condition, "direction": direction,
                "true_x": true[0], "true_y": true[1], "true_z": true[2],
                "pred_x": pred[0], "pred_y": pred[1], "pred_z": pred[2],
                "body_right_cm": -raw[0], "body_forward_cm": -raw[1],
                "body_up_cm": raw[2], "l2_at_peak": np.linalg.norm(raw),
            })
    return rows


def plot_comparison(result):
    colors = {"Archived poster TMIN": "#8a008a", "Corrected TMIN only": "#2678b8"}
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.3), sharex=True, sharey=True)
    for ax, condition in zip(axes, colors):
        data = result[result.condition == condition]
        for _, row in data.iterrows():
            ax.annotate("", xy=(row.true_x, row.true_y), xytext=(row.pred_x, row.pred_y),
                        arrowprops=dict(arrowstyle="->", color="#cfcfcf", lw=.75, alpha=.5, mutation_scale=8))
        grouped = data.groupby("direction", sort=False)
        for _, group in grouped:
            true_x, true_y = group.true_x.iloc[0], group.true_y.iloc[0]
            ax.annotate("", xy=(true_x, true_y), xytext=(group.pred_x.mean(), group.pred_y.mean()),
                        arrowprops=dict(arrowstyle="->", color=colors[condition], lw=1.5, mutation_scale=13))
            ax.scatter(true_x, true_y, c=colors[condition], s=8, zorder=5)
        mean = data.body_right_cm.mean()
        target_means = grouped.body_right_cm.mean()
        ax.set_title(f"{condition}\nmean body-right bias {mean:+.2f} cm; {(target_means > 0).sum()}/80 target means outward", fontsize=13)
        ax.set_xlabel("Archived analytic X (cm)"); ax.set_ylabel("Archived analytic Y (cm)")
        ax.set_xlim(25, 57); ax.set_ylim(-20, 6.5); ax.set_aspect("equal")
        ax.grid(alpha=.18); ax.axhline(0, color="#888888", lw=.5)
    fig.suptitle("Historical multi-ring poster experiment: exact replay and TMIN-only correction\n"
                 "Gray = individual models; colored = 20-model mean; arrows retain notebook convention (predicted -> true)", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "multiring_poster_archived_vs_corrected_tmin.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "multiring_poster_archived_vs_corrected_tmin.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = coefficient_files()
    base_paths = files[0]
    corrected_lengths = corrected_length_cache(base_paths)
    all_rows = []
    integrity = []
    for coefficient_seed in range(5):
        original, labels, directions, times = [], [], [], None
        lengths = []
        for path in files[coefficient_seed]:
            source = git_npz(path)
            original.append(source["firing_rates"][0].astype(np.float32))
            joints = source["joint_angles"].astype(np.float32)
            label = np.zeros((len(joints), 7), dtype=np.float32)
            label[:, 3:7] = joints[:, :4]
            label[:, :3] = get_shoulder_elbow_wrist_loc(label)[2]
            labels.append(label)
            direction = Path(path).name.split("_spindles_coefseed", 1)[0].replace("center_out_", "")
            directions.append(direction)
            lengths.append(corrected_lengths[direction])
            times = source["times"]
        original = np.stack(original); labels = np.stack(labels); lengths = np.stack(lengths)
        replacement = tmin_rates_batch(lengths, times, coefficient_seed)
        corrected = original.copy(); corrected[:, :, TMIN, :] = replacement[:, :, TMIN, :]
        integrity.append({
            "coefficient_seed": coefficient_seed,
            "max_change_outside_tmin_hz": float(np.max(np.abs(corrected[:, :, np.arange(25) != TMIN] - original[:, :, np.arange(25) != TMIN]))),
            "corrected_tmin_min_mm": float(lengths.min()), "corrected_tmin_max_mm": float(lengths.max()),
        })
        for training_seed in TRAINING_SEEDS:
            all_rows.extend(infer_pair(coefficient_seed, training_seed, original, corrected, labels, directions))
            print("completed", coefficient_seed, training_seed, flush=True)
    result = pd.DataFrame(all_rows)
    result.to_csv(OUT / "multiring_archived_vs_corrected_results.csv", index=False)
    pd.DataFrame(integrity).to_csv(OUT / "multiring_integrity.csv", index=False)

    archived = pd.read_csv(io.BytesIO(git_bytes(PREFIX + "multi_seed_results_matched.csv")))
    original = result[result.condition == "Archived poster TMIN"]
    merged = original.merge(archived, on=["seed", "coef_seed", "train_seed", "direction"], suffixes=("_replay", "_saved"))
    max_replay_difference = max(float(np.max(np.abs(merged[f"{field}_replay"] - merged[f"{field}_saved"]))) for field in ("true_x", "true_y", "true_z", "pred_x", "pred_y", "pred_z"))
    summary = result.groupby("condition").agg(
        mean_body_right_cm=("body_right_cm", "mean"),
        model_target_fraction_outward=("body_right_cm", lambda x: float((x > 0).mean())),
        mean_l2_cm=("l2_at_peak", "mean"),
    ).reset_index()
    targets = result.groupby(["condition", "direction"]).body_right_cm.mean().reset_index()
    counts = targets.groupby("condition").body_right_cm.apply(lambda x: int((x > 0).sum()))
    summary["outward_target_means"] = summary.condition.map(counts)
    summary["max_saved_replay_difference"] = max_replay_difference
    summary.to_csv(OUT / "multiring_condition_summary.csv", index=False)
    print(summary.to_string(index=False))

    archived_mean = float(summary.loc[summary.condition == "Archived poster TMIN", "mean_body_right_cm"].iloc[0])
    corrected_mean = float(summary.loc[summary.condition == "Corrected TMIN only", "mean_body_right_cm"].iloc[0])
    if archived_mean > 0 and corrected_mean < archived_mean - 0.25:
        plot_comparison(result)
        print("TMIN causal criterion passed; comparison figure saved.")
    else:
        print("TMIN causal criterion did not pass; figure intentionally not created.")


if __name__ == "__main__":
    main()
