"""
Run inference on ONE randomly-chosen real movement trial from the held-out
EF3D dataset (real motion-capture reaches, not the synthetic center-out star)
and check whether the pred-vs-truth start/end "spikes" seen in the center-out
plots are a property of the frozen CNN itself, or an artifact specific to how
the center-out inputs were built.

Background / why this script exists
-------------------------------------
`centeroutinference.py` plots pred-vs-truth wrist X/Y/Z, shoulder elevation,
shoulder rotation, and elbow flexion for the *synthetic* center-out reach
inputs, and those plots show artifacts/spikes at the start and end of the
1152-frame window. Two competing explanations:

  (A) CNN-intrinsic: `SpatiotemporalNetworkCausal` (model/model_definitions.py)
      is only "causal" in its per-timestep fully-connected readout
      (`fc_per_time_step`). The convolutional stack underneath uses ordinary
      symmetric "SAME" zero-padding in time
      (`padding=((s_kernelsize-1)//2, (t_kernelsize-1)//2)`, applied to BOTH
      sides). That means the first/last few frames' receptive fields extend
      into synthetic zero-padding instead of real signal, which is a known
      source of edge degradation in "SAME"-padded conv stacks -- and would
      show up on ANY input, real or synthetic.
  (B) Not-yet-controlled-for: something specific to how the center-out inputs
      are constructed (the reach path's own onset/offset kinematics, the
      windowing done in `centeroutinference.py`/`generatereachpath.py`, or the
      IK/FK pipeline) introduces edge transients that the CNN is faithfully
      reproducing errors on, rather than the CNN failing at sequence
      boundaries per se.

Using a real EF3D trial instead of a synthetic center-out reach removes any
center-out-specific construction as a variable. If the same start/end
artifact appears here too, that is evidence for (A). This script also runs a
direct empirical test of (A): it tiles the same trial's spindle input 3x
along time (so the model sees real signal, not zero-padding, on both sides of
the *middle* copy) and compares the edge error of the middle copy to the edge
error of the single-copy run. If tiling makes the edge error collapse, the
artifact lives in the zero-padded receptive field, not in the input.

This script needs the SAME heavy, cluster-only dependencies as
`centeroutinference.py` (torch, h5py, the frozen checkpoint, and a full
`seze23/MotorMetamersuPNC` checkout) -- it will not run on a dependency-light
local clone. Point PROPRIO_REPO_DIR at that checkout (defaults to the cluster
path already used elsewhere in this repo).

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions
  conda activate proprioception
  python3 /path/to/ef3d_random_inference.py [--trial N] [--seed S] [--no-context-diagnostic]
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Output dir: defaults to a folder next to this script (works standalone on
# the cluster, no dependency on the local Mac workspace's paths.py, whose
# MODEL_PATH points at a Mac-only OpenSim install anyway). Override with
# EF3D_OUT_DIR if you want outputs somewhere else, e.g. alongside the
# center-out pipeline's dataexp/centerout/.
OUT_DIR = os.environ.get(
    "EF3D_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataexp", "ef3d_random"),
)
os.makedirs(OUT_DIR, exist_ok=True)

PROPRIO_REPO_DIR = os.environ.get(
    "PROPRIO_REPO_DIR", "/home/sydneyez/sydneyez/ProprioceptiveIllusions")
sys.path.insert(0, PROPRIO_REPO_DIR)

from utils.spindle_FR_helper import (          # noqa: E402
    load_coefficients, get_sampled_coefficients, normalize,
    clipped_spindle_transfer_function_coeffs,
)
from inference.test_model_utils_new import load_model, parse_config_value  # noqa: E402
from train.new_spindle_dataset import SpindleDataset                       # noqa: E402

COEF_SEED, TRAIN_SEED = 0, 9
N_AFF = 5
MODEL_PATH = os.path.join(
    PROPRIO_REPO_DIR,
    "trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints",
    f"spatiotemporal_4_8-8-32-64_7171_{COEF_SEED}_{TRAIN_SEED}",
)
TIME_STEPS = 1152
SAMPLE_RATE = 240
EDGE_FRAMES = 60  # ~0.25 s at 240 Hz; window used to quantify the "spike"

PLOT_COLS = [
    ("shoulder_elv", 4, "Shoulder elevation (deg)"),
    ("shoulder_rot", 5, "Shoulder rotation (deg)"),
    ("elbow_flexion", 6, "Elbow flexion (deg)"),
    ("wrist_x", 0, "Wrist X (cm)"),
    ("wrist_y", 1, "Wrist Y (cm)"),
    ("wrist_z", 2, "Wrist Z (cm)"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", type=int, default=None,
                    help="EF3D trial index to use; random if omitted")
    p.add_argument("--seed", type=int, default=None,
                    help="RNG seed for choosing the random trial (reproducibility)")
    p.add_argument("--context-diagnostic", dest="context_diagnostic",
                    action="store_true", default=True)
    p.add_argument("--no-context-diagnostic", dest="context_diagnostic",
                    action="store_false")
    return p.parse_args()


# ------------------------------------------------------------------
# Input construction: prefer the official preprocessed EF3D test file
# (matches inference/test_model.py's naming convention); fall back to
# building one trial on the fly from the raw data/EF3D.hdf5 using the
# exact same coefficient-sampling + spindle transfer function as
# extract_data/generate_train_test_data.py, so a missing preprocessed
# file doesn't block this diagnostic.
# ------------------------------------------------------------------

def find_preprocessed_ef3d(n_aff=N_AFF, seed=COEF_SEED):
    candidates = glob.glob(os.path.join(
        PROPRIO_REPO_DIR, "**",
        f"optimized_linear_extended_{seed}_{n_aff}_{n_aff}_EF3D.hdf5"),
        recursive=True)
    return candidates[0] if candidates else None


def pick_trial_from_preprocessed(path, trial_idx, rng):
    with h5py.File(path, "r") as f:
        n_trials = f["data"].shape[0]
        if trial_idx is None:
            trial_idx = int(rng.integers(n_trials))
        chunk = f["data"][trial_idx:trial_idx + 1].astype(np.float32)   # (1,10,25,1152)
        labels = f["labels"][trial_idx:trial_idx + 1].astype(np.float32)  # (1,1152,7)
    return trial_idx, chunk, labels


def build_trial_from_raw(trial_idx, rng, n_aff=N_AFF, coef_seed=COEF_SEED):
    """Reproduce extract_data/generate_train_test_data.py's process_chunk /
    process_data for a single EF3D trial, without writing the full 99-trial
    preprocessed file. Uses the same coefficient CSVs and the same
    deterministic afferent-sampling scheme (get_sampled_coefficients reads
    the already-saved `sampled_coefficients_{type}_{n_aff}_{seed}.csv` files
    when present, exactly as the original preprocessing did).
    """
    raw_path = os.path.join(PROPRIO_REPO_DIR, "data", "EF3D.hdf5")
    cfg_path = os.path.join(PROPRIO_REPO_DIR, "extract_data", "configs",
                             "train_test_data_spindles_extended.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    config["num_i_a"] = config["num_ii"] = n_aff
    config["seed"] = coef_seed
    np.random.seed(config["seed"])

    with h5py.File(raw_path, "r") as f:
        n_trials = f["muscle_lengths"].shape[0]
        if trial_idx is None:
            trial_idx = int(rng.integers(n_trials))
        lengths       = f["muscle_lengths"][trial_idx:trial_idx + 1]
        velocities    = f["muscle_velocities"][trial_idx:trial_idx + 1]
        accelerations = f["muscle_accelerations"][trial_idx:trial_idx + 1]
        coords = np.transpose(f["endeffector_coords"][trial_idx:trial_idx + 1], (0, 2, 1))
        joints = np.transpose(f["joint_coords"][trial_idx:trial_idx + 1], (0, 2, 1))

    muscles = config["muscles"]
    coefficients = {
        k: load_coefficients(os.path.join(PROPRIO_REPO_DIR, config[f"{k}_coeff_path"]))
        for k in ["i_a", "ii"]
    }
    # get_sampled_coefficients mutates config in place with the resolved
    # sampled-coefficient CSV paths; join to PROPRIO_REPO_DIR if relative.
    for k in ["i_a", "ii"]:
        p = config[f"{k}_coeff_path"]
        if not os.path.isabs(p):
            config[f"{k}_coeff_path"] = os.path.join(PROPRIO_REPO_DIR, p)
    num_coefficients = [config["num_i_a"], config["num_ii"]]
    sampled_coefficients = get_sampled_coefficients(config, num_coefficients, muscles, coefficients)

    data = normalize(lengths, velocities, accelerations, config["optimal_lengths"])
    T = data["lengths"].shape[2]
    chunk = np.zeros((1, sum(num_coefficients), len(muscles), T), dtype=np.float32)
    for muscle in muscles:
        for i, coeff_type in enumerate(["i_a", "ii"]):
            for j in range(num_coefficients[i]):
                idx = sum(num_coefficients[:i]) + j
                sampled_index = sampled_coefficients[coeff_type][muscle][j]
                coeffs = {
                    key: coefficients[coeff_type][muscle][key][sampled_index]
                    for key in ["k_l", "k_v", "e_v", "k_a", "k_c", "max_rate"]
                }
                chunk[0, idx, muscle, :] = clipped_spindle_transfer_function_coeffs(
                    data["lengths"][0, muscle, :],
                    data["velocities"][0, muscle, :],
                    data["accelerations"][0, muscle, :],
                    coeffs,
                )

    labels = np.concatenate((coords, joints), axis=2).astype(np.float32)  # (1, T, 7)
    return trial_idx, chunk, labels


def load_random_ef3d_trial(trial_idx, seed):
    rng = np.random.default_rng(seed)
    preprocessed = find_preprocessed_ef3d()
    if preprocessed is not None:
        print(f"Using official preprocessed EF3D test file: {preprocessed}")
        return pick_trial_from_preprocessed(preprocessed, trial_idx, rng)
    print("No preprocessed *_5_5_EF3D.hdf5 found -- building one trial on the "
          "fly from data/EF3D.hdf5 (same coefficient CSVs/sampling as "
          "extract_data/generate_train_test_data.py).")
    return build_trial_from_raw(trial_idx, rng)


# ------------------------------------------------------------------
# Model loading + inference
# ------------------------------------------------------------------

def load_config():
    with open(os.path.join(MODEL_PATH, "config.yaml"), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return {k: parse_config_value(v) for k, v in cfg.items()}


def run_inference(chunk, labels, device=torch.device("cpu")):
    """chunk: (1, C, 25, T) float32; labels: (1, T, 7) float32.
    Returns pred, true as (T, 7) numpy arrays.
    """
    tmp_hdf5 = os.path.join(OUT_DIR, "_tmp_ef3d_random.hdf5")
    with h5py.File(tmp_hdf5, "w") as f:
        f.create_dataset("data", data=chunk)
        f.create_dataset("labels", data=labels)

    model_config = load_config()
    test_data = SpindleDataset(
        tmp_hdf5, dataset_type="test", key="spindle_info",
        task="letter_reconstruction_joints", aclass=None, need_muscles=False,
        new_size=model_config["input_shape"][-1],
    )
    tester = load_model(
        model_config, MODEL_PATH, "letter_reconstruction_joints",
        device, test_data, causal=True, save_dir=PROPRIO_REPO_DIR,
    )
    predictions, _ = tester.get_predictions()
    pred = predictions[0].cpu().detach().numpy()
    os.remove(tmp_hdf5)
    return pred, labels[0], tester


def run_context_diagnostic(chunk, labels, n_reps=3):
    """Tile the trial n_reps times along time so the model sees real signal
    (not zero-padding) at the boundaries of the *middle* copy, then compare
    edge error there to the single-copy run. If the edge artifact is caused
    by the conv stack's symmetric zero-padding at sequence boundaries, it
    should shrink substantially in the middle copy, since that copy's
    boundaries are now flanked by real (repeated) signal instead of zeros.
    """
    tiled_chunk = np.concatenate([chunk] * n_reps, axis=3)
    tiled_labels = np.concatenate([labels] * n_reps, axis=1)
    T = chunk.shape[3]

    tmp_hdf5 = os.path.join(OUT_DIR, "_tmp_ef3d_random_tiled.hdf5")
    with h5py.File(tmp_hdf5, "w") as f:
        f.create_dataset("data", data=tiled_chunk)
        f.create_dataset("labels", data=tiled_labels)

    model_config = load_config()
    test_data = SpindleDataset(
        tmp_hdf5, dataset_type="test", key="spindle_info",
        task="letter_reconstruction_joints", aclass=None, need_muscles=False,
        new_size=model_config["input_shape"][-1],
    )
    tester = load_model(
        model_config, MODEL_PATH, "letter_reconstruction_joints",
        torch.device("cpu"), test_data, causal=True, save_dir=PROPRIO_REPO_DIR,
    )

    original_outtime = tester.model.outtime
    tester.model.outtime = n_reps * T
    try:
        predictions, _ = tester.get_predictions()
    except RuntimeError as e:
        print("Context-padding diagnostic could not run for this checkpoint "
              f"(likely t_stride != 1, so tiling doesn't line up cleanly): {e}")
        tester.model.outtime = original_outtime
        os.remove(tmp_hdf5)
        return None
    tester.model.outtime = original_outtime
    os.remove(tmp_hdf5)

    pred_tiled = predictions[0].cpu().detach().numpy()  # (n_reps*T, 7)
    middle = pred_tiled[T:2 * T]  # the copy flanked by real signal on both sides
    return middle


def edge_rmse(pred, true, k=EDGE_FRAMES):
    start = np.sqrt(np.mean((pred[:k] - true[:k]) ** 2, axis=0))
    end = np.sqrt(np.mean((pred[-k:] - true[-k:]) ** 2, axis=0))
    middle = np.sqrt(np.mean((pred[k:-k] - true[k:-k]) ** 2, axis=0))
    return start, end, middle


TRANSIENT_FRAMES = 20   # observed transient duration is ~15-20 frames, well
                        # inside EDGE_FRAMES=60 -- a plain RMSE over 60 frames
                        # is dominated by the ~40 settled-but-biased frames
                        # that sit alongside the transient in that window,
                        # which swamps out any real shrinkage in the transient
                        # itself. This isolates just the fast excursion.
SETTLE_WINDOW = (25, 55)  # frames used to estimate each run's own settled
                          # baseline, clearly past the transient


def transient_magnitude(pred, k=TRANSIENT_FRAMES, settle=SETTLE_WINDOW):
    """Peak |pred - own settled baseline| within the first/last k frames.

    Comparing this between the single-copy and tiled-context runs isolates
    the fast boundary excursion from the (much slower, much longer-lived)
    persistent offset/bias that a plain windowed RMSE conflates it with --
    the bias contributes equally to both runs and to every frame in the
    window, so it should mostly cancel out of *this* metric even though it
    dominates a 60-frame RMSE.
    """
    s0, s1 = settle
    start_baseline = np.median(pred[s0:s1], axis=0)
    end_baseline = np.median(pred[-s1:-s0], axis=0)
    start = np.max(np.abs(pred[:k] - start_baseline), axis=0)
    end = np.max(np.abs(pred[-k:] - end_baseline), axis=0)
    return start, end


def main():
    args = parse_args()
    trial_idx, chunk, labels = load_random_ef3d_trial(args.trial, args.seed)
    print(f"EF3D trial index: {trial_idx}  (chunk shape {chunk.shape}, "
          f"labels shape {labels.shape})")

    pred, true, tester = run_inference(chunk, labels)
    t = np.arange(pred.shape[0]) / SAMPLE_RATE

    start_rmse, end_rmse, mid_rmse = edge_rmse(pred, true)
    print("\nPer-channel RMSE -- first vs last vs middle "
          f"{EDGE_FRAMES}-frame windows:")
    print(f"{'channel':<18}{'start':>10}{'end':>10}{'middle':>10}")
    for name, idx, _ in PLOT_COLS:
        print(f"{name:<18}{start_rmse[idx]:>10.3f}{end_rmse[idx]:>10.3f}{mid_rmse[idx]:>10.3f}")

    # ---- full-trace + edge-zoom figure ----
    fig, axes = plt.subplots(len(PLOT_COLS), 3, figsize=(15, 2.4 * len(PLOT_COLS)))
    for row, (name, idx, ylabel) in enumerate(PLOT_COLS):
        full_ax, start_ax, end_ax = axes[row]
        for ax, sl, title in [
            (full_ax, slice(None), "full trial"),
            (start_ax, slice(0, EDGE_FRAMES), f"first {EDGE_FRAMES} frames"),
            (end_ax, slice(-EDGE_FRAMES, None), f"last {EDGE_FRAMES} frames"),
        ]:
            ax.plot(t[sl], true[sl, idx], c="#1a1a1a", lw=1.5, label="Truth")
            ax.plot(t[sl], pred[sl, idx], c="#c0392b", lw=1.2, ls="--", label="Predicted")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(title, fontsize=9)
        full_ax.set_ylabel(ylabel, fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"EF3D trial {trial_idx}: pred vs truth, full trace + edge zoom", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"ef3d_trial_{trial_idx}_pred_vs_truth.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out_path}")

    if not args.context_diagnostic:
        return

    print("\nRunning context-padding diagnostic (tiling the trial 3x so the "
          "model sees real signal, not zero-padding, at the boundaries of "
          "the middle copy)...")
    middle_pred = run_context_diagnostic(chunk, labels)
    if middle_pred is None:
        return

    mid_start_rmse, mid_end_rmse, mid_mid_rmse = edge_rmse(middle_pred, true)
    print(f"\n[Windowed RMSE, {EDGE_FRAMES}-frame -- includes the persistent "
          f"bias alongside the transient, so shrinkage here is diluted]")
    print(f"{'channel':<18}{'single-copy start':>18}{'tiled-middle start':>20}"
          f"{'single-copy end':>18}{'tiled-middle end':>18}")
    for name, idx, _ in PLOT_COLS:
        print(f"{name:<18}{start_rmse[idx]:>18.3f}{mid_start_rmse[idx]:>20.3f}"
              f"{end_rmse[idx]:>18.3f}{mid_end_rmse[idx]:>18.3f}")

    single_start_tm, single_end_tm = transient_magnitude(pred)
    tiled_start_tm, tiled_end_tm = transient_magnitude(middle_pred)
    print(f"\n[Isolated transient magnitude, peak deviation from own settled "
          f"baseline within {TRANSIENT_FRAMES} frames -- the real test of the "
          f"zero-padding hypothesis]")
    print(f"{'channel':<18}{'single-copy start':>18}{'tiled-middle start':>20}"
          f"{'single-copy end':>18}{'tiled-middle end':>18}")
    for name, idx, _ in PLOT_COLS:
        print(f"{name:<18}{single_start_tm[idx]:>18.3f}{tiled_start_tm[idx]:>20.3f}"
              f"{single_end_tm[idx]:>18.3f}{tiled_end_tm[idx]:>18.3f}")

    shrink_start = 1 - (tiled_start_tm.mean() / (single_start_tm.mean() + 1e-9))
    shrink_end = 1 - (tiled_end_tm.mean() / (single_end_tm.mean() + 1e-9))
    print(f"\nMean isolated-transient shrinkage when flanked by real context: "
          f"start {shrink_start*100:.1f}%, end {shrink_end*100:.1f}%")
    if shrink_start > 0.4 and shrink_end > 0.4:
        print("-> Edge error drops substantially with real flanking context: "
              "consistent with (A), the conv stack's symmetric zero-padding "
              "at sequence boundaries, not something specific to the "
              "center-out input construction.")
    else:
        print("-> Edge error does NOT collapse with real flanking context: "
              "the zero-padding hypothesis alone does not explain the "
              "artifact -- look harder at (B), something in how the input "
              "sequence itself is built.")

    fig, axes = plt.subplots(len(PLOT_COLS), 1, figsize=(9, 2.2 * len(PLOT_COLS)), sharex=True)
    k = EDGE_FRAMES
    for row, (name, idx, ylabel) in enumerate(PLOT_COLS):
        ax = axes[row]
        ax.plot(t[:k], true[:k, idx], c="#1a1a1a", lw=1.5, label="Truth")
        ax.plot(t[:k], pred[:k, idx], c="#c0392b", lw=1.3, ls="--", label="Pred (single copy)")
        ax.plot(t[:k], middle_pred[:k, idx], c="#2980b9", lw=1.3, ls=":",
                label="Pred (tiled, real context)")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(f"EF3D trial {trial_idx}: start-of-sequence, single-copy vs tiled-context", fontsize=10)
    plt.tight_layout()
    diag_path = os.path.join(OUT_DIR, f"ef3d_trial_{trial_idx}_context_diagnostic.png")
    plt.savefig(diag_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {diag_path}")


if __name__ == "__main__":
    main()
