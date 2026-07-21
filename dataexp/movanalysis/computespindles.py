"""
Compute synthetic muscle spindle firing rates from extracted fiber lengths
using the lab's published linear transfer function and optimized coefficients.

This replicates the math in utils/spindle_FR_helper.py and
extract_data/generate_train_test_data.py exactly, using the lab's own
functions directly rather than reimplementing them.

Pipeline:
  fiber_lengths.npz (mm, getFiberLength*1000)
  -> normalize by optimal fiber lengths
  -> derive velocity + acceleration via np.gradient
  -> Savitzky-Golay smoothing on velocity (matches smooth_data.py)
  -> spindle transfer function (Ia and II)
  -> chunk_data shape (1, 10, 25, 1152) ready for the pretrained model

Run on the cluster:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions
  conda activate proprioception
  python3 computespindles.py
"""

import os
import sys
import yaml
import numpy as np
import h5py
from scipy.signal import savgol_filter

REPO_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions"
sys.path.insert(0, REPO_DIR)

from utils.spindle_FR_helper import normalize, load_coefficients, get_sampled_coefficients
from extract_data.generate_train_test_data import process_chunk

# --- Paths ---
NPZ_PATH    = os.path.join(REPO_DIR, "dataexp/elbow_fiber_lengths.npz")
CONFIG_PATH = os.path.join(REPO_DIR, "extract_data/configs/train_test_data_spindles_extended.yaml")
OUTPUT_NPZ  = os.path.join(REPO_DIR, "dataexp/elbow_spindle_firing_rates.npz")
OUTPUT_HDF5 = os.path.join(REPO_DIR, "dataexp/my_trial_final.hdf5")

SAMPLE_RATE = 240
dt = 1.0 / SAMPLE_RATE

MUSCLE_NAMES = [
    'CORB',    'DELT1',   'DELT2',    'DELT3',  'INFSP',
    'LAT1',    'LAT2',    'LAT3',     'PECM1',  'PECM2',
    'PECM3',   'SUBSC',   'SUPSP',    'TMAJ',   'TMIN',
    'ANC',     'BIClong', 'BICshort', 'BRA',    'BRD',
    'ECRL',    'PT',      'TRIlat',   'TRIlong','TRImed'
]

# --- Load fiber lengths ---
print("Loading fiber lengths...")
d = np.load(NPZ_PATH, allow_pickle=True)
fiber_lengths = d['fiber_lengths']  # (1152, 25) mm
times         = d['times']          # (1152,)
N             = fiber_lengths.shape[0]
assert N == 1152, f"Expected 1152 timepoints, got {N}"
print(f"  shape: {fiber_lengths.shape}")
print(f"  BIClong std: {fiber_lengths[:, MUSCLE_NAMES.index('BIClong')].std():.4f}mm")

if fiber_lengths[:, MUSCLE_NAMES.index('BIClong')].std() < 0.01:
    raise ValueError(
        "BIClong std is near zero -- fiber lengths are constant. "
        "Re-run extract_fiber_lengths_cluster.py with equilibrateMuscles() fix first."
    )

# --- Load config ---
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

optimal_mm = np.array(config["optimal_lengths"])  # (25,) mm

# Make coeff paths absolute
config["i_a_coeff_path"] = os.path.join(REPO_DIR, config["i_a_coeff_path"])
config["ii_coeff_path"]  = os.path.join(REPO_DIR, config["ii_coeff_path"])

# --- Build (1, 25, 1152) arrays in mm ---
# fiber_lengths is (N, 25) -- transpose and add batch dim
muscle_lengths_mm = fiber_lengths.T[np.newaxis, :, :]  # (1, 25, 1152)

# Derive velocity via gradient then smooth (matches smooth_data.py pipeline)
muscle_velocities_raw = np.gradient(muscle_lengths_mm, dt, axis=2)

# Savitzky-Golay smoothing on velocity -- window=31, polyorder=1
# matches smooth_data.py: savgol_filter(velocity, 31, 1, axis=2)
muscle_velocities = np.zeros_like(muscle_velocities_raw)
for m in range(25):
    muscle_velocities[0, m, :] = savgol_filter(
        muscle_velocities_raw[0, m, :], window_length=31, polyorder=1
    )

# Derive acceleration from smoothed velocity
muscle_accelerations = np.gradient(muscle_velocities, dt, axis=2)

muscle_lengths_mm    = muscle_lengths_mm.astype(np.float32)
muscle_velocities    = muscle_velocities.astype(np.float32)
muscle_accelerations = muscle_accelerations.astype(np.float32)

print(f"\nMuscle lengths (mm):      {muscle_lengths_mm.min():.2f} -> {muscle_lengths_mm.max():.2f}")
print(f"Muscle velocities (mm/s): {muscle_velocities.min():.2f} -> {muscle_velocities.max():.2f}")
print(f"  (EF3D reference: {-361.8:.1f} -> {361.7:.1f})")

# --- Load coefficients ---
muscles          = config["muscles"]
num_coefficients = [config["num_i_a"], config["num_ii"]]
coefficients     = {
    key: load_coefficients(config[key + "_coeff_path"])
    for key in ["i_a", "ii"]
}
sampled_coefficients = get_sampled_coefficients(
    config, num_coefficients, muscles, coefficients
)

# --- Normalize and compute spindle firing rates ---
print("\nNormalizing and computing spindle firing rates...")
data       = normalize(muscle_lengths_mm, muscle_velocities,
                       muscle_accelerations, config["optimal_lengths"])
chunk_data = process_chunk(
    data, coefficients, num_coefficients, muscles,
    chunk_size=1, sampled_coefficients=sampled_coefficients
)
chunk_data = chunk_data.astype(np.float32)

print(f"Spindle firing rates shape: {chunk_data.shape}")  # (1, 10, 25, 1152)
print(f"Spindle firing rates range: {chunk_data.min():.1f} -> {chunk_data.max():.1f}")
print(f"  (EF3D reference: 50.0 -> 180.0)")

# --- Save firing rates ---
np.savez(OUTPUT_NPZ,
         times=times,
         firing_rates=chunk_data,      # (1, 10, 25, 1152)
         muscle_names=np.array(MUSCLE_NAMES))
print(f"\nSaved firing rates -> {OUTPUT_NPZ}")

# --- Quick visualization: plot Ia firing rates for BIClong and TRIlat ---
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    bic_idx = MUSCLE_NAMES.index('BIClong')
    tri_idx = MUSCLE_NAMES.index('TRIlat')

    # Ia afferents (channels 0-4)
    for ch in range(5):
        axes[0, 0].plot(times, chunk_data[0, ch, bic_idx, :], alpha=0.6, linewidth=0.8)
        axes[0, 1].plot(times, chunk_data[0, ch, tri_idx, :], alpha=0.6, linewidth=0.8)

    # II afferents (channels 5-9)
    for ch in range(5, 10):
        axes[1, 0].plot(times, chunk_data[0, ch, bic_idx, :], alpha=0.6, linewidth=0.8)
        axes[1, 1].plot(times, chunk_data[0, ch, tri_idx, :], alpha=0.6, linewidth=0.8)

    axes[0, 0].set_ylabel("BIClong Ia FR (Hz)")
    axes[0, 1].set_ylabel("TRIlat Ia FR (Hz)")
    axes[1, 0].set_ylabel("BIClong II FR (Hz)")
    axes[1, 1].set_ylabel("TRIlat II FR (Hz)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[0, 0].set_title("BIClong - type Ia")
    axes[0, 1].set_title("TRIlat - type Ia")
    axes[1, 0].set_title("BIClong - type II")
    axes[1, 1].set_title("TRIlat - type II")

    plt.suptitle("Horizontal Elbow Sweep Synthetic Spindle FR", fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(REPO_DIR, "dataexp/elbow_spindle_fr.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved firing rate plot -> {fig_path}")
except Exception as e:
    print(f"Plot skipped: {e}")

print("\nDone. Next: run inferencetest.py pointed at my_trial_final.hdf5")
print("(Update inferencetest.py to load firing rates from elbow_spindle_firing_rates.npz")
print(" or use my_trial_final.hdf5 directly with SpindleDataset)")