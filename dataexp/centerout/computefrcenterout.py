"""
Compute synthetic spindle firing rates for all center-out directions.
Reads center_out_<direction>.npz, writes center_out_<direction>_spindles.npz

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions
  conda activate proprioception
  python3 computefrcenterout.py
"""

import os
import sys
import yaml
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

REPO_DIR      = "/home/sydneyez/sydneyez/ProprioceptiveIllusions"
CENTEROUT_DIR = os.path.join(REPO_DIR, "dataexp/centerout")
CONFIG_PATH   = os.path.join(REPO_DIR, "extract_data/configs/train_test_data_spindles_extended.yaml")

sys.path.insert(0, REPO_DIR)
from utils.spindle_FR_helper import normalize, load_coefficients, get_sampled_coefficients
from extract_data.generate_train_test_data import process_chunk

SAMPLE_RATE = 240
dt          = 1.0 / SAMPLE_RATE

# Load config and coefficients once
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
config["i_a_coeff_path"] = os.path.join(REPO_DIR, config["i_a_coeff_path"])
config["ii_coeff_path"]  = os.path.join(REPO_DIR, config["ii_coeff_path"])

muscles          = config["muscles"]
num_coefficients = [config["num_i_a"], config["num_ii"]]
coefficients     = {key: load_coefficients(config[key + "_coeff_path"])
                    for key in ["i_a", "ii"]}
sampled_coefficients = get_sampled_coefficients(
    config, num_coefficients, muscles, coefficients
)
print("Spindle coefficients loaded.")
print()

npz_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "center_out_*.npz")))
# Exclude already-computed spindle files
npz_files = [f for f in npz_files if "_spindles" not in f]

if not npz_files:
    raise FileNotFoundError(f"No center_out_*.npz files found in {CENTEROUT_DIR}")

print(f"Found {len(npz_files)} directions to process:")
for f in npz_files:
    print(f"  {os.path.basename(f)}")
print()

for npz_path in npz_files:
    direction = os.path.basename(npz_path).replace("center_out_","").replace(".npz","")
    out_path  = os.path.join(CENTEROUT_DIR, f"center_out_{direction}_spindles.npz")

    print(f"Processing: {direction}")

    d             = np.load(npz_path, allow_pickle=True)
    fiber_lengths = d['fiber_lengths']  # (1152, 25) mm
    joint_angles  = d['joint_angles']   # (1152, 7) degrees
    times         = d['times']

    # (1, 25, 1152) mm
    fl_mm = fiber_lengths.T[np.newaxis, :, :].astype(np.float32)

    # Velocity via gradient then Savitzky-Golay smooth (matches smooth_data.py)
    vel_raw = np.gradient(fl_mm, dt, axis=2)
    vel     = np.zeros_like(vel_raw)
    for m in range(25):
        vel[0, m, :] = savgol_filter(vel_raw[0, m, :], window_length=31, polyorder=1)

    acc = np.gradient(vel, dt, axis=2).astype(np.float32)
    fl_mm = fl_mm.astype(np.float32)
    vel   = vel.astype(np.float32)

    print(f"  lengths: {fl_mm.min():.1f}->{fl_mm.max():.1f} mm  "
          f"vel: {vel.min():.1f}->{vel.max():.1f} mm/s")

    data       = normalize(fl_mm, vel, acc, config["optimal_lengths"])
    chunk_data = process_chunk(data, coefficients, num_coefficients, muscles,
                               chunk_size=1, sampled_coefficients=sampled_coefficients)
    chunk_data = chunk_data.astype(np.float32)

    print(f"  firing rates: {chunk_data.min():.1f}->{chunk_data.max():.1f} Hz  "
          f"shape: {chunk_data.shape}")

    np.savez(out_path,
             times=times,
             firing_rates=chunk_data,   # (1, 10, 25, 1152)
             joint_angles=joint_angles,
             direction=direction)

    # --- Spindle firing rate plot: Ia and II for BIClong and TRIlat ---
    MUSCLE_NAMES = [
        'CORB',    'DELT1',   'DELT2',    'DELT3',  'INFSP',
        'LAT1',    'LAT2',    'LAT3',     'PECM1',  'PECM2',
        'PECM3',   'SUBSC',   'SUPSP',    'TMAJ',   'TMIN',
        'ANC',     'BIClong', 'BICshort', 'BRA',    'BRD',
        'ECRL',    'PT',      'TRIlat',   'TRIlong','TRImed'
    ]

matplotlib.use('Agg')

bic_idx = MUSCLE_NAMES.index('BIClong')
tri_idx = MUSCLE_NAMES.index('TRIlat')
t_plot  = np.arange(chunk_data.shape[3]) / SAMPLE_RATE

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True,
                        gridspec_kw={"hspace": 0.35, "wspace": 0.25})

ia_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 5))
ii_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 5))

for ch in range(5):
    axes[0, 0].plot(t_plot, chunk_data[0, ch,   bic_idx, :],
                    c=ia_colors[ch], alpha=0.8, linewidth=0.9,
                    label=f"sample {ch+1}" if ch == 0 else None)
    axes[0, 1].plot(t_plot, chunk_data[0, ch,   tri_idx, :],
                    c=ia_colors[ch], alpha=0.8, linewidth=0.9)
    axes[1, 0].plot(t_plot, chunk_data[0, ch+5, bic_idx, :],
                    c=ii_colors[ch], alpha=0.8, linewidth=0.9)
    axes[1, 1].plot(t_plot, chunk_data[0, ch+5, tri_idx, :],
                    c=ii_colors[ch], alpha=0.8, linewidth=0.9)

axes[0, 0].set_ylabel("Firing rate (Hz)", fontsize=9)
axes[1, 0].set_ylabel("Firing rate (Hz)", fontsize=9)
axes[1, 0].set_xlabel("Time (s)", fontsize=9)
axes[1, 1].set_xlabel("Time (s)", fontsize=9)
axes[0, 0].set_title("BIClong - type Ia", fontsize=10)
axes[0, 1].set_title("TRIlat - type Ia",  fontsize=10)
axes[1, 0].set_title("BIClong - type II", fontsize=10)
axes[1, 1].set_title("TRIlat - type II",  fontsize=10)

for ax in axes.flat:
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)

# 5-sample legend on top-left panel only
handles = [plt.Line2D([0], [0], c=ia_colors[i], linewidth=1.2,
                    label=f"Sample {i+1}") for i in range(5)]
axes[0, 0].legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)

fig.suptitle(
    f"Center-out {direction} spindle FR\n",
    fontsize=10, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(CENTEROUT_DIR, f"spindles_{direction}.png"),
            dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved {os.path.basename(out_path)}")
print()

print(f"All done. {len(npz_files)} spindle files written.")