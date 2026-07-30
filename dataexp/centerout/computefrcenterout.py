"""
Compute synthetic spindle firing rates for all center-out directions.
Reads center_out_<direction>.npz, writes:
  center_out_<direction>_spindles.npz
  spindles_<direction>.png  (Ia + II for BIClong, TRIlat, DELT1)

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
matplotlib.use('Agg')
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

MUSCLE_NAMES = [
    'CORB',    'DELT1',   'DELT2',    'DELT3',  'INFSP',
    'LAT1',    'LAT2',    'LAT3',     'PECM1',  'PECM2',
    'PECM3',   'SUBSC',   'SUPSP',    'TMAJ',   'TMIN',
    'ANC',     'BIClong', 'BICshort', 'BRA',    'BRD',
    'ECRL',    'PT',      'TRIlat',   'TRIlong','TRImed'
]

# Muscles to plot - BIClong (elbow flexor), TRIlat (elbow extensor), DELT1 (shoulder)
PLOT_MUSCLES = {
    'BIClong': MUSCLE_NAMES.index('BIClong'),
    'TRIlat':  MUSCLE_NAMES.index('TRIlat'),
    'DELT1':   MUSCLE_NAMES.index('DELT1'),
}

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

# Find all fiber length npz files (exclude spindle outputs)
npz_files = sorted([
    f for f in glob.glob(os.path.join(CENTEROUT_DIR, "center_out_*.npz"))
    if "_spindles" not in f
])

if not npz_files:
    raise FileNotFoundError(f"No center_out_*.npz files in {CENTEROUT_DIR}")

print(f"Found {len(npz_files)} directions:")
for f in npz_files:
    print(f"  {os.path.basename(f)}")
print()

ia_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 5))
ii_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 5))

for npz_path in npz_files:
    direction = os.path.basename(npz_path).replace("center_out_","").replace(".npz","")
    out_npz   = os.path.join(CENTEROUT_DIR, f"center_out_{direction}_spindles.npz")
    out_png   = os.path.join(CENTEROUT_DIR, f"spindles_{direction}.png")

    print(f"Processing: {direction}")

    d             = np.load(npz_path, allow_pickle=True)
    fiber_lengths = d['fiber_lengths']    # (1152, 25) mm
    joint_angles  = d['joint_angles']     # (1152, 7) degrees
    times         = d['times']
    # Pass through wrist/elbow world positions for inference script
    wrist_xyz_world = d['wrist_xyz_world'] if 'wrist_xyz_world' in d else None
    elbow_xyz_world = d['elbow_xyz_world'] if 'elbow_xyz_world' in d else None

    fl_mm   = fiber_lengths.T[np.newaxis, :, :].astype(np.float32)  # (1,25,1152)
    vel_raw = np.gradient(fl_mm, dt, axis=2)
    vel     = np.zeros_like(vel_raw)
    for m in range(25):
        vel[0, m, :] = savgol_filter(vel_raw[0, m, :], window_length=31, polyorder=1)
    acc   = np.gradient(vel, dt, axis=2).astype(np.float32)
    fl_mm = fl_mm.astype(np.float32)
    vel   = vel.astype(np.float32)

    print(f"  lengths: {fl_mm.min():.1f}->{fl_mm.max():.1f} mm  "
          f"vel: {vel.min():.1f}->{vel.max():.1f} mm/s")

    if np.abs(vel).max() > 500:
        print(f"  WARNING: extreme velocity ({np.abs(vel).max():.0f} mm/s) "
              f"- {direction} may be outside EF3D training distribution")

    data       = normalize(fl_mm, vel, acc, config["optimal_lengths"])
    chunk_data = process_chunk(data, coefficients, num_coefficients, muscles,
                               chunk_size=1, sampled_coefficients=sampled_coefficients)
    chunk_data = chunk_data.astype(np.float32)

    print(f"  firing rates: {chunk_data.min():.1f}->{chunk_data.max():.1f} Hz  "
          f"shape: {chunk_data.shape}")

    # Save spindle npz -- include wrist/elbow positions for downstream use
    save_kwargs = dict(
        times=times,
        firing_rates=chunk_data,
        joint_angles=joint_angles,
        direction=direction,
    )
    if wrist_xyz_world is not None:
        save_kwargs['wrist_xyz_world'] = wrist_xyz_world
    if elbow_xyz_world is not None:
        save_kwargs['elbow_xyz_world'] = elbow_xyz_world
    np.savez(out_npz, **save_kwargs)
    print(f"  Saved {os.path.basename(out_npz)}")

    # --- Spindle firing rate figure: 2 rows x 3 muscles ---
    t_plot    = np.arange(chunk_data.shape[3]) / SAMPLE_RATE
    n_muscles = len(PLOT_MUSCLES)

    fig, axes = plt.subplots(2, n_muscles, figsize=(5*n_muscles, 7), sharex=True,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.3})

    for col, (mname, midx) in enumerate(PLOT_MUSCLES.items()):
        for ch in range(5):
            axes[0, col].plot(t_plot, chunk_data[0, ch,   midx, :],
                              c=ia_colors[ch], alpha=0.8, linewidth=0.9)
            axes[1, col].plot(t_plot, chunk_data[0, ch+5, midx, :],
                              c=ii_colors[ch], alpha=0.8, linewidth=0.9)
        axes[0, col].set_title(f"{mname} (Ia)", fontsize=10)
        axes[1, col].set_title(f"{mname} (II)", fontsize=10)
        axes[1, col].set_xlabel("Time (s)", fontsize=9)

    axes[0, 0].set_ylabel("Firing rate (Hz)", fontsize=9)
    axes[1, 0].set_ylabel("Firing rate (Hz)", fontsize=9)

    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)

    handles = [plt.Line2D([0],[0], c=ia_colors[i], linewidth=1.2,
               label=f"Sample {i+1}") for i in range(5)]
    axes[0, 0].legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)

    fig.suptitle(
        f"Center-out: {direction.replace('_',' ')} - Spindle FR\n"
        f"peak vel: {np.abs(vel).max():.0f} mm/s   "
        f"FR range: {chunk_data.min():.1f}–{chunk_data.max():.1f} Hz",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {os.path.basename(out_png)}")
    print()

print(f"All done. {len(npz_files)} directions processed.")