"""
Timing probe for computefrcenterout.py -- times the spindle computation for a
single already-extracted direction, reusing the existing center_out_*.npz
files from the completed pipeline run (no new extraction needed).

Run on the cluster:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout
  conda activate proprioception
  python3 spindle_timing_probe.py
"""

import os
import sys
import time
import yaml
import glob
import numpy as np
from scipy.signal import savgol_filter

from paths import CENTEROUT_DIR
REPO_DIR = os.environ.get(
    "PROPRIO_REPO_DIR", "/home/sydneyez/sydneyez/ProprioceptiveIllusions")
CONFIG_PATH = os.path.join(REPO_DIR, "extract_data/configs/train_test_data_spindles_extended.yaml")
sys.path.insert(0, REPO_DIR)
from utils.spindle_FR_helper import normalize, load_coefficients, get_sampled_coefficients
from extract_data.generate_train_test_data import process_chunk

SAMPLE_RATE = 240
dt = 1.0 / SAMPLE_RATE

t0 = time.time()
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
config["i_a_coeff_path"] = os.path.join(REPO_DIR, config["i_a_coeff_path"])
config["ii_coeff_path"] = os.path.join(REPO_DIR, config["ii_coeff_path"])
muscles = config["muscles"]
num_coefficients = [config["num_i_a"], config["num_ii"]]
coefficients = {key: load_coefficients(config[key + "_coeff_path"]) for key in ["i_a", "ii"]}
sampled_coefficients = get_sampled_coefficients(config, num_coefficients, muscles, coefficients)
t_setup = time.time() - t0
print(f"Config + coefficient load (once, fixed cost): {t_setup:.2f} s")

npz_files = sorted([f for f in glob.glob(os.path.join(CENTEROUT_DIR, "center_out_*.npz"))
                     if "_spindles" not in f])
if not npz_files:
    raise FileNotFoundError(f"No center_out_*.npz in {CENTEROUT_DIR}")

npz_path = npz_files[0]
direction = os.path.basename(npz_path).replace("center_out_", "").replace(".npz", "")
print(f"Probing with direction: {direction}\n")

t0 = time.time()
d = np.load(npz_path, allow_pickle=True)
fiber_lengths = d['fiber_lengths']
joint_angles = d['joint_angles']

fl_mm = fiber_lengths.T[np.newaxis, :, :].astype(np.float32)
vel_raw = np.gradient(fl_mm, dt, axis=2)
vel = np.zeros_like(vel_raw)
for m in range(25):
    vel[0, m, :] = savgol_filter(vel_raw[0, m, :], window_length=31, polyorder=1)
acc = np.gradient(vel, dt, axis=2).astype(np.float32)
fl_mm = fl_mm.astype(np.float32)
vel = vel.astype(np.float32)

data = normalize(fl_mm, vel, acc, config["optimal_lengths"])
chunk_data = process_chunk(data, coefficients, num_coefficients, muscles,
                           chunk_size=1, sampled_coefficients=sampled_coefficients)
t_per_direction = time.time() - t0
print(f"Per-direction spindle computation: {t_per_direction:.2f} s")
print(f"\nFor N new grid points, expected additional cost (fixed setup paid once):")
print(f"  N=72:  {t_setup:.1f}s + 72 x {t_per_direction:.2f}s = {t_setup + 72*t_per_direction:.1f} s total")
