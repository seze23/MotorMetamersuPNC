"""
Multi-seed version of computefrcenterout.py -- computes spindle firing rates
for all 5 COEF_SEED values (0-4), reusing the already-cached sampled
coefficient files (sampled_coefficients_{i_a,ii}_5_{seed}.csv, confirmed
present for all 5 seeds -- no fresh sampling needed).

Fiber lengths (center_out_<direction>.npz) do NOT depend on COEF_SEED --
only the spindle-model coefficient sampling does -- so this reuses the
existing fiber-length files unchanged, only re-deriving the firing-rate
step per seed.

Output naming: center_out_<direction>_spindles_coefseed<N>.npz (distinct
from the original center_out_<direction>_spindles.npz, which stays as the
seed-0 default used everywhere else in the pipeline -- this script does not
touch or overwrite it).

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout
  conda activate proprioception
  python3 computefrcenterout_multiseed.py
"""

import os
import sys
import copy
import glob

import yaml
import numpy as np
from scipy.signal import savgol_filter

REPO_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions"
CENTEROUT_DIR = os.path.join(REPO_DIR, "dataexp/centerout")
CONFIG_PATH = os.path.join(REPO_DIR, "extract_data/configs/train_test_data_spindles_extended.yaml")

sys.path.insert(0, REPO_DIR)
from utils.spindle_FR_helper import normalize, load_coefficients, get_sampled_coefficients
from extract_data.generate_train_test_data import process_chunk

SAMPLE_RATE = 240
dt = 1.0 / SAMPLE_RATE
COEF_SEEDS = [0, 1, 2, 3, 4]

with open(CONFIG_PATH) as f:
    base_config = yaml.safe_load(f)
base_config["i_a_coeff_path"] = os.path.join(REPO_DIR, base_config["i_a_coeff_path"])
base_config["ii_coeff_path"] = os.path.join(REPO_DIR, base_config["ii_coeff_path"])
muscles = base_config["muscles"]
num_coefficients = [base_config["num_i_a"], base_config["num_ii"]]
# Full coefficient pool is the same for every seed -- only the SAMPLED
# subset (drawn according to config["seed"]) differs.
coefficients = {key: load_coefficients(base_config[key + "_coeff_path"]) for key in ["i_a", "ii"]}

npz_files = sorted([
    f for f in glob.glob(os.path.join(CENTEROUT_DIR, "center_out_*.npz"))
    if "_spindles" not in f
])
if not npz_files:
    raise FileNotFoundError(f"No center_out_*.npz (fiber-length) files in {CENTEROUT_DIR}")
print(f"Found {len(npz_files)} directions x {len(COEF_SEEDS)} coefficient seeds to process.\n")

for coef_seed in COEF_SEEDS:
    print(f"{'='*60}\nCOEF_SEED = {coef_seed}\n{'='*60}")
    config = copy.deepcopy(base_config)
    config["seed"] = coef_seed  # this is what get_sampled_coefficients reads;
                                 # cached sampled_coefficients_*_5_{seed}.csv
                                 # already exists for all 5 seeds -- confirmed
                                 # via `find` before writing this script.
    sampled_coefficients = get_sampled_coefficients(config, num_coefficients, muscles, coefficients)

    for npz_path in npz_files:
        direction = os.path.basename(npz_path).replace("center_out_", "").replace(".npz", "")
        out_npz = os.path.join(CENTEROUT_DIR, f"center_out_{direction}_spindles_coefseed{coef_seed}.npz")

        d = np.load(npz_path, allow_pickle=True)
        fiber_lengths = d['fiber_lengths']
        joint_angles = d['joint_angles']
        times = d['times']
        wrist_xyz_world = d['wrist_xyz_world'] if 'wrist_xyz_world' in d else None
        elbow_xyz_world = d['elbow_xyz_world'] if 'elbow_xyz_world' in d else None

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
        chunk_data = chunk_data.astype(np.float32)

        save_kwargs = dict(times=times, firing_rates=chunk_data,
                            joint_angles=joint_angles, direction=direction,
                            coef_seed=coef_seed)
        if wrist_xyz_world is not None:
            save_kwargs['wrist_xyz_world'] = wrist_xyz_world
        if elbow_xyz_world is not None:
            save_kwargs['elbow_xyz_world'] = elbow_xyz_world
        np.savez(out_npz, **save_kwargs)

    print(f"  Saved {len(npz_files)} spindle files for coef_seed={coef_seed}")

print("\nAll done. Original center_out_*_spindles.npz (coef_seed=0 default) left untouched.")
