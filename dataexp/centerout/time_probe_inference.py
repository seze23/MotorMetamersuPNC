"""
Timing probe for centeroutinference.py -- measures real per-unit costs before
committing to the 20-checkpoint x N-spatial-point grid job, instead of
guessing. Run once on the cluster:

  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout
  conda activate proprioception
  python3 time_probe_inference.py

Reports:
  - whether CUDA/the RTX PRO 6000 is actually usable from this env
  - config + model load time (paid once per checkpoint)
  - per-direction inference time (paid once per checkpoint x direction)
  - extrapolated estimate for the full grid, clearly labeled as an
    extrapolation from n=1, not a measured total
"""

import os
import sys
import time
import glob
import yaml
import h5py
import numpy as np
import torch

from paths import CENTEROUT_DIR
REPO_DIR = os.environ.get(
    "PROPRIO_REPO_DIR", "/home/sydneyez/sydneyez/ProprioceptiveIllusions")
sys.path.insert(0, REPO_DIR)

from utils.visualize_sample import get_shoulder_elbow_wrist_loc
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset

COEF_SEED, TRAIN_SEED = 0, 9  # the already-validated checkpoint
MODEL_PATH = os.path.join(
    REPO_DIR,
    "trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints",
    f"spatiotemporal_4_8-8-32-64_7171_{COEF_SEED}_{TRAIN_SEED}",
)
TIME_STEPS = 1152

print("=" * 60)
print("DEVICE CHECK")
print("=" * 60)
print(f"torch.__version__      = {torch.__version__}")
print(f"torch.cuda.is_available = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device name           = {torch.cuda.get_device_name(0)}")
    print(f"  capability (sm_XX)    = {torch.cuda.get_device_capability(0)}")
    try:
        # sanity: does a trivial matmul actually run on this GPU without
        # kernel-image errors (would surface unsupported-architecture issues)
        _ = (torch.randn(4, 4, device="cuda") @ torch.randn(4, 4, device="cuda"))
        torch.cuda.synchronize()
        print("  trivial CUDA matmul   = OK")
    except Exception as e:
        print(f"  trivial CUDA matmul   = FAILED: {e}")
else:
    print("  No CUDA device visible to this process/environment.")
print()

print("=" * 60)
print("TIMING")
print("=" * 60)

t0 = time.time()
with open(os.path.join(MODEL_PATH, "config.yaml"), "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
model_config = {k: parse_config_value(v) for k, v in model_config.items()}
t_config = time.time() - t0
print(f"Config load:              {t_config:.2f} s")

spindle_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "center_out_*_spindles.npz")))
if not spindle_files:
    raise FileNotFoundError(f"No *_spindles.npz files in {CENTEROUT_DIR}")
sp_path = spindle_files[0]
direction = (os.path.basename(sp_path)
             .replace("center_out_", "").replace("_spindles.npz", ""))
print(f"Probing with direction: {direction}\n")

sp_data = np.load(sp_path, allow_pickle=True)
chunk_data = sp_data['firing_rates'].astype(np.float32)
joint_angles = sp_data['joint_angles']

labels_for_fk = np.zeros((TIME_STEPS, 7), dtype=np.float32)
labels_for_fk[:, 3] = joint_angles[:, 0]
labels_for_fk[:, 4] = joint_angles[:, 1]
labels_for_fk[:, 5] = joint_angles[:, 2]
labels_for_fk[:, 6] = joint_angles[:, 3]
_, _, wrist_loc = get_shoulder_elbow_wrist_loc(labels_for_fk)

labels = np.zeros((1, TIME_STEPS, 7), dtype=np.float32)
labels[0, :, 0:3] = wrist_loc
labels[0, :, 3] = joint_angles[:, 0]
labels[0, :, 4] = joint_angles[:, 1]
labels[0, :, 5] = joint_angles[:, 2]
labels[0, :, 6] = joint_angles[:, 3]

tmp_hdf5 = os.path.join(CENTEROUT_DIR, "_tmp_timeprobe.hdf5")
with h5py.File(tmp_hdf5, "w") as f:
    f.create_dataset("data", data=chunk_data)
    f.create_dataset("labels", data=labels)

t0 = time.time()
test_data = SpindleDataset(
    tmp_hdf5, dataset_type="test", key="spindle_info",
    task="letter_reconstruction_joints", aclass=None, need_muscles=False,
    new_size=model_config["input_shape"][-1],
)
t_dataset = time.time() - t0
print(f"SpindleDataset build:     {t_dataset:.2f} s")

candidate_devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
for device_name in candidate_devices:
    try:
        device = torch.device(device_name)
        t0 = time.time()
        tester = load_model(
            model_config, MODEL_PATH, "letter_reconstruction_joints",
            device, test_data, causal=True, save_dir=REPO_DIR,
        )
        t_load = time.time() - t0
        print(f"[{device_name}] Model load (checkpoint):  {t_load:.2f} s")

        t0 = time.time()
        predictions, _ = tester.get_predictions()
        _ = predictions[0].cpu().detach().numpy()
        t_infer = time.time() - t0
        print(f"[{device_name}] Inference (1 direction):  {t_infer:.2f} s")
        print(f"[{device_name}] Per-checkpoint fixed cost + 1 direction: {t_load + t_infer:.2f} s\n")
    except RuntimeError as e:
        print(f"[{device_name}] FAILED ({e.__class__.__name__}): {e}")
        print(f"[{device_name}] Skipping -- not usable in this environment.\n")

os.remove(tmp_hdf5)

print("=" * 60)
print("EXTRAPOLATION (from n=1 -- treat as order-of-magnitude, not exact)")
print("=" * 60)
print("Formula: total = N_checkpoints * (model_load + N_spatial_points * per_direction_inference)")
print("Fill in N_checkpoints (up to 20) and N_spatial_points (directions x radii) using the")
print("per-device numbers above. This does NOT include stages 1-5 (IK/extraction/spindle),")
print("which must also run once per NEW spatial point (but only once total, not per checkpoint,")
print("since spindle firing rates don't depend on which CNN checkpoint will read them).")
