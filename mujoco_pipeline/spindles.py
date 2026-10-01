"""Convert MuJoCo actuator lengths into the OpenSim spindle firing rates."""

import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.signal import savgol_filter

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# The shared spindle module imports torch at import time. The transfer
# function itself is NumPy, so a temporary placeholder is enough. It is
# removed immediately: leaving a fake torch module installed makes SciPy
# treat every array as a torch tensor.
_created_torch_stub = False
try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")
    _created_torch_stub = True

from extract_data.generate_train_test_data import process_chunk
from utils.muscle_names import MUSCLE_NAMES
from utils.spindle_FR_helper import (
    get_sampled_coefficients,
    load_coefficients,
    normalize,
)

if _created_torch_stub:
    sys.modules.pop("torch", None)

SPINDLE_CONFIG = REPO_DIR / "extract_data" / "configs" / "train_test_data_spindles_extended.yaml"


def spindle_coefficients(seed, n_afferents):
    with SPINDLE_CONFIG.open() as config_file:
        config = yaml.safe_load(config_file)
    config["seed"] = int(seed)
    config["num_i_a"] = int(n_afferents)
    config["num_ii"] = int(n_afferents)
    for kind in ("i_a", "ii"):
        config[f"{kind}_coeff_path"] = str(REPO_DIR / config[f"{kind}_coeff_path"])
        sampled = (
            REPO_DIR / "data" / "extended_spindle_coefficients" / kind / "linear"
            / f"sampled_coefficients_{kind}_{n_afferents}_{seed}.csv"
        )
        if not sampled.is_file():
            raise FileNotFoundError(
                f"Missing spindle sample {sampled.name}. The OpenSim pipeline "
                "ships samples for 5 afferents and seeds 0-4."
            )
        config[f"{kind}_sampled_coeff_path"] = str(sampled)
    muscles = list(range(len(MUSCLE_NAMES)))
    coefficients = {
        kind: load_coefficients(config[f"{kind}_coeff_path"])
        for kind in ("i_a", "ii")
    }
    sampled = get_sampled_coefficients(
        config, [n_afferents, n_afferents], muscles, coefficients
    )
    return config, coefficients, sampled, muscles


def firing_rates(lengths_mm, times, length_reference_mm, spindle_config, coefficients, sampled, muscles):
    """Same Ia/II calculation as process/computefrcenterout.py.

    ``lengths_mm`` is (time, muscles). ``length_reference_mm`` plays the role
    of optimal length in the shared ``normalize`` function.
    """
    dt = float(np.median(np.diff(times)))
    series = lengths_mm.T[np.newaxis, :, :].astype(np.float32)
    velocity = np.gradient(series, dt, axis=2)
    window = 31 if series.shape[2] >= 31 else series.shape[2] // 2 * 2 + 1
    if window >= 5:
        for muscle_index in range(series.shape[1]):
            velocity[0, muscle_index, :] = savgol_filter(
                velocity[0, muscle_index, :], window_length=window, polyorder=1
            )
    acceleration = np.gradient(velocity, dt, axis=2).astype(np.float32)
    normalized = normalize(
        series.astype(np.float32),
        velocity.astype(np.float32),
        acceleration,
        np.asarray(length_reference_mm, dtype=np.float32),
    )
    rates = process_chunk(
        normalized, coefficients, [spindle_config["num_i_a"], spindle_config["num_ii"]],
        muscles, chunk_size=1, sampled_coefficients=sampled,
    ).astype(np.float32)
    return rates, velocity.astype(np.float32), acceleration


def record_muscle_lengths(arm, joint_qpos):
    """Actuator length in millimeters at each solved posture."""
    lengths = np.empty((len(joint_qpos), len(arm.actuator_ids)), dtype=np.float32)
    for frame_index, qpos in enumerate(joint_qpos):
        arm.data.qpos[:] = qpos
        arm.data.qvel[:] = 0.0
        mujoco.mj_forward(arm.model, arm.data)
        lengths[frame_index] = arm.data.actuator_length[arm.actuator_ids] * 1000.0
    return lengths
