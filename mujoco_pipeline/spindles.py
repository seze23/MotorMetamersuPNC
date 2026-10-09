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

# Parameters of the 25 muscles in MOBL_ARMS_41_seb_writing_pos.osim, in the
# canonical MUSCLE_NAMES order. The adapter assumes a rigid tendon and
# constant muscle thickness h = L0 sin(alpha0).
MOBL_OPTIMAL_FIBER_LENGTH_MM = np.array([
    93.2, 97.6, 107.8, 136.7, 75.5, 254.0, 232.4, 278.9, 144.2, 138.5,
    138.5, 87.3, 68.2, 162.4, 74.1, 27.0, 115.7, 132.1, 85.8, 172.6,
    81.0, 49.2, 113.8, 134.0, 113.8,
], dtype=np.float32)
MOBL_TENDON_SLACK_LENGTH_MM = np.array([
    97.0, 93.0, 109.5, 38.0, 30.8, 120.0, 176.5, 140.3, 2.8, 89.0,
    132.0, 33.0, 39.5, 20.0, 71.3, 18.0, 272.3, 192.3, 53.5, 133.0,
    244.0, 98.0, 98.0, 143.0, 90.8,
], dtype=np.float32)
MOBL_OPTIMAL_PENNATION_RAD = np.array([
    0.4712389, 0.38397244, 0.26179939, 0.31415927, 0.3316126,
    0.43633231, 0.33161256, 0.36651914, 0.29670597, 0.4537856,
    0.43633231, 0.34906585, 0.12217305, 0.27925268, 0.41887902,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17453293, 0.15707963,
    0.20943951, 0.15707963,
], dtype=np.float32)

# Equilibrated OpenSim fiber lengths at the pipeline's canonical braced rest
# pose (20, 40, 25, 85, -30, 0, 0 degrees). The rigid-tendon adapter preserves
# posture dependence but not OpenSim's compliant-tendon equilibrium offset.
MOBL_CANONICAL_REST_FIBER_LENGTH_MM = np.array([
    78.534477, 112.42647, 79.597488, 124.4935, 82.57122,
    172.89792, 140.50862, 205.20668, 129.49788, 109.01423,
    87.411636, 68.763367, 60.044487, 134.5542, 69.761574,
    19.941072, 103.96536, 129.80994, 69.148018, 131.80704,
    59.997055, 41.862862, 99.97332, 180.85269, 95.162003,
], dtype=np.float32)


def musculotendon_to_mobl_fiber_length(lengths_mm):
    """Map MuJoCo musculotendon length to MoBL fiber length in millimeters.

    This is the rigid-tendon, fixed-width pennation geometry used by the
    Millard equilibrium muscle model. It changes length semantics only; it
    does not fit or alter posture-dependent tendon routing.
    """
    lengths = np.asarray(lengths_mm, dtype=np.float32)
    if lengths.shape[-1] != len(MUSCLE_NAMES):
        raise ValueError(
            f"Expected {len(MUSCLE_NAMES)} muscles on the last axis; "
            f"received {lengths.shape}"
        )
    along_tendon = np.maximum(lengths - MOBL_TENDON_SLACK_LENGTH_MM, 0.0)
    thickness = MOBL_OPTIMAL_FIBER_LENGTH_MM * np.sin(MOBL_OPTIMAL_PENNATION_RAD)
    return np.sqrt(along_tendon**2 + thickness**2).astype(np.float32)


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
