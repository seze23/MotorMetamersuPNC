import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

from directory_paths import SAVE_DIR
from utils.spindle_FR_helper import (
    normalize,
    load_coefficients,
    get_sampled_coefficients,
    spindle_transfer_function_coeffs,
)
from data_generation.extract_flag3d_data_utils import Arm

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 sweep constants and design choices
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 240            # Hz
N_FRAMES = 1152              # 4.8 sec at 240 Hz
DURATION = N_FRAMES / SAMPLE_RATE
N_MUSCLES = 25
N_IA = 5                     # Ia afferents per muscle
N_II = 5                     # II afferents per muscle
N_AFFERENTS = N_IA + N_II    # 10 total afferent channels

# Canonical posture (#16): default resting posture in ArmModel
Q0_DEG = np.array([50.0, 40.0, 20.0, 75.0], dtype=np.float32)

# Optimal fiber lengths from the repo's spindle data config
OPTIMAL_LENGTHS = np.array(
    [
        93.2, 97.6, 107.8, 136.7, 75.5,
        254.0, 232.4, 278.9, 144.2, 138.5,
        138.5, 87.3, 68.2, 162.4, 74.1,
        27.0, 115.7, 132.1, 85.8, 172.6,
        81.0, 49.2, 113.8, 134.0, 113.8,
    ],
    dtype=np.float32,
)

# Paths for spindle coefficient files
COEFF_PATHS = {
    "i_a": os.path.join(SAVE_DIR, "data/spindle_coefficients/i_a/linear/coefficients.csv"),
    "ii": os.path.join(SAVE_DIR, "data/spindle_coefficients/ii/linear/coefficients.csv"),
}
EXTENDED_COEFF_PATHS_TEMPLATE = {
    "i_a": os.path.join(SAVE_DIR, "data/extended_spindle_coefficients/i_a/linear/coefficients_i_a_5_{seed}.csv"),
    "ii": os.path.join(SAVE_DIR, "data/extended_spindle_coefficients/ii/linear/coefficients_ii_5_{seed}.csv"),
}

MUSCLE_SCALE = 1000.0  # convert lengths to mm to match normalization assumptions


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def minimum_jerk_profile(t, T):
    """Minimum-jerk scalar profile from Flash & Hogan (1985)."""
    tau = t / T
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def make_wrist_trajectory(start_xyz, direction_xy, amplitude, n_frames, sample_rate):
    """Generate a wrist trajectory in the shoulder-horizontal sweep plane.

    The plane is defined so that the vertical component remains fixed, matching
    the horizontal-plane sweep design (#17). vx displaces the wrist along the
    arm's medial-lateral axis and vy along the anterior-posterior axis.
    """
    T = n_frames / sample_rate
    t = np.linspace(0.0, T, n_frames, dtype=np.float32)
    profile = minimum_jerk_profile(t, T)

    delta = np.array([direction_xy[0], 0.0, direction_xy[1]], dtype=np.float32)
    traj = start_xyz[None, :] + profile[:, None] * (delta[None, :] * amplitude)
    return traj


def build_spindle_config(coeff_seed, extended=False):
    """Create a config dict for loading spindle coefficients and sampling."""
    if extended:
        ia_path = EXTENDED_COEFF_PATHS_TEMPLATE["i_a"].format(seed=coeff_seed)
        ii_path = EXTENDED_COEFF_PATHS_TEMPLATE["ii"].format(seed=coeff_seed)
    else:
        ia_path = COEFF_PATHS["i_a"]
        ii_path = COEFF_PATHS["ii"]

    return {
        "seed": coeff_seed,
        "i_a_coeff_path": ia_path,
        "ii_coeff_path": ii_path,
        "i_a_sampled_coeff_path": None,
        "ii_sampled_coeff_path": None,
    }


def muscle_kinematics_to_spindle_rates(lengths, velocities, accelerations, coeff_seed=0, extended=False):
    """Convert normalized muscle kinematics into spindle firing rates.

    This reuses the repository's spindle coefficient sampling pipeline, so the
    generated sweep matches the published spindle input format.
    """
    muscles = list(range(N_MUSCLES))
    num_coefficients = [N_IA, N_II]
    config = build_spindle_config(coeff_seed, extended=extended)

    coefficients_raw = {
        "i_a": load_coefficients(config["i_a_coeff_path"]),
        "ii": load_coefficients(config["ii_coeff_path"]),
    }

    sampled = get_sampled_coefficients(config, num_coefficients, muscles, coefficients_raw)

    T = lengths.shape[2]
    spindle_input = np.zeros((N_AFFERENTS, N_MUSCLES, T), dtype=np.float32)

    for muscle_idx in muscles:
        for i, coeff_type in enumerate(["i_a", "ii"]):
            for j in range(num_coefficients[i]):
                channel_idx = sum(num_coefficients[:i]) + j
                coeff_idx = sampled[coeff_type][muscle_idx][j]
                raw_coeffs = coefficients_raw[coeff_type][muscle_idx]
                coeffs = {
                    "k_l": raw_coeffs["k_l"][coeff_idx],
                    "k_v": raw_coeffs["k_v"][coeff_idx],
                    "e_v": raw_coeffs["e_v"][coeff_idx],
                    "k_a": raw_coeffs["k_a"][coeff_idx],
                    "k_c": raw_coeffs["k_c"][coeff_idx],
                    "max_rate": raw_coeffs["max_rate"][coeff_idx],
                }
                rates = spindle_transfer_function_coeffs(
                    lengths[0, muscle_idx, :],
                    velocities[0, muscle_idx, :],
                    accelerations[0, muscle_idx, :],
                    coeffs,
                )
                spindle_input[channel_idx, muscle_idx, :] = rates.astype(np.float32)

    return spindle_input


def make_grid(grid_n=21):
    """Create a 2D sweep grid over [-1,1]^2 excluding the origin."""
    vals = np.linspace(-1.0, 1.0, grid_n, dtype=np.float32)
    vx, vy = np.meshgrid(vals, vals)
    grid = np.stack([vx.ravel(), vy.ravel()], axis=1)
    mask = ~((grid[:, 0] == 0.0) & (grid[:, 1] == 0.0))
    return grid[mask]


def trajectory_to_joint_angles(arm, wrist_traj):
    """Solve IK per frame to recover the joint-angle trajectory from wrist xyz."""
    T = wrist_traj.shape[0]
    q_traj = np.zeros((T, 4), dtype=np.float32)
    q_prev = arm.q0.copy()

    for t in range(T):
        hand_xyz = wrist_traj[t]
        _, link_pos = arm.get_xyz(q=q_prev)
        elbow_xyz = link_pos[:, 1]
        arm.q = q_prev
        q_sol = arm.inv_kin(elbow_xyz, hand_xyz)
        q_traj[t] = q_sol
        q_prev = q_sol

    return q_traj


def joint_angles_to_muscle_kinematics(arm, q_traj, dt):
    """Convert joint-angle trajectory into muscle length/velocity/acceleration."""
    T = q_traj.shape[0]
    hand_xyz = np.zeros((T, 3), dtype=np.float32)
    elbow_xyz = np.zeros((T, 3), dtype=np.float32)

    for t in range(T):
        h, lp = arm.get_xyz(q=q_traj[t])
        hand_xyz[t] = h
        elbow_xyz[t] = lp[:, 1]

    rng = np.random.default_rng(seed=42)
    moment_arms = rng.uniform(-0.3, 0.3, size=(N_MUSCLES, 4)).astype(np.float32)
    q_rad = np.deg2rad(q_traj).T

    raw_lengths = OPTIMAL_LENGTHS[:, None] * (1.0 + moment_arms @ q_rad)
    raw_lengths = raw_lengths * MUSCLE_SCALE
    raw_velocities = np.gradient(raw_lengths, dt, axis=1)
    raw_accelerations = np.gradient(raw_velocities, dt, axis=1)

    lengths = raw_lengths[None, :, :]
    velocities = raw_velocities[None, :, :]
    accelerations = raw_accelerations[None, :, :]

    return lengths, velocities, accelerations


def run_sweep(out_path, amplitude=5.0, grid_n=21, coeff_seed=0, extended=False):
    """Generate the full directional sweep dataset and save it to HDF5."""
    arm = Arm(q0=Q0_DEG.copy())
    dt = 1.0 / SAMPLE_RATE
    start_xyz, link_pos = arm.get_xyz(q=Q0_DEG)

    grid = make_grid(grid_n)
    N_dir = len(grid)
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None

    with h5py.File(out_path, "w") as f:
        f.create_dataset("spindle_inputs", shape=(N_dir, N_AFFERENTS, N_MUSCLES, N_FRAMES), dtype=np.float32)
        f.create_dataset("joint_angles", shape=(N_dir, N_FRAMES, 4), dtype=np.float32)
        f.create_dataset("labels", shape=(N_dir, N_FRAMES, 7), dtype=np.float32)
        f.create_dataset("directions", data=grid.astype(np.float32))
        f.create_dataset("start_xyz", data=start_xyz.astype(np.float32))
        f.create_dataset("q0", data=Q0_DEG.astype(np.float32))
        f.attrs["amplitude"] = amplitude
        f.attrs["grid_n"] = grid_n
        f.attrs["coeff_seed"] = coeff_seed
        f.attrs["extended"] = int(extended)
        f.attrs["sample_rate"] = SAMPLE_RATE
        f.attrs["n_frames"] = N_FRAMES

        for i, (vx, vy) in enumerate(tqdm(grid, desc="Sweep directions")):
            wrist_traj = make_wrist_trajectory(start_xyz, np.array([vx, vy], dtype=np.float32), amplitude, N_FRAMES, SAMPLE_RATE)
            arm_i = Arm(q0=Q0_DEG.copy())
            q_traj = trajectory_to_joint_angles(arm_i, wrist_traj)
            lengths, velocities, accelerations = joint_angles_to_muscle_kinematics(arm_i, q_traj, dt)
            norm_data = normalize(lengths, velocities, accelerations, OPTIMAL_LENGTHS)
            spindle = muscle_kinematics_to_spindle_rates(norm_data["lengths"], norm_data["velocities"], norm_data["accelerations"], coeff_seed=coeff_seed, extended=extended)

            labels = np.zeros((N_FRAMES, 7), dtype=np.float32)
            labels[:, 0:3] = wrist_traj
            labels[:, 3:6] = q_traj[:, 0:3]
            labels[:, 6] = q_traj[:, 3]

            f["spindle_inputs"][i] = spindle
            f["joint_angles"][i] = q_traj.astype(np.float32)
            f["labels"][i] = labels

    return out_path


def verify_output(out_path):
    """Quickly verify the generated sweep HDF5 file."""
    with h5py.File(out_path, "r") as f:
        print("\n[verify] HDF5 keys:", list(f.keys()))
        print("[verify] spindle_inputs shape:", f["spindle_inputs"].shape)
        print("[verify] joint_angles shape:", f["joint_angles"].shape)
        print("[verify] labels shape:", f["labels"].shape)
        print("[verify] directions shape:", f["directions"].shape)
        print("[verify] start_xyz:", f["start_xyz"][:])
        print("[verify] q0:", f["q0"][:])

        sample = f["spindle_inputs"][0]
        print("[verify] sample min/max:", sample.min(), sample.max())
        dirs = f["directions"][:]
        print("[verify] (0,0) excluded:", not np.any((dirs[:, 0] == 0.0) & (dirs[:, 1] == 0.0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Phase 3 directional sweep dataset.")
    parser.add_argument("--out_path", type=str, default=os.path.join(SAVE_DIR, "data/sweep/sweep_seed0.hdf5"), help="Output HDF5 path")
    parser.add_argument("--amplitude", type=float, default=5.0, help="Sweep amplitude in cm")
    parser.add_argument("--grid_n", type=int, default=21, help="Grid resolution")
    parser.add_argument("--coeff_seed", type=int, default=0, help="Spindle coefficient seed")
    parser.add_argument("--extended", action="store_true", help="Use extended spindle coefficient sets")
    parser.add_argument("--verify_only", type=str, default=None, help="Verify an existing output HDF5 without regenerating")
    args = parser.parse_args()

    if args.verify_only:
        verify_output(args.verify_only)
    else:
        out_path = run_sweep(args.out_path, amplitude=args.amplitude, grid_n=args.grid_n, coeff_seed=args.coeff_seed, extended=args.extended)
        verify_output(out_path)
