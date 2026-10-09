"""Compute Mathis spindle rates from corrected-MyoArm fiber artifacts."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import load_manifest, resolve_artifact, set_artifact
from paths import EXPERIMENT_CONFIG, FIGURES_DIR, SPINDLES_DIR
from mujoco_pipeline.spindles import (
    MOBL_OPTIMAL_FIBER_LENGTH_MM,
    firing_rates,
    spindle_coefficients,
)
from utils.muscle_names import MUSCLE_NAMES

PLOT_MUSCLES = ("BIClong", "TRIlat", "DELT1")


def _plot(times, rates, output):
    fig, axes = plt.subplots(2, len(PLOT_MUSCLES), figsize=(15, 7), sharex=True)
    for column, name in enumerate(PLOT_MUSCLES):
        muscle = MUSCLE_NAMES.index(name)
        for channel in range(5):
            axes[0, column].plot(times, rates[0, channel, muscle], linewidth=0.9)
            axes[1, column].plot(times, rates[0, channel + 5, muscle], linewidth=0.9)
        axes[0, column].set_title(f"{name} (Ia)")
        axes[1, column].set_title(f"{name} (II)")
        axes[1, column].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Firing rate (Hz)")
    axes[1, 0].set_ylabel("Firing rate (Hz)")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    options = EXPERIMENT_CONFIG.get("proprioception", {})
    family = str(options.get("model_family", "extended")).lower()
    if family != "extended":
        raise ValueError(
            "myosuite_corrected currently uses the extended Mathis spindle "
            "configuration; set proprioception.model_family: extended"
        )
    seed = int(options.get("coefficient_seed", 0))
    spindle_config, coefficients, sampled, muscles = spindle_coefficients(seed, 5)
    trajectories = load_manifest()["trajectories"]
    for trajectory in trajectories:
        trajectory_id = trajectory["id"]
        data = np.load(resolve_artifact(trajectory["muscle_data"]), allow_pickle=True)
        rates, velocity, acceleration = firing_rates(
            data["fiber_lengths"], data["times"],
            MOBL_OPTIMAL_FIBER_LENGTH_MM,
            spindle_config, coefficients, sampled, muscles,
        )
        output = Path(SPINDLES_DIR) / f"{trajectory_id}.npz"
        figure = Path(FIGURES_DIR) / f"spindles_{trajectory_id}.png"
        save = {
            "times": data["times"],
            "firing_rates": rates,
            "joint_angles": data["joint_angles"],
            "fiber_velocity": velocity,
            "fiber_acceleration": acceleration,
            "direction": trajectory_id,
        }
        for key in ("wrist_xyz_world", "elbow_xyz_world"):
            if key in data:
                save[key] = data[key]
        np.savez(output, **save)
        _plot(data["times"], rates, figure)
        set_artifact(trajectory_id, "spindle_data", output)
        set_artifact(trajectory_id, "spindle_figure", figure)
        print(
            f"  {trajectory_id:<20} firing rates "
            f"{rates.min():.1f}-{rates.max():.1f} Hz -> {output}"
        )
    print(f"Computed spindle signals for {len(trajectories)} path(s).")


if __name__ == "__main__":
    main()
