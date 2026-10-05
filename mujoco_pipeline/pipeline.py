"""Run center-out reaches to spindle firing rates on both MuJoCo arms.

This does not call the OpenSim pipeline. Outputs go to ``outputs_mujoco/``.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from mujoco_pipeline.models import load_arm
from mujoco_pipeline.reaching import apply_rest_pose, center_out_paths, free_qpos, track_path
from mujoco_pipeline.spindles import (
    MOBL_OPTIMAL_FIBER_LENGTH_MM,
    SPINDLE_CONFIG,
    firing_rates,
    musculotendon_to_mobl_fiber_length,
    record_muscle_lengths,
    spindle_coefficients,
)
from utils.muscle_names import MUSCLE_NAMES

REPO_DIR = Path(__file__).resolve().parents[1]
BACKENDS = ("myosuite", "myosuite_corrected", "ms_human_700")


def _experiment_dir(config):
    directory = REPO_DIR / "outputs_mujoco" / config["experiment"]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save_path(directory, path, axes, shoulder):
    path_dir = directory / "paths"
    path_dir.mkdir(parents=True, exist_ok=True)
    output = path_dir / f"{path['id']}.npz"
    np.savez(
        output,
        xyz=path["xyz"],
        times=path["times"],
        center_xyz=path["center_xyz"],
        target_xyz=path["target_xyz"],
        sample_rate_hz=path["sample_rate_hz"],
        position_units="cm",
        coordinate_frame="shoulder_centered_horizontal",
        axes_mujoco=axes,
        shoulder_m=shoulder,
    )
    return output


def _plot_spindles(times, rates, direction, peak_velocity, rate_range, muscle_names, figure_path):
    ia_colors = plt.cm.Reds(np.linspace(0.4, 0.9, rates.shape[1] // 2))
    ii_colors = plt.cm.Blues(np.linspace(0.4, 0.9, rates.shape[1] // 2))
    n_ia = rates.shape[1] // 2
    fig, axes = plt.subplots(
        2, len(muscle_names), figsize=(5 * len(muscle_names), 7), sharex=True,
        layout="constrained", gridspec_kw={"hspace": 0.35, "wspace": 0.3},
    )
    for column, muscle_name in enumerate(muscle_names):
        muscle_index = MUSCLE_NAMES.index(muscle_name)
        for channel in range(n_ia):
            axes[0, column].plot(
                times, rates[0, channel, muscle_index, :],
                color=ia_colors[channel], linewidth=0.9,
            )
            axes[1, column].plot(
                times, rates[0, n_ia + channel, muscle_index, :],
                color=ii_colors[channel], linewidth=0.9,
            )
        axes[0, column].set_title(f"{muscle_name} Ia")
        axes[1, column].set_title(f"{muscle_name} II")
        axes[1, column].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Firing rate (Hz)")
    axes[1, 0].set_ylabel("Firing rate (Hz)")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{direction.replace('_', ' ')}    "
        f"peak |velocity| {peak_velocity:.0f} mm/s    "
        f"rates {rate_range[0]:.1f}–{rate_range[1]:.1f} Hz"
    )
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _run_backend(name, config, experiment_dir, spindle_bundle):
    print(f"\n=== {name} ===", flush=True)
    arm = load_arm(name, config)
    print(arm.description)
    for note in arm.notes:
        print(f"  - {note}")
    backend_dir = experiment_dir / name
    for subdirectory in ("paths", "ik", "muscles", "spindles", "figures"):
        (backend_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    paths, axes, shoulder, center = center_out_paths(arm, config)
    apply_rest_pose(arm, config)
    rest_q = free_qpos(arm)
    rest_lengths = record_muscle_lengths(arm, arm.data.qpos[None, :])[0]
    uses_fiber_adapter = name == "myosuite_corrected"
    length_reference = (
        MOBL_OPTIMAL_FIBER_LENGTH_MM.copy() if uses_fiber_adapter else rest_lengths
    )
    print(
        f"  Rest hand, shoulder-centered cm: "
        f"{np.array2string(center, precision=2)}"
    )
    print(
        "  Rest actuator length (mm): "
        f"{rest_lengths.min():.1f}–{rest_lengths.max():.1f}"
    )

    spindle_config, coefficients, sampled, muscles = spindle_bundle
    summaries = []
    opensim_opt = np.asarray(spindle_config["optimal_lengths"], dtype=np.float32)
    plot_muscles = config.get("plot_muscles", ["BIClong", "TRIlat", "DELT1"])

    for path in paths:
        _save_path(backend_dir, path, axes, shoulder)
        solved = track_path(arm, path, axes, shoulder, rest_q, config["ik"])
        musculotendon_lengths = record_muscle_lengths(arm, solved["joint_qpos"])
        lengths = (
            musculotendon_to_mobl_fiber_length(musculotendon_lengths)
            if uses_fiber_adapter else musculotendon_lengths
        )
        rates, velocity, acceleration = firing_rates(
            lengths, path["times"], length_reference,
            spindle_config, coefficients, sampled, muscles,
        )
        joint_degrees = np.degrees(solved["free_qpos"]).astype(np.float32)
        muscle_path = backend_dir / "muscles" / f"{path['id']}.npz"
        np.savez(
            muscle_path,
            times=path["times"],
            actuator_length_mm=musculotendon_lengths,
            spindle_input_length_mm=lengths,
            musculotendon_length_mm=musculotendon_lengths,
            fiber_length_mm=lengths if uses_fiber_adapter else np.array([], dtype=np.float32),
            length_reference_mm=length_reference,
            opensim_optimal_fiber_length_mm=opensim_opt,
            joint_angles_deg=joint_degrees,
            joint_names=np.array(arm.free_joint_names),
            achieved_xyz_cm=solved["achieved_xyz"],
            ik_error_cm=solved["ik_error_cm"],
            muscle_names=np.array(MUSCLE_NAMES),
            actuator_names=np.array(arm.actuator_names),
            muscle_match=np.array(arm.muscle_match),
            length_kind=(
                "mobl_adapter_fiber_length_mm" if uses_fiber_adapter
                else "mujoco_musculotendon_actuator_mm"
            ),
        )
        spindle_path = backend_dir / "spindles" / f"{path['id']}.npz"
        np.savez(
            spindle_path,
            times=path["times"],
            firing_rates=rates,
            velocity_mm_s=velocity,
            acceleration_mm_s2=acceleration,
            joint_angles_deg=joint_degrees,
            achieved_xyz_cm=solved["achieved_xyz"],
            muscle_names=np.array(MUSCLE_NAMES),
        )
        figure_path = backend_dir / "figures" / f"spindles_{path['id']}.png"
        _plot_spindles(
            path["times"], rates, path["id"],
            float(np.abs(velocity).max()),
            (float(rates.min()), float(rates.max())),
            plot_muscles, figure_path,
        )
        summary = {
            "id": path["id"],
            "mean_ik_error_cm": float(solved["ik_error_cm"].mean()),
            "max_ik_error_cm": float(solved["ik_error_cm"].max()),
            "unique_targets": int(solved["unique_targets"]),
            "length_mm": [float(lengths.min()), float(lengths.max())],
            "firing_rate_hz": [float(rates.min()), float(rates.max())],
            "peak_velocity_mm_s": float(np.abs(velocity).max()),
        }
        summaries.append(summary)
        print(
            f"  {path['id']:<16} IK mean {summary['mean_ik_error_cm']:.3f} cm, "
            f"max {summary['max_ik_error_cm']:.3f} cm, "
            f"rates {summary['firing_rate_hz'][0]:.1f}–"
            f"{summary['firing_rate_hz'][1]:.1f} Hz",
            flush=True,
        )

    manifest = {
        "backend": name,
        "description": arm.description,
        "source": arm.source,
        "notes": arm.notes,
        "muscle_names": list(MUSCLE_NAMES),
        "actuator_names": arm.actuator_names,
        "muscle_match": arm.muscle_match,
        "length_reference_mm": length_reference.tolist(),
        "length_kind": (
            "mobl_adapter_fiber_length_mm" if uses_fiber_adapter
            else "mujoco_musculotendon_actuator_mm"
        ),
        "fiber_adapter": "millard_rigid_tendon_fixed_width" if uses_fiber_adapter else None,
        "trajectories": summaries,
    }
    with (backend_dir / "manifest.yaml").open("w") as manifest_file:
        yaml.safe_dump(manifest, manifest_file, sort_keys=False)
    _export_example_hdf5(backend_dir, length_reference, arm.free_joint_names)
    return manifest


def _export_example_hdf5(backend_dir, length_reference_mm, joint_names):
    """Write the eight reaches in the training-file layout.

    Eight trials are a format example. They are not a training set.
    """
    import h5py

    muscle_files = sorted((backend_dir / "muscles").glob("*.npz"))
    lengths, velocities, accelerations, coords, joints = [], [], [], [], []
    for muscle_file in muscle_files:
        data = np.load(muscle_file)
        key = "spindle_input_length_mm" if "spindle_input_length_mm" in data else "actuator_length_mm"
        length = data[key].T[None, ...].astype(np.float32)
        times = data["times"]
        dt = float(np.median(np.diff(times)))
        velocity = np.gradient(length, dt, axis=2).astype(np.float32)
        acceleration = np.gradient(velocity, dt, axis=2).astype(np.float32)
        lengths.append(length[0])
        velocities.append(velocity[0])
        accelerations.append(acceleration[0])
        coords.append(data["achieved_xyz_cm"].T.astype(np.float32))
        joints.append(data["joint_angles_deg"].T.astype(np.float32))
    output = backend_dir / "example_reaches_not_for_training.hdf5"
    with h5py.File(output, "w") as handle:
        handle.create_dataset("muscle_lengths", data=np.stack(lengths))
        handle.create_dataset("muscle_velocities", data=np.stack(velocities))
        handle.create_dataset("muscle_accelerations", data=np.stack(accelerations))
        handle.create_dataset("endeffector_coords", data=np.stack(coords))
        handle.create_dataset("joint_coords", data=np.stack(joints))
        handle.attrs["training_set"] = False
        handle.attrs["n_trials"] = len(muscle_files)
        handle.attrs["length_unit"] = "mm musculotendon actuator length"
        handle.attrs["endeffector_unit"] = "cm, shoulder-centered, right/forward/up"
        handle.attrs["joint_unit"] = "degrees"
        handle.attrs["joint_names"] = np.array(joint_names, dtype="S")
        handle.attrs["length_reference_mm"] = length_reference_mm
        handle.attrs["warning"] = (
            "Eight center-out reaches are not enough to train the decoder. "
            "Replace optimal_lengths in the spindle YAML with length_reference_mm "
            "before converting this file, and generate thousands of trials first."
        )
    print(f"  Wrote format example {output.name} ({len(muscle_files)} trials)")


def _comparison_figure(experiment_dir, config):
    plot_muscles = config.get("plot_muscles", ["BIClong", "TRIlat", "DELT1"])
    direction = "90_forward"
    present = []
    for name in BACKENDS:
        spindle_file = experiment_dir / name / "spindles" / f"{direction}.npz"
        muscle_file = experiment_dir / name / "muscles" / f"{direction}.npz"
        if spindle_file.is_file() and muscle_file.is_file():
            present.append((name, np.load(spindle_file), np.load(muscle_file)))
    if len(present) < 2:
        return None
    colors = {
        "myosuite": "#b45309",
        "myosuite_corrected": "#15803d",
        "ms_human_700": "#1d4ed8",
    }
    fig, axes = plt.subplots(
        2, len(plot_muscles), figsize=(12, 6), sharex=True, layout="constrained",
    )
    for column, muscle_name in enumerate(plot_muscles):
        muscle_index = MUSCLE_NAMES.index(muscle_name)
        for name, spindles, muscles in present:
            times = spindles["times"]
            ia = spindles["firing_rates"][0, :5, muscle_index, :].mean(axis=0)
            key = "spindle_input_length_mm" if "spindle_input_length_mm" in muscles else "actuator_length_mm"
            length = muscles[key][:, muscle_index]
            reference = muscles["length_reference_mm"][muscle_index]
            axes[0, column].plot(times, ia, color=colors[name], label=name, linewidth=1.4)
            axes[1, column].plot(
                times, (length / reference - 1.0) * 100.0,
                color=colors[name], label=name, linewidth=1.4,
            )
        axes[0, column].set_title(f"{muscle_name} mean Ia")
        axes[1, column].set_xlabel("Time (s)")
        axes[0, column].spines[["top", "right"]].set_visible(False)
        axes[1, column].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Firing rate (Hz)")
    axes[1, 0].set_ylabel("Length change from rest (%)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Forward reach: same spindle equation, each model's own actuator lengths")
    figure_path = experiment_dir / "comparison_forward_reach.png"
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def run(config_path=None, backends=None):
    config_path = Path(config_path or REPO_DIR / "mujoco_pipeline" / "configs" / "center_out.yaml")
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)
    experiment_dir = _experiment_dir(config)
    proprioception = config.get("proprioception", {})
    spindle_bundle = spindle_coefficients(
        proprioception.get("coefficient_seed", 0),
        proprioception.get("num_afferents", 5),
    )
    selected = backends or BACKENDS
    manifests = {}
    for name in selected:
        manifests[name] = _run_backend(name, config, experiment_dir, spindle_bundle)
    figure = _comparison_figure(experiment_dir, config)
    summary = {
        "experiment": config["experiment"],
        "output_dir": str(experiment_dir),
        "comparison_figure": str(figure) if figure else None,
        "spindle_config": str(SPINDLE_CONFIG),
        "backends": {
            name: {
                "source": item["source"],
                "notes": item["notes"],
                "trajectories": item["trajectories"],
            }
            for name, item in manifests.items()
        },
    }
    with (experiment_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"\nOutputs: {experiment_dir}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_DIR / "mujoco_pipeline" / "configs" / "center_out.yaml"),
    )
    parser.add_argument("--backend", choices=BACKENDS, action="append")
    args = parser.parse_args(argv)
    run(args.config, args.backend)


if __name__ == "__main__":
    main()
