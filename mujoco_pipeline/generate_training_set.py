"""Generate decoder training reaches from a MuJoCo arm.

Each trial is one planar minimum-jerk reach from the same rest pose used by
the center-out experiment. Direction and distance are sampled independently.
The file is what ``train_decoder.py`` reads: spindle ``data`` and 7-value
``labels``. You do not need an existing 30,000-trial archive.

Run from the repository root:

    python -m mujoco_pipeline.generate_training_set --num-trials 64

Thirty thousand trials is the original study size. It is about 35 GB and
several hours on one CPU. A few hundred trials is enough to test the cluster
job. Eight reaches are not a training set.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from mujoco_pipeline.models import end_effector_position, load_arm
from mujoco_pipeline.reaching import (
    apply_rest_pose,
    free_qpos,
    make_planar_reach,
    shoulder_axes,
    time_base,
    to_shoulder_cm,
    track_path,
)
from mujoco_pipeline.spindles import firing_rates, record_muscle_lengths, spindle_coefficients
from utils.muscle_names import MUSCLE_NAMES

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_DIR / "mujoco_pipeline" / "configs" / "center_out.yaml"
N_AFFERENTS = 10
N_MUSCLES = 25
N_LABELS = 7


def _load_config(path):
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def _pose_arm(backend, config):
    arm = load_arm(backend, config)
    apply_rest_pose(arm, config)
    axes, shoulder = shoulder_axes(arm)
    center = to_shoulder_cm(end_effector_position(arm), shoulder, axes)
    rest_q = free_qpos(arm)
    rest_lengths = record_muscle_lengths(arm, arm.data.qpos[None, :])[0]
    sample_rate, counts, times = time_base(config)
    if len(times) != 1152:
        raise RuntimeError(
            f"This rest timing produces {len(times)} frames. The decoder "
            "architecture used by train_decoder.py expects 1152."
        )
    return arm, axes, shoulder, center, rest_q, rest_lengths, sample_rate, counts, times


def _sample_offset(rng, config):
    min_cm = float(config.get("train_min_reach_cm", 3.0))
    max_cm = float(config.get("train_max_reach_cm", 12.0))
    limit = float(config.get("max_displacement_cm", 15.0))
    if max_cm > limit:
        raise ValueError("train_max_reach_cm cannot exceed max_displacement_cm")
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    distance = float(rng.uniform(min_cm, max_cm))
    offset = distance * np.array([np.cos(angle), np.sin(angle), 0.0])
    return angle, offset


def _one_trial(arm, rng, config, axes, shoulder, center, rest_q, rest_lengths,
               sample_rate, counts, times, spindle_bundle, trial_index, max_error_cm):
    spindle_config, coefficients, sampled, muscles = spindle_bundle
    attempts = 8
    for _ in range(attempts):
        angle, offset = _sample_offset(rng, config)
        path = make_planar_reach(
            center, offset, counts, times, sample_rate,
            f"trial_{trial_index}", np.degrees(angle),
        )
        solved = track_path(arm, path, axes, shoulder, rest_q, config["ik"])
        if float(solved["ik_error_cm"].max()) <= max_error_cm:
            lengths = record_muscle_lengths(arm, solved["joint_qpos"])
            rates, _, _ = firing_rates(
                lengths, times, rest_lengths,
                spindle_config, coefficients, sampled, muscles,
            )
            labels = np.concatenate(
                [solved["achieved_xyz"], np.degrees(solved["free_qpos"])],
                axis=1,
            ).astype(np.float32)
            return rates[0], labels, float(solved["ik_error_cm"].max())
    raise RuntimeError(
        f"Trial {trial_index} stayed above {max_error_cm} cm of hand error "
        f"after {attempts} targets."
    )


def _create_file(path, count, time_steps, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    handle.create_dataset(
        "data", shape=(count, N_AFFERENTS, N_MUSCLES, time_steps),
        dtype="float32", chunks=(1, N_AFFERENTS, N_MUSCLES, time_steps),
    )
    handle.create_dataset(
        "labels", shape=(count, time_steps, N_LABELS),
        dtype="float32", chunks=(1, time_steps, N_LABELS),
    )
    handle.attrs["completed"] = 0
    for key, value in attrs.items():
        handle.attrs[key] = value
    return handle


def generate_shard(backend, config, seed, start, count, shard_path, max_error_cm):
    """Write trials ``[start, start + count)`` into one HDF5 shard."""
    shard_path = Path(shard_path)
    if shard_path.is_file():
        with h5py.File(shard_path, "r") as existing:
            done = int(existing.attrs.get("completed", 0))
            if done >= count and existing["data"].shape[0] == count:
                print(f"Shard already complete: {shard_path.name}", flush=True)
                return str(shard_path)
    arm, axes, shoulder, center, rest_q, rest_lengths, sample_rate, counts, times = (
        _pose_arm(backend, config)
    )
    proprioception = config.get("proprioception", {})
    spindle_bundle = spindle_coefficients(
        proprioception.get("coefficient_seed", 0),
        proprioception.get("num_afferents", 5),
    )
    attrs = _attrs(backend, arm, config, rest_lengths, seed, count)
    handle = _create_file(shard_path, count, len(times), attrs)
    rng = np.random.default_rng(seed + start * 10007)
    try:
        for local_index in range(count):
            rates, labels, error_cm = _one_trial(
                arm, rng, config, axes, shoulder, center, rest_q, rest_lengths,
                sample_rate, counts, times, spindle_bundle,
                start + local_index, max_error_cm,
            )
            handle["data"][local_index] = rates
            handle["labels"][local_index] = labels
            handle.attrs["completed"] = local_index + 1
            handle.flush()
            if (local_index + 1) % 10 == 0 or local_index + 1 == count:
                print(
                    f"  {shard_path.name}: {local_index + 1}/{count} "
                    f"(hand error {error_cm:.3f} cm)",
                    flush=True,
                )
    finally:
        handle.close()
    return str(shard_path)


def _attrs(backend, arm, config, rest_lengths, seed, count):
    return {
        "training_set": True,
        "backend": backend,
        "source": arm.source,
        "seed": int(seed),
        "n_trials": int(count),
        "muscle_names": np.array(MUSCLE_NAMES, dtype="S"),
        "joint_names": np.array(arm.free_joint_names, dtype="S"),
        "length_reference_mm": rest_lengths,
        "length_kind": "mujoco_musculotendon_actuator_mm",
        "label_layout": "wrist_xyz_cm then elv_angle shoulder_elv shoulder_rot elbow_flexion degrees",
        "distribution": (
            "Planar reaches from one braced rest pose. Direction is uniform "
            "and distance is uniform between train_min_reach_cm and train_max_reach_cm."
        ),
    }


def _shard_plan(count, workers):
    workers = max(1, min(int(workers), int(count)))
    sizes = [count // workers] * workers
    for index in range(count % workers):
        sizes[index] += 1
    plan = []
    start = 0
    for size in sizes:
        if size:
            plan.append((start, size))
            start += size
    return plan


def _merge_shards(shards, output, count):
    output = Path(output)
    if output.is_file():
        with h5py.File(output, "r") as existing:
            if existing["data"].shape[0] == count and int(existing.attrs.get("completed", 0)) >= count:
                print(f"Training file already complete: {output}", flush=True)
                return output
    first = h5py.File(shards[0], "r")
    time_steps = first["labels"].shape[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = h5py.File(output, "w")
    destination.create_dataset(
        "data", shape=(count, N_AFFERENTS, N_MUSCLES, time_steps), dtype="float32",
        chunks=(1, N_AFFERENTS, N_MUSCLES, time_steps),
    )
    destination.create_dataset(
        "labels", shape=(count, time_steps, N_LABELS), dtype="float32",
        chunks=(1, time_steps, N_LABELS),
    )
    for key, value in first.attrs.items():
        if key not in {"completed", "n_trials"}:
            destination.attrs[key] = value
    first.close()
    cursor = 0
    for shard in shards:
        with h5py.File(shard, "r") as source:
            rows = source["data"].shape[0]
            for start in range(0, rows, 8):
                stop = min(start + 8, rows)
                destination["data"][cursor + start:cursor + stop] = source["data"][start:stop]
                destination["labels"][cursor + start:cursor + stop] = source["labels"][start:stop]
            cursor += rows
            print(f"  merged {shard.name} ({cursor}/{count})", flush=True)
    destination.attrs["completed"] = cursor
    destination.attrs["n_trials"] = cursor
    destination.close()
    if cursor != count:
        raise RuntimeError(f"Merged {cursor} trials, expected {count}")
    return output


def generate(backend, num_trials, output, seed, workers, config_path, max_error_cm):
    config = _load_config(config_path)
    output = Path(output)
    if not output.is_absolute():
        output = REPO_DIR / output
    shard_dir = output.parent / f"{output.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    plan = _shard_plan(num_trials, workers)
    gigabytes = num_trials * (N_AFFERENTS * N_MUSCLES * 1152 + 1152 * N_LABELS) * 4 / 1e9
    print(
        f"Generating {num_trials} {backend} reaches -> {output}\n"
        f"About {gigabytes:.1f} GB. The trainer loads this entire file into RAM.",
        flush=True,
    )
    shard_paths = [
        shard_dir / f"shard_{start:06d}_{count:06d}.hdf5" for start, count in plan
    ]
    if len(plan) == 1:
        start, count = plan[0]
        generate_shard(backend, config, seed, start, count, shard_paths[0], max_error_cm)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(plan)) as pool:
            pool.starmap(
                generate_shard,
                [
                    (backend, config, seed, start, count, str(path), max_error_cm)
                    for (start, count), path in zip(plan, shard_paths)
                ],
            )
    merged = _merge_shards(shard_paths, output, num_trials)
    sidecar = merged.with_suffix(".json")
    with h5py.File(merged, "r") as handle:
        summary = {
            "path": str(merged),
            "trials": int(handle["data"].shape[0]),
            "data_shape": list(handle["data"].shape),
            "labels_shape": list(handle["labels"].shape),
            "backend": backend,
            "seed": seed,
        }
    sidecar.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("myosuite", "ms_human_700"), default="myosuite")
    parser.add_argument("--num-trials", type=int, default=30000)
    parser.add_argument(
        "--output",
        default=None,
        help="Training HDF5. Default: outputs_mujoco/training/<backend>_<trials>.hdf5",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-ik-error-cm", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.num_trials < 2:
        raise SystemExit("Need at least 2 trials so the loader can hold out a validation split.")
    output = args.output or (
        f"outputs_mujoco/training/{args.backend}_{args.num_trials}.hdf5"
    )
    generate(
        args.backend, args.num_trials, output, args.seed,
        args.workers, args.config, args.max_ik_error_cm,
    )


if __name__ == "__main__":
    main()
