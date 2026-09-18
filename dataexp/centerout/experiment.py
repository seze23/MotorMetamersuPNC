"""Experiment manifest and path-artifact validation helpers."""

from pathlib import Path
import os

import numpy as np
import yaml

from paths import CENTEROUT_DIR, EXPERIMENT_CONFIG, EXPERIMENT_NAME, MANIFEST_PATH


def relative_artifact(path):
    return Path(path).resolve().relative_to(Path(CENTEROUT_DIR).resolve()).as_posix()


def resolve_artifact(path):
    resolved = (Path(CENTEROUT_DIR) / path).resolve()
    resolved.relative_to(Path(CENTEROUT_DIR).resolve())
    return str(resolved)


def load_manifest(required=True):
    if not os.path.exists(MANIFEST_PATH):
        if required:
            raise FileNotFoundError(
                f"No manifest found at {MANIFEST_PATH}. Run the path stage first."
            )
        return {"experiment": EXPERIMENT_NAME, "trajectories": []}
    with open(MANIFEST_PATH, encoding="utf-8") as manifest_file:
        return yaml.safe_load(manifest_file)


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as manifest_file:
        yaml.safe_dump(manifest, manifest_file, sort_keys=False)


def create_manifest(trajectories):
    ids = [item["id"] for item in trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError("Trajectory IDs must be unique within an experiment.")
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "config": os.path.relpath(EXPERIMENT_CONFIG["_config_path"], CENTEROUT_DIR),
        "trajectories": trajectories,
    }
    save_manifest(manifest)
    return manifest


def set_artifact(trajectory_id, key, path):
    manifest = load_manifest()
    for trajectory in manifest["trajectories"]:
        if trajectory["id"] == trajectory_id:
            trajectory[key] = relative_artifact(path)
            save_manifest(manifest)
            return
    raise KeyError(f"Unknown trajectory ID: {trajectory_id}")


def validate_path_artifact(path, constraints=None):
    constraints = constraints or EXPERIMENT_CONFIG.get("path", {})
    data = np.load(path, allow_pickle=True)
    if "xyz" not in data or "times" not in data:
        raise ValueError(f"{path} must contain 'xyz' and 'times' arrays")
    xyz = np.asarray(data["xyz"], dtype=float)
    times = np.asarray(data["times"], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2:
        raise ValueError(f"{path}: xyz must have shape (N, 3), N >= 2")
    if times.shape != (len(xyz),):
        raise ValueError(f"{path}: times must have shape ({len(xyz)},)")
    if not np.isfinite(xyz).all() or not np.isfinite(times).all():
        raise ValueError(f"{path}: path data must be finite")
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"{path}: times must be strictly increasing")
    max_allowed = float(constraints.get("max_displacement_cm", np.inf))
    displacement = np.linalg.norm(xyz - xyz[0], axis=1)
    if displacement.max() > max_allowed + 1e-6:
        raise ValueError(
            f"{path}: maximum displacement {displacement.max():.2f} cm "
            f"exceeds configured limit {max_allowed:.2f} cm"
        )
    return xyz, times
