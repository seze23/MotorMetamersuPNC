"""Generate all 100 uniformly scaled Wang-grid center-out trajectories."""

import os
import sys
from pathlib import Path

import numpy as np
import opensim as osm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from body_coordinates import body_to_legacy_shoulder_cm, opensim_to_body_cm
from experiment import create_manifest, relative_artifact, validate_path_artifact
from paths import EXPERIMENT_CONFIG, MODEL_PATH, PATHS_DIR


X_VALUES = np.array([(column - 9.5) * 2.438 for column in range(20)])
Y_VALUES = np.array([22.671, 28.063, 33.455, 38.847, 44.239])
CENTER_XY = np.array([0.0, 33.455])


def marker_xyz(markers, name, state):
    value = markers.get(name).getLocationInGround(state)
    return np.array([value.get(i) for i in range(3)])


def minimum_jerk(count):
    phase = np.linspace(0.0, 1.0, count)
    return 10 * phase**3 - 15 * phase**4 + 6 * phase**5


def reference_geometry():
    model = osm.Model(MODEL_PATH)
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    pose = dict(EXPERIMENT_CONFIG["path"]["rest_pose_degrees"])
    pose.update(EXPERIMENT_CONFIG["ik"]["locked_wrist_degrees"])
    for name, degrees in pose.items():
        coordinates.get(name).setValue(state, np.radians(float(degrees)))
    model.realizePosition(state)
    markers = model.getMarkerSet()
    c7 = marker_xyz(markers, "C7", state)
    shoulder = opensim_to_body_cm(marker_xyz(markers, "R.Shoulder", state), c7)
    handle = opensim_to_body_cm(marker_xyz(markers, "Handle", state), c7)
    return shoulder, handle[2]


def generate():
    config = EXPERIMENT_CONFIG["path"]
    scale = float(config["uniform_grid_scale"])
    if not 0.0 < scale <= 1.0:
        raise ValueError("path.uniform_grid_scale must be in (0, 1]")
    rate = float(config["sample_rate_hz"])
    timing = config["timing_seconds"]
    pre_count = round(float(timing["hold_before"]) * rate)
    reach_count = round(float(timing["reach"]) * rate)
    endpoint_count = round(float(timing["hold_target"]) * rate)
    n_total = pre_count + reach_count + endpoint_count
    times = np.arange(n_total, dtype=np.float64) / rate
    profile = minimum_jerk(reach_count)
    shoulder_body, table_height = reference_geometry()
    center_body = np.array([*CENTER_XY, table_height])

    trajectories = []
    for row, original_y in enumerate(Y_VALUES):
        for column, original_x in enumerate(X_VALUES):
            unscaled_xy = np.array([original_x, original_y])
            scaled_xy = CENTER_XY + scale * (unscaled_xy - CENTER_XY)
            target_body = np.array([*scaled_xy, table_height])
            body_path = np.empty((n_total, 3), dtype=np.float32)
            body_path[:pre_count] = center_body
            body_path[pre_count:pre_count + reach_count] = (
                center_body + (target_body - center_body) * profile[:, None]
            )
            body_path[pre_count + reach_count:] = target_body
            legacy_path = body_to_legacy_shoulder_cm(body_path, shoulder_body).astype(
                np.float32
            )
            target_id = f"r{row:02d}_c{column:02d}"
            output = os.path.join(PATHS_DIR, f"{target_id}.npz")
            np.savez(
                output, xyz=legacy_path, times=times,
                center_xyz=legacy_path[0], target_xyz=legacy_path[-1],
                xyz_body=body_path, center_xyz_body=center_body,
                target_xyz_body=target_body,
                unscaled_target_xy_body=unscaled_xy,
                grid_row=row, grid_column=column, uniform_grid_scale=scale,
                trajectory_id=target_id, position_units="cm",
                coordinate_frame="legacy_shoulder_input_with_body_metadata",
                sample_rate_hz=rate,
            )
            validate_path_artifact(output, config)
            trajectories.append({
                "id": target_id, "desired_path": relative_artifact(output),
                "samples": n_total, "sample_rate_hz": rate,
                "grid_row": row, "grid_column": column,
            })
    create_manifest(trajectories)
    print(f"Generated {len(trajectories)} Wang grid paths at scale {scale:.3f}.")


if __name__ == "__main__":
    generate()
