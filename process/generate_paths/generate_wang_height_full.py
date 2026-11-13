"""Generate the full 20x5 Wang grid at matched lower and raised heights."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from body_coordinates import body_to_legacy_shoulder_cm
from experiment import create_manifest, relative_artifact, validate_path_artifact
from paths import EXPERIMENT_CONFIG, PATHS_DIR
from generate_wang_grid import X_VALUES, Y_VALUES, CENTER_XY, minimum_jerk, reference_geometry

HEIGHTS_CM = {"lower": -23.2, "raised": 4.0}


def generate():
    config = EXPERIMENT_CONFIG["path"]
    scale = float(config["uniform_grid_scale"])
    rate = float(config["sample_rate_hz"])
    timing = config["timing_seconds"]
    pre_count = round(float(timing["hold_before"]) * rate)
    reach_count = round(float(timing["reach"]) * rate)
    endpoint_count = round(float(timing["hold_target"]) * rate)
    n_total = pre_count + reach_count + endpoint_count
    times = np.arange(n_total, dtype=np.float64) / rate
    profile = minimum_jerk(reach_count)
    shoulder_body, _ = reference_geometry()
    trajectories = []

    for height_name, height_cm in HEIGHTS_CM.items():
        center_body = np.array([*CENTER_XY, height_cm], dtype=float)
        for row, original_y in enumerate(Y_VALUES):
            for column, original_x in enumerate(X_VALUES):
                unscaled_xy = np.array([original_x, original_y])
                scaled_xy = CENTER_XY + scale * (unscaled_xy - CENTER_XY)
                target_body = np.array([*scaled_xy, height_cm], dtype=float)
                body_path = np.empty((n_total, 3), dtype=np.float32)
                body_path[:pre_count] = center_body
                body_path[pre_count:pre_count + reach_count] = (
                    center_body + (target_body - center_body) * profile[:, None]
                )
                body_path[pre_count + reach_count:] = target_body
                legacy_path = body_to_legacy_shoulder_cm(body_path, shoulder_body).astype(np.float32)
                target_id = f"{height_name}_r{row:02d}_c{column:02d}"
                output = os.path.join(PATHS_DIR, f"{target_id}.npz")
                np.savez(
                    output, xyz=legacy_path, times=times,
                    center_xyz=legacy_path[0], target_xyz=legacy_path[-1],
                    xyz_body=body_path, center_xyz_body=center_body,
                    target_xyz_body=target_body, unscaled_target_xy_body=unscaled_xy,
                    grid_row=row, grid_column=column, height_name=height_name,
                    height_cm=height_cm, uniform_grid_scale=scale,
                    trajectory_id=target_id, position_units="cm",
                    coordinate_frame="legacy_shoulder_input_with_body_metadata",
                    sample_rate_hz=rate,
                )
                validate_path_artifact(output, config)
                trajectories.append({"id": target_id, "desired_path": relative_artifact(output),
                                     "samples": n_total, "sample_rate_hz": rate,
                                     "grid_row": row, "grid_column": column,
                                     "height_name": height_name})
    create_manifest(trajectories)
    print(f"Generated {len(trajectories)} paired full-grid paths.")


if __name__ == "__main__":
    generate()
