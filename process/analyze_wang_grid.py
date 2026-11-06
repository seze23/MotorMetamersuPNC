"""Save geometry-gated physical and raw errors for one Wang-grid model."""

from pathlib import Path

import numpy as np
import opensim as osm
import pandas as pd

from body_coordinates import opensim_to_body_cm
from experiment import load_manifest, resolve_artifact
from paths import EXPERIMENT_CONFIG, MODEL_PATH, PREDICTIONS_DIR, REPO_DIR


OUTPUT_DIR = Path(REPO_DIR) / "dataexp" / "wang_grid"
COORDS = ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion")


def marker_xyz(markers, name, state):
    value = markers.get(name).getLocationInGround(state)
    return np.array([value.get(i) for i in range(3)])


def main():
    proprioception = EXPERIMENT_CONFIG["proprioception"]
    coefficient_seed = int(proprioception["coefficient_seed"])
    training_seed = int(proprioception["training_seed"])
    model_name = f"{coefficient_seed}_{training_seed}"
    rate = float(EXPERIMENT_CONFIG["path"]["sample_rate_hz"])
    timing = EXPERIMENT_CONFIG["path"]["timing_seconds"]
    arrival = round(timing["hold_before"] * rate) + round(timing["reach"] * rate) - 1
    windows = {
        "arrival": [arrival],
        "plus_100ms": [arrival + round(0.100 * rate)],
        "plus_250ms": [arrival + round(0.250 * rate)],
        "plus_500ms": [arrival + round(0.500 * rate)],
        "late_hold_250ms": list(range(arrival + round(0.900 * rate),
                                      arrival + round(1.150 * rate))),
    }
    needed = sorted({index for indices in windows.values() for index in indices})

    model = osm.Model(MODEL_PATH)
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    markers = model.getMarkerSet()
    c7 = marker_xyz(markers, "C7", state)
    locked = EXPERIMENT_CONFIG["ik"]["locked_wrist_degrees"]
    tolerance = float(EXPERIMENT_CONFIG["path"]["endpoint_tolerance_cm"])
    rows = []
    decoded_pose_cache = {}

    for item in load_manifest()["trajectories"]:
        target_id = item["id"]
        path = np.load(resolve_artifact(item["desired_path"]))
        target_body = np.asarray(path["target_xyz_body"], dtype=float)
        unscaled_xy = np.asarray(path["unscaled_target_xy_body"], dtype=float)
        muscles = np.load(resolve_artifact(item["muscle_data"]))
        actual_body = np.asarray(muscles["wrist_xyz_body"], dtype=float)
        predictions = pd.read_csv(Path(PREDICTIONS_DIR) / f"{target_id}.csv")
        endpoint_error = np.linalg.norm(actual_body[arrival:] - target_body, axis=1)
        if endpoint_error.max() >= tolerance:
            raise ValueError(
                f"{target_id}: endpoint error {endpoint_error.max():.3f} cm "
                f"exceeds {tolerance:.3f} cm"
            )

        decoded_body = {}
        for index in needed:
            prediction = predictions.iloc[index]
            joint_values = np.array(
                [prediction[f"pred_{name}_deg"] for name in COORDS],
                dtype=np.float64,
            )
            pose_key = joint_values.tobytes()
            decoded = decoded_pose_cache.get(pose_key)
            if decoded is None:
                for name, degrees in zip(COORDS, joint_values):
                    coordinates.get(name).setValue(state, np.radians(degrees))
                for name, degrees in locked.items():
                    coordinates.get(name).setValue(
                        state, np.radians(float(degrees))
                    )
                model.realizePosition(state)
                decoded = opensim_to_body_cm(
                    marker_xyz(markers, "Handle", state), c7
                )
                decoded_pose_cache[pose_key] = decoded
            decoded_body[index] = decoded

        for window_name, indices in windows.items():
            actual = actual_body[indices].mean(axis=0)
            decoded = np.mean([decoded_body[index] for index in indices], axis=0)
            physical_error = decoded - actual
            raw_true = predictions.loc[indices, [
                "true_wrist_X_cm", "true_wrist_Y_cm", "true_wrist_Z_cm"
            ]].to_numpy().mean(axis=0)
            raw_pred = predictions.loc[indices, [
                "pred_wrist_X_cm", "pred_wrist_Y_cm", "pred_wrist_Z_cm"
            ]].to_numpy().mean(axis=0)
            raw_error = raw_pred - raw_true
            rows.append({
                "model": model_name, "coefficient_seed": coefficient_seed,
                "training_seed": training_seed, "target": target_id,
                "grid_row": int(path["grid_row"]),
                "grid_column": int(path["grid_column"]),
                "window": window_name,
                "uniform_grid_scale": float(path["uniform_grid_scale"]),
                "unscaled_right_cm": unscaled_xy[0],
                "unscaled_forward_cm": unscaled_xy[1],
                "requested_right_cm": target_body[0],
                "requested_forward_cm": target_body[1],
                "requested_up_cm": target_body[2],
                "actual_right_cm": actual[0], "actual_forward_cm": actual[1],
                "actual_up_cm": actual[2],
                "decoded_right_cm": decoded[0],
                "decoded_forward_cm": decoded[1], "decoded_up_cm": decoded[2],
                "error_outward_cm": physical_error[0],
                "error_forward_cm": physical_error[1],
                "error_up_cm": physical_error[2],
                "physical_l2_cm": np.linalg.norm(physical_error),
                "requested_achieved_l2_cm": np.linalg.norm(actual - target_body),
                "raw_analytic_error_x_cm": raw_error[0],
                "raw_analytic_error_y_cm": raw_error[1],
                "raw_analytic_error_z_cm": raw_error[2],
            })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"model_{model_name}_bias.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(
        f"Saved {len(rows)} rows to {output}; "
        f"evaluated {len(decoded_pose_cache)} unique decoded poses."
    )


if __name__ == "__main__":
    main()
