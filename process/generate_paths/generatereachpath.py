"""Generate configurable minimum-jerk center-out path artifacts."""

import os
import sys
from pathlib import Path

import numpy as np
import opensim as osm

# This file is also launched directly when selected in an experiment YAML.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import create_manifest, relative_artifact, validate_path_artifact
from paths import EXPERIMENT_CONFIG, MODEL_PATH, PATHS_DIR

S2W = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
DIRECTIONS = {
    0: "0_right", 45: "45_fwd_right", 90: "90_forward",
    135: "135_fwd_left", 180: "180_left", 225: "225_back_left",
    270: "270_backward", 315: "315_back_right",
}


def minimum_jerk(count):
    phase = np.linspace(0, 1, count)
    return 10 * phase**3 - 15 * phase**4 + 6 * phase**5


def model_rest_center(rest_pose):
    model = osm.Model(MODEL_PATH)
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    pose = dict(rest_pose)
    if EXPERIMENT_CONFIG.get("ik", {}).get("lock_wrist", True):
        pose.update(EXPERIMENT_CONFIG.get("ik", {}).get("locked_wrist_degrees", {}))
    for name, degrees in pose.items():
        coordinates.get(name).setValue(state, np.radians(degrees))
    model.realizePosition(state)
    markers = model.getMarkerSet()
    shoulder = markers.get("R.Shoulder").getLocationInGround(state)
    handle = markers.get("Handle").getLocationInGround(state)
    shoulder = np.array([shoulder.get(i) for i in range(3)])
    handle = np.array([handle.get(i) for i in range(3)])
    return S2W @ (handle - shoulder) * 100.0


def generate():
    config = EXPERIMENT_CONFIG["path"]
    sample_rate = float(config["sample_rate_hz"])
    reach_cm = float(config["reach_cm"])
    rest_pose = config["rest_pose_degrees"]
    timing = config["timing_seconds"]
    segment_names = ("hold_before", "reach", "hold_target", "return", "hold_after")
    counts = {name: max(1, round(float(timing[name]) * sample_rate))
              for name in segment_names}
    n_total = sum(counts.values())
    times = np.arange(n_total, dtype=np.float64) / sample_rate
    center = model_rest_center(rest_pose)
    reach_profile = minimum_jerk(counts["reach"])
    return_profile = minimum_jerk(counts["return"])
    trajectories = []

    for degrees, trajectory_id in DIRECTIONS.items():
        angle = np.radians(degrees)
        target = center + reach_cm * np.array([np.cos(angle), np.sin(angle), 0.0])
        xyz = np.empty((n_total, 3), dtype=np.float32)
        cursor = 0
        xyz[cursor:cursor + counts["hold_before"]] = center
        cursor += counts["hold_before"]
        xyz[cursor:cursor + counts["reach"]] = (
            center + (target - center) * reach_profile[:, None]
        )
        cursor += counts["reach"]
        xyz[cursor:cursor + counts["hold_target"]] = target
        cursor += counts["hold_target"]
        xyz[cursor:cursor + counts["return"]] = (
            target + (center - target) * return_profile[:, None]
        )
        cursor += counts["return"]
        xyz[cursor:] = center

        output = os.path.join(PATHS_DIR, f"{trajectory_id}.npz")
        np.savez(
            output,
            xyz=xyz,
            times=times,
            center_xyz=center,
            target_xyz=target,
            rest_posture=np.array(list(rest_pose.values())),
            trajectory_id=trajectory_id,
            position_units="cm",
            coordinate_frame="shoulder_centered_world",
            sample_rate_hz=sample_rate,
        )
        validate_path_artifact(output, config)
        trajectories.append({
            "id": trajectory_id,
            "desired_path": relative_artifact(output),
            "samples": n_total,
            "sample_rate_hz": sample_rate,
        })
        print(f"  {trajectory_id:<20} {n_total} samples -> {output}")

    create_manifest(trajectories)
    print(f"Generated {len(trajectories)} validated paths and manifest.")


if __name__ == "__main__":
    generate()
