"""Independently audit archived poster Handle height relative to C7."""

import io
import subprocess
from pathlib import Path

import numpy as np
import opensim as osm


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1e1dd51"
PREFIX = "dataexp/centerout/"
COORDS = [
    "elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion",
    "pro_sup", "deviation", "flexion",
]


def archived_npz(path):
    raw = subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)
    return np.load(io.BytesIO(raw))


def main():
    listing = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", COMMIT, "--", PREFIX],
        cwd=ROOT, text=True,
    ).splitlines()
    files = {
        seed: sorted(
            path for path in listing
            if path.endswith(f"_spindles_coefseed{seed}.npz")
        )
        for seed in range(5)
    }
    print("files_per_seed", {seed: len(paths) for seed, paths in files.items()})

    base = {
        Path(path).name.split("_spindles_", 1)[0]: archived_npz(path)["joint_angles"]
        for path in files[0]
    }
    max_difference = 0.0
    for seed in range(1, 5):
        for path in files[seed]:
            key = Path(path).name.split("_spindles_", 1)[0]
            difference = np.max(np.abs(archived_npz(path)["joint_angles"] - base[key]))
            max_difference = max(max_difference, float(difference))
    print("max_joint_difference_across_coefficient_copies_deg", max_difference)

    model_path = ROOT / (
        "MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/"
        "Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
    )
    model = osm.Model(str(model_path))
    state = model.initSystem()
    coordinates = [model.getCoordinateSet().get(name) for name in COORDS]
    markers = model.getMarkerSet()
    handle = markers.get("Handle")
    c7 = markers.get("C7")

    def height_cm(angles):
        for coordinate, degrees in zip(coordinates, angles):
            coordinate.setValue(state, np.deg2rad(float(degrees)))
        model.realizePosition(state)
        return 100.0 * (
            handle.getLocationInGround(state).get(1)
            - c7.getLocationInGround(state).get(1)
        )

    frame_515 = np.array([height_cm(joints[515]) for joints in base.values()])
    last_frame = np.array([height_cm(joints[-1]) for joints in base.values()])
    sampled_hold = np.array([
        [height_cm(joints[frame]) for frame in range(515, len(joints), 30)]
        for joints in base.values()
    ])

    print("frame_515_mean_median_min_max_cm", *[
        float(value) for value in (
            frame_515.mean(), np.median(frame_515), frame_515.min(), frame_515.max()
        )
    ])
    print("last_frame_mean_median_min_max_cm", *[
        float(value) for value in (
            last_frame.mean(), np.median(last_frame), last_frame.min(), last_frame.max()
        )
    ])
    print("max_abs_frame_515_vs_last_cm", float(np.max(np.abs(frame_515 - last_frame))))
    print("max_sampled_hold_range_within_target_cm", float(np.max(np.ptp(sampled_hold, axis=1))))
    print("targets_above_c7_at_frame_515", int((frame_515 > 0).sum()), "of", len(frame_515))


if __name__ == "__main__":
    main()
