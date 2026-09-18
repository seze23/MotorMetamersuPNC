"""Convert manifest IK solutions to variable-length OpenSim motion files."""

import os

import numpy as np

from experiment import load_manifest, resolve_artifact, set_artifact
from paths import MOTIONS_DIR

COORD_ORDER = [
    "elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion",
    "pro_sup", "deviation", "flexion",
]


def write_mot(filepath, times, joint_angles_deg):
    lines = [
        "inDegrees=no\n", "DataType=double\n", "version=3\n",
        "OpenSimVersion=4.4-2022-10-11-798caa840\n", "endheader\n",
        "\t".join(["time"] + COORD_ORDER) + "\n",
    ]
    for time_value, angles in zip(times, joint_angles_deg):
        row = [time_value] + np.radians(angles).tolist()
        lines.append("\t".join(f"{value:.10f}" for value in row) + "\n")
    with open(filepath, "w", encoding="utf-8") as motion_file:
        motion_file.writelines(lines)


def main():
    trajectories = load_manifest()["trajectories"]
    for trajectory in trajectories:
        trajectory_id = trajectory["id"]
        ik_path = resolve_artifact(trajectory["ik_solution"])
        data = np.load(ik_path, allow_pickle=True)
        angles = data["joint_angles"]
        times = data["times"]
        if len(angles) != len(times):
            raise ValueError(f"{trajectory_id}: joint angles and times differ in length")

        output = os.path.join(MOTIONS_DIR, f"{trajectory_id}.mot")
        write_mot(output, times, angles)
        set_artifact(trajectory_id, "motion", output)
        print(f"  {trajectory_id:<20} {len(times)} frames -> {output}")
    print(f"Generated {len(trajectories)} motion files.")


if __name__ == "__main__":
    main()
