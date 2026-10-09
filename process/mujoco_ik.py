"""Solve manifest paths with the CONFIG-selected MuJoCo arm."""

from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import load_manifest, resolve_artifact, set_artifact, validate_path_artifact
from paths import EXPERIMENT_CONFIG, IK_DIR
from mujoco_pipeline.experiment_config import arm_config, selected_model
from mujoco_pipeline.models import body_position, load_arm
from mujoco_pipeline.reaching import (
    apply_rest_pose,
    free_qpos,
    shoulder_axes,
    to_shoulder_cm,
    track_path,
)

COORD_NAMES = [
    "elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion",
    "pro_sup", "deviation", "flexion",
]


def _body_trajectory(arm, qpos, body_id, axes, shoulder):
    positions = np.empty((len(qpos), 3), dtype=np.float32)
    for index, pose in enumerate(qpos):
        arm.data.qpos[:] = pose
        arm.data.qvel[:] = 0.0
        mujoco.mj_forward(arm.model, arm.data)
        positions[index] = to_shoulder_cm(
            body_position(arm, body_id), shoulder, axes
        )
    return positions


def main():
    name = selected_model(EXPERIMENT_CONFIG)
    config = arm_config(EXPERIMENT_CONFIG)
    arm = load_arm(name, config)
    apply_rest_pose(arm, config)
    axes, shoulder = shoulder_axes(arm)
    rest_q = free_qpos(arm)
    locked = np.array(
        [config["locked_wrist_degrees"][key] for key in COORD_NAMES[4:]],
        dtype=np.float32,
    )

    trajectories = load_manifest()["trajectories"]
    for trajectory in trajectories:
        trajectory_id = trajectory["id"]
        path_file = resolve_artifact(trajectory["desired_path"])
        xyz, times = validate_path_artifact(path_file)
        solved = track_path(
            arm, {"xyz": xyz, "times": times}, axes, shoulder, rest_q, config["ik"]
        )
        joint_angles = np.empty((len(times), 7), dtype=np.float32)
        joint_angles[:, :4] = np.degrees(solved["free_qpos"])
        joint_angles[:, 4:] = locked
        elbow = _body_trajectory(
            arm, solved["joint_qpos"], arm.elbow_body_id, axes, shoulder
        )
        output = Path(IK_DIR) / f"{trajectory_id}.npz"
        np.savez(
            output,
            times=times,
            joint_angles=joint_angles,
            joint_qpos=solved["joint_qpos"],
            free_qpos=solved["free_qpos"],
            achieved_xyz=solved["achieved_xyz"],
            elbow_xyz_world=elbow,
            ik_error_cm=solved["ik_error_cm"],
            coordinate_names=np.array(COORD_NAMES),
            model=name,
        )
        set_artifact(trajectory_id, "ik_solution", output)
        print(
            f"  {trajectory_id:<20} {len(times)} frames | "
            f"IK mean {solved['ik_error_cm'].mean():.3f} cm, "
            f"max {solved['ik_error_cm'].max():.3f} cm"
        )
    print(f"Solved {len(trajectories)} path(s) with {name}.")


if __name__ == "__main__":
    main()
