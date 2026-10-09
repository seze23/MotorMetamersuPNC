"""Extract adapter fiber lengths from CONFIG-selected MuJoCo motions."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import load_manifest, resolve_artifact, set_artifact
from paths import EXPERIMENT_CONFIG, MUSCLES_DIR
from mujoco_pipeline.experiment_config import arm_config, selected_model
from mujoco_pipeline.models import load_arm
from mujoco_pipeline.spindles import (
    MOBL_CANONICAL_REST_FIBER_LENGTH_MM,
    musculotendon_to_mobl_fiber_length,
    record_muscle_lengths,
)
from mujoco_pipeline.reaching import apply_rest_pose
from utils.muscle_names import MUSCLE_NAMES


def main():
    name = selected_model(EXPERIMENT_CONFIG)
    if name != "myosuite_corrected":
        raise ValueError(
            "The shared pipeline currently supports MuJoCo fiber conversion only "
            "for musculoskeletal_model: myosuite_corrected"
        )
    config = arm_config(EXPERIMENT_CONFIG)
    arm = load_arm(name, config)
    apply_rest_pose(arm, config)
    rest_musculotendon = record_muscle_lengths(arm, arm.data.qpos[None, :])[0]
    adapter_rest = musculotendon_to_mobl_fiber_length(rest_musculotendon)
    alignment = EXPERIMENT_CONFIG.get("muscle_lengths", {}).get(
        "align_opensim_rest", True
    )
    if alignment:
        expected_pose = {
            "elv_angle": 20.0, "shoulder_elv": 40.0,
            "shoulder_rot": 25.0, "elbow_flexion": 85.0,
        }
        actual_pose = config["rest_pose_degrees"]
        if any(abs(float(actual_pose[key]) - value) > 1e-6
               for key, value in expected_pose.items()):
            raise ValueError(
                "align_opensim_rest requires the canonical rest pose "
                "(20, 40, 25, 85 degrees); disable it for another pose"
            )
        rest_offset = MOBL_CANONICAL_REST_FIBER_LENGTH_MM - adapter_rest
    else:
        rest_offset = np.zeros_like(adapter_rest)
    trajectories = load_manifest()["trajectories"]
    for trajectory in trajectories:
        motion = np.load(resolve_artifact(trajectory["motion"]), allow_pickle=True)
        musculotendon = record_muscle_lengths(arm, motion["joint_qpos"])
        fiber_unaligned = musculotendon_to_mobl_fiber_length(musculotendon)
        fiber = fiber_unaligned + rest_offset
        output = Path(MUSCLES_DIR) / f"{trajectory['id']}.npz"
        np.savez(
            output,
            times=motion["times"],
            fiber_lengths=fiber,
            adapter_fiber_lengths=fiber_unaligned,
            opensim_rest_offset_mm=rest_offset,
            musculotendon_lengths=musculotendon,
            joint_angles=motion["joint_angles"],
            wrist_xyz_world=motion["achieved_xyz"],
            elbow_xyz_world=motion["elbow_xyz_world"],
            coord_names=np.array(motion["coordinate_names"]),
            muscle_names=np.array(MUSCLE_NAMES),
            length_kind=(
                "mobl_adapter_fiber_length_opensim_rest_aligned_mm"
                if alignment else "mobl_adapter_fiber_length_mm"
            ),
            model=name,
        )
        set_artifact(trajectory["id"], "muscle_data", output)
        print(
            f"  {trajectory['id']:<20} fiber {fiber.min():.1f}-{fiber.max():.1f} mm "
            f"-> {output}"
        )
    print(f"Extracted {len(trajectories)} corrected-MyoArm motion(s).")


if __name__ == "__main__":
    main()
