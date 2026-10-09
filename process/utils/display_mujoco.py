"""Play CONFIG-selected MuJoCo motion artifacts in the native viewer."""

from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment import load_manifest, resolve_artifact
from paths import EXPERIMENT_CONFIG
from mujoco_pipeline.experiment_config import arm_config, selected_model
from mujoco_pipeline.models import load_arm


def main():
    name = selected_model(EXPERIMENT_CONFIG)
    arm = load_arm(name, arm_config(EXPERIMENT_CONFIG))
    trajectories = load_manifest()["trajectories"]
    motions = [
        (item["id"], np.load(resolve_artifact(item["motion"]), allow_pickle=True))
        for item in trajectories
    ]
    if not motions:
        raise ValueError("The experiment manifest contains no motion artifacts.")
    print("Playing MuJoCo paths in manifest order. Close the viewer to continue.")
    with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
        while viewer.is_running():
            for trajectory_id, motion in motions:
                print(f"Playing {trajectory_id}")
                times = np.asarray(motion["times"], dtype=float)
                poses = motion["joint_qpos"]
                for index, pose in enumerate(poses):
                    if not viewer.is_running():
                        return
                    started = time.perf_counter()
                    arm.data.qpos[:] = pose
                    arm.data.qvel[:] = 0.0
                    mujoco.mj_forward(arm.model, arm.data)
                    viewer.sync()
                    dt = times[index + 1] - times[index] if index + 1 < len(times) else 0
                    time.sleep(max(0.0, dt - (time.perf_counter() - started)))


if __name__ == "__main__":
    main()
