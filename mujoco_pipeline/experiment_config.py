"""Translate the shared experiment CONFIG into MuJoCo arm settings."""

FREE_JOINTS = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion"]
DEFAULT_LOCKED_WRIST = {"pro_sup": -30.0, "deviation": 0.0, "flexion": 0.0}
DEFAULT_IK = {
    "iterations": 20,
    "damping": 0.001,
    "nullspace_gain": 0.02,
    "max_step_rad": 0.15,
    "tolerance_m": 0.0005,
}


def arm_config(experiment_config):
    """Return the shape expected by ``models`` and ``reaching``."""
    ik = experiment_config.get("ik", {})
    options = dict(DEFAULT_IK)
    options.update(ik.get("mujoco", {}))
    return {
        "rest_pose_degrees": dict(experiment_config["path"]["rest_pose_degrees"]),
        "locked_wrist_degrees": dict(
            ik.get("locked_wrist_degrees", DEFAULT_LOCKED_WRIST)
        ),
        "free_joints": list(FREE_JOINTS),
        "ik": options,
    }


def selected_model(experiment_config):
    """Normalize the top-level model selector."""
    name = str(experiment_config.get("musculoskeletal_model", "opensim")).lower()
    aliases = {"mobl_opensim": "opensim", "corrected_myoarm": "myosuite_corrected"}
    return aliases.get(name, name)
