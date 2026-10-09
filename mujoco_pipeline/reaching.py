"""Shoulder-centered center-out paths and damped inverse kinematics."""

import mujoco
import numpy as np

from mujoco_pipeline.models import (
    apply_joint_equalities,
    body_position,
    end_effector_position,
    set_joint_value,
)

DIRECTIONS = {
    0: "0_right",
    45: "45_fwd_right",
    90: "90_forward",
    135: "135_fwd_left",
    180: "180_left",
    225: "225_back_left",
    270: "270_backward",
    315: "315_back_right",
}


def minimum_jerk(count):
    phase = np.linspace(0, 1, count)
    return 10 * phase**3 - 15 * phase**4 + 6 * phase**5


def time_base(config):
    sample_rate = float(config["sample_rate_hz"])
    timing = config["timing_seconds"]
    segment_names = ("hold_before", "reach", "hold_target", "return", "hold_after")
    counts = {
        name: max(1, round(float(timing[name]) * sample_rate))
        for name in segment_names
    }
    n_total = sum(counts.values())
    times = np.arange(n_total, dtype=np.float64) / sample_rate
    return sample_rate, counts, times


def shoulder_axes(arm):
    """OpenSim task X, forward, and up in the MuJoCo world frame.

    The paper pipeline's ``S2W`` transform is right-handed.  The former
    construction used shoulder-minus-torso followed by ``cross(right, up)``;
    those three columns had determinant -1 and reflected lateral reaches.
    The OpenSim task-X direction for the right arm points from the shoulder
    toward the torso.  Together with ``cross(up, task_x)`` this gives the
    corresponding right-handed basis without changing any path labels or
    decoded outputs after the fact.
    """
    shoulder = body_position(arm, arm.shoulder_body_id)
    torso = body_position(arm, arm.torso_body_id)
    up = np.array([0.0, 0.0, 1.0])
    task_x = torso - shoulder
    task_x[2] = 0.0
    norm = np.linalg.norm(task_x)
    if norm < 1e-6:
        raise RuntimeError("Shoulder and torso are vertically aligned.")
    task_x /= norm
    forward = np.cross(up, task_x)
    forward /= np.linalg.norm(forward)
    axes = np.column_stack([task_x, forward, up])
    if np.linalg.det(axes) < 0.0:
        raise RuntimeError("Shoulder task frame must be right-handed.")
    return axes, shoulder


def to_shoulder_cm(point_m, shoulder_m, axes):
    return axes.T @ (point_m - shoulder_m) * 100.0


def from_shoulder_cm(point_cm, shoulder_m, axes):
    return shoulder_m + axes @ (np.asarray(point_cm, dtype=float) / 100.0)


def apply_rest_pose(arm, config):
    """Pose the four arm joints and the locked wrist, then the shoulder rhythm."""
    arm.data.qpos[:] = arm.model.qpos0
    arm.data.qvel[:] = 0.0
    rest = config["rest_pose_degrees"]
    locked = config["locked_wrist_degrees"]
    for joint_name, joint_id in zip(arm.free_joint_names, arm.free_joint_ids):
        base_name = joint_name[:-2] if joint_name.endswith("_r") else joint_name
        set_joint_value(arm, joint_id, np.deg2rad(rest[base_name]))
    for joint_name, joint_id in zip(arm.locked_joint_names, arm.locked_joint_ids):
        base_name = joint_name[:-2] if joint_name.endswith("_r") else joint_name
        set_joint_value(arm, joint_id, np.deg2rad(locked[base_name]))
    apply_joint_equalities(arm)
    end_effector_position(arm)


def make_planar_reach(center, offset, counts, times, sample_rate, trajectory_id, degrees):
    """Hold, minimum-jerk reach, hold, return, hold. ``offset`` is in centimeters."""
    reach_profile = minimum_jerk(counts["reach"])
    return_profile = minimum_jerk(counts["return"])
    target = np.asarray(center, dtype=np.float64) + np.asarray(offset, dtype=np.float64)
    xyz = np.empty((len(times), 3), dtype=np.float64)
    cursor = 0
    xyz[cursor:cursor + counts["hold_before"]] = center
    cursor += counts["hold_before"]
    xyz[cursor:cursor + counts["reach"]] = center + offset * reach_profile[:, None]
    cursor += counts["reach"]
    xyz[cursor:cursor + counts["hold_target"]] = target
    cursor += counts["hold_target"]
    xyz[cursor:cursor + counts["return"]] = target - offset * return_profile[:, None]
    cursor += counts["return"]
    xyz[cursor:] = center
    return {
        "id": trajectory_id,
        "degrees": float(degrees),
        "xyz": xyz.astype(np.float32),
        "times": times,
        "center_xyz": np.asarray(center, dtype=np.float32),
        "target_xyz": target.astype(np.float32),
        "sample_rate_hz": sample_rate,
    }


def center_out_paths(arm, config):
    """Eight minimum-jerk reaches in a horizontal plane through the rest hand."""
    apply_rest_pose(arm, config)
    axes, shoulder = shoulder_axes(arm)
    center = to_shoulder_cm(end_effector_position(arm), shoulder, axes)
    sample_rate, counts, times = time_base(config)
    reach_cm = float(config["reach_cm"])
    selected = config.get("directions_degrees")
    directions = DIRECTIONS if selected is None else {
        int(degrees): DIRECTIONS[int(degrees)] for degrees in selected
    }
    paths = []
    limit = float(config.get("max_displacement_cm", np.inf))
    for degrees, trajectory_id in directions.items():
        angle = np.radians(degrees)
        offset = reach_cm * np.array([np.cos(angle), np.sin(angle), 0.0])
        path = make_planar_reach(
            center, offset, counts, times, sample_rate, trajectory_id, degrees,
        )
        displacement = np.linalg.norm(path["xyz"] - path["xyz"][0], axis=1).max()
        if displacement > limit + 1e-6:
            raise ValueError(
                f"{trajectory_id} displaces {displacement:.2f} cm, "
                f"above the {limit:.2f} cm limit."
            )
        paths.append(path)
    return paths, axes, shoulder, center


def free_qpos(arm):
    return np.array([
        arm.data.qpos[arm.model.jnt_qposadr[joint_id]]
        for joint_id in arm.free_joint_ids
    ], dtype=float)


def _set_free_qpos(arm, values):
    for joint_id, value in zip(arm.free_joint_ids, values):
        low, high = arm.model.jnt_range[joint_id]
        arm.data.qpos[arm.model.jnt_qposadr[joint_id]] = np.clip(value, low, high)
    apply_joint_equalities(arm)


def _coupled_jacobian(arm):
    """Site Jacobian of the free joints, including polynomial couplings."""
    model, data = arm.model, arm.data
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, arm.end_effector_id)
    jacobian = np.column_stack([
        jacp[:, int(model.jnt_dofadr[joint_id])]
        for joint_id in arm.free_joint_ids
    ])
    free_index = {joint_id: column for column, joint_id in enumerate(arm.free_joint_ids)}
    for index in range(model.neq):
        if model.eq_type[index] != mujoco.mjtEq.mjEQ_JOINT or model.eq_active0[index] == 0:
            continue
        independent = int(model.eq_obj2id[index])
        dependent = int(model.eq_obj1id[index])
        column = free_index.get(independent)
        if column is None:
            continue
        if model.jnt_type[dependent] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        coefficients = model.eq_data[index]
        q_value = data.qpos[model.jnt_qposadr[independent]]
        derivative = (
            coefficients[1]
            + 2 * coefficients[2] * q_value
            + 3 * coefficients[3] * q_value**2
            + 4 * coefficients[4] * q_value**3
        )
        jacobian[:, column] += jacp[:, int(model.jnt_dofadr[dependent])] * derivative
    return jacobian


def solve_ik(arm, target_m, rest_q, ik_config):
    """Damped Gauss-Newton IK with a soft pull back toward the rest pose."""
    q_value = free_qpos(arm)
    damping = float(ik_config["damping"])
    nullspace_gain = float(ik_config["nullspace_gain"])
    max_step = float(ik_config["max_step_rad"])
    tolerance = float(ik_config["tolerance_m"])
    identity = np.eye(len(q_value))
    error = target_m - end_effector_position(arm)
    for _ in range(int(ik_config["iterations"])):
        if np.linalg.norm(error) <= tolerance:
            break
        jacobian = _coupled_jacobian(arm)
        normal = jacobian.T @ jacobian + damping * identity
        step = np.linalg.solve(normal, jacobian.T @ error)
        if nullspace_gain:
            projection = np.linalg.solve(normal, jacobian.T)
            nullspace = identity - projection @ jacobian
            step = step + nullspace_gain * (nullspace @ (rest_q - q_value))
        step_norm = np.linalg.norm(step)
        if step_norm > max_step:
            step *= max_step / step_norm
        q_value = q_value + step
        _set_free_qpos(arm, q_value)
        q_value = free_qpos(arm)
        error = target_m - end_effector_position(arm)
    return q_value, error


def track_path(arm, path, axes, shoulder_m, rest_q, ik_config):
    """Solve one reach. Identical hand targets reuse one joint solution."""
    n_frames = len(path["times"])
    n_free = len(arm.free_joint_ids)
    n_joints = arm.model.nq
    joint_qpos = np.empty((n_frames, n_joints), dtype=np.float32)
    free_qpos = np.empty((n_frames, n_free), dtype=np.float32)
    achieved = np.empty((n_frames, 3), dtype=np.float32)
    error_cm = np.empty(n_frames, dtype=np.float32)
    cache = {}
    _set_free_qpos(arm, rest_q)
    for frame_index, point_cm in enumerate(path["xyz"]):
        key = np.asarray(point_cm, dtype=np.float32).tobytes()
        cached = cache.get(key)
        if cached is None:
            target_m = from_shoulder_cm(point_cm, shoulder_m, axes)
            q_value, error = solve_ik(arm, target_m, rest_q, ik_config)
            position_cm = to_shoulder_cm(end_effector_position(arm), shoulder_m, axes)
            cached = (
                q_value.astype(np.float32),
                arm.data.qpos.copy().astype(np.float32),
                position_cm.astype(np.float32),
                np.float32(np.linalg.norm(error) * 100.0),
            )
            cache[key] = cached
        else:
            _set_free_qpos(arm, cached[0])
        free_qpos[frame_index] = cached[0]
        joint_qpos[frame_index] = cached[1]
        achieved[frame_index] = cached[2]
        error_cm[frame_index] = cached[3]
    return {
        "joint_qpos": joint_qpos,
        "free_qpos": free_qpos,
        "achieved_xyz": achieved,
        "ik_error_cm": error_cm,
        "unique_targets": len(cache),
    }
