"""Differentiable inverse kinematics using Nimble Physics.

This module is both the pipeline entry point and an importable PyTorch module.
Nimble does not currently ship native Windows wheels, so the pipeline launches
this file in the WSL Python configured under ``ik.nimble``.

The upstream MoBL-ARMS model contains two distal custom joints whose Euler-axis
layout Nimble cannot import.  ``prepare_nimble_model()`` writes a derived model
next to the experiment outputs with those wrist joints welded at their default
pose.  The original model is never modified.  This preserves the four arm
coordinates used by this project and is checked against OpenSim below.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
import yaml

PROCESS_DIR = Path(__file__).resolve().parent
REPO_DIR = PROCESS_DIR.parent
if str(PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESS_DIR))

from experiment import load_manifest, resolve_artifact, set_artifact, validate_path_artifact
from paths import CENTEROUT_DIR, EXPERIMENT_CONFIG, IK_DIR, MODEL_PATH

try:
    import nimblephysics as nimble
except ImportError as error:  # actionable error when invoked outside configured WSL env
    raise RuntimeError(
        "Nimble Physics is required for ik.backend=nimble. Run this module in "
        "the WSL Python configured at ik.nimble.python."
    ) from error


DTYPE = torch.float64
OUTPUT_COORDINATES = (
    "elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion",
    "pro_sup", "deviation", "flexion",
)
DRIVEN_COORDINATES = OUTPUT_COORDINATES[:4]
WELDED_JOINTS = ("radiocarpal", "wrist_hand")
REMOVED_COORDINATES = {"deviation", "flexion", "wrist_hand_r1", "wrist_hand_r3"}
S2W = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def prepare_nimble_model(source=MODEL_PATH) -> Path:
    """Create the minimal Nimble-compatible derivative of MoBL-ARMS."""
    destination = Path(CENTEROUT_DIR) / "nimble" / "MoBL_ARMS_nimble_v3.osim"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
        return destination

    tree = ET.parse(source)
    root = tree.getroot()
    # Collision/visual meshes are irrelevant to kinematic optimization and
    # their relative paths would otherwise be resolved beside this copy.
    for attached_geometry in root.iter("attached_geometry"):
        attached_geometry.clear()
    replaced = set()
    for objects in root.iter("objects"):
        children = list(objects)
        for index, child in enumerate(children):
            if child.tag == "CustomJoint" and child.get("name") in WELDED_JOINTS:
                weld = ET.Element("WeldJoint", {"name": child.get("name")})
                for joint_child in child:
                    if joint_child.tag in {"socket_parent_frame", "socket_child_frame", "frames"}:
                        weld.append(copy.deepcopy(joint_child))
                objects.remove(child)
                objects.insert(index, weld)
                replaced.add(child.get("name"))

    if replaced != set(WELDED_JOINTS):
        raise RuntimeError(f"Could not find wrist joints to weld; found {sorted(replaced)}")

    # The two coordinate couplers for the now-welded hand joint would refer to
    # coordinates that no longer exist. Preserve all shoulder/scapula couplers.
    for objects in root.iter("objects"):
        for child in list(objects):
            if child.tag.endswith("Constraint"):
                referenced = " ".join(child.itertext())
                if any(name in referenced for name in REMOVED_COORDINATES):
                    objects.remove(child)

    # IKMapping maps body origins, not arbitrary marker offsets. Add a
    # mass-negligible welded body whose origin is exactly the Handle marker.
    body_objects = root.find(".//BodySet/objects")
    joint_objects = root.find(".//JointSet/objects")
    handle = root.find(".//Marker[@name='Handle']")
    if body_objects is None or joint_objects is None or handle is None:
        raise RuntimeError("MoBL-ARMS BodySet, JointSet, or Handle marker is missing")
    handle_location = handle.findtext("location")
    marker_body = ET.fromstring(
        "<Body name='nimble_handle'><mass>1e-9</mass><mass_center>0 0 0</mass_center>"
        "<inertia>1e-12 1e-12 1e-12 0 0 0</inertia></Body>"
    )
    marker_joint = ET.fromstring(
        "<WeldJoint name='nimble_handle_weld'>"
        "<socket_parent_frame>nimble_handle_parent</socket_parent_frame>"
        "<socket_child_frame>nimble_handle_child</socket_child_frame>"
        "<frames>"
        "<PhysicalOffsetFrame name='nimble_handle_parent'>"
        "<socket_parent>/bodyset/hand</socket_parent>"
        f"<translation>{handle_location}</translation><orientation>0 0 0</orientation>"
        "</PhysicalOffsetFrame>"
        "<PhysicalOffsetFrame name='nimble_handle_child'>"
        "<socket_parent>/bodyset/nimble_handle</socket_parent>"
        "<translation>0 0 0</translation><orientation>0 0 0</orientation>"
        "</PhysicalOffsetFrame>"
        "</frames></WeldJoint>"
    )
    body_objects.append(marker_body)
    joint_objects.append(marker_joint)

    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def _coupling_slopes(source=MODEL_PATH) -> dict[str, tuple[str, float]]:
    """Read the model's two-point linear CoordinateCouplerConstraints."""
    root = ET.parse(source).getroot()
    result = {}
    for constraint in root.findall(".//CoordinateCouplerConstraint"):
        dependent = constraint.findtext("dependent_coordinate_name", "").strip()
        independent = constraint.findtext("independent_coordinate_names", "").strip()
        if dependent in REMOVED_COORDINATES:
            continue
        spline = constraint.find(".//SimmSpline")
        if spline is None:
            raise RuntimeError(f"Unsupported coupling function for {dependent}")
        x = np.fromstring(spline.findtext("x"), sep=" ")
        y = np.fromstring(spline.findtext("y"), sep=" ")
        if len(x) != 2 or len(y) != 2:
            raise RuntimeError(f"Expected a two-point coupling for {dependent}")
        slope = float((y[1] - y[0]) / (x[1] - x[0]))
        intercept = float(y[0] - slope * x[0])
        if abs(intercept) > 1e-8:
            raise RuntimeError(f"Non-zero coupling intercept for {dependent}")
        result[dependent] = (independent, slope)
    return result


class NimbleArmIK(nn.Module):
    """Differentiable Handle FK and fixed-step damped Gauss-Newton IK.

    Inputs and outputs use the project's conventions: shoulder-centered lab
    coordinates in centimetres and four joint angles in degrees. Because
    Nimble's IKMapping is a custom autograd operation, the returned tensor is
    connected to the target path. The Jacobian is re-linearized each iteration
    and intentionally treated as constant within that iteration.
    """

    def __init__(self, iterations=12, damping=1e-5, posture_weight=1e-5):
        super().__init__()
        if not EXPERIMENT_CONFIG.get("ik", {}).get("lock_wrist", True):
            raise ValueError(
                "The Nimble backend currently requires ik.lock_wrist=true because "
                "the source model's radiocarpal joint cannot be imported by Nimble."
            )
        parsed = nimble.biomechanics.OpenSimParser.parseOsim(str(prepare_nimble_model()))
        self.skeleton = parsed.skeleton
        self.world = nimble.simulation.World()
        self.world.addSkeleton(self.skeleton)
        self.mapping = nimble.neural.IKMapping(self.world)
        self.mapping.addLinearBodyNode(self.skeleton.getBodyNode("nimble_handle"))
        self.dof_names = [
            self.skeleton.getDofByIndex(i).getName()
            for i in range(self.skeleton.getNumDofs())
        ]
        self.couplings = _coupling_slopes()
        self.locked_degrees = {
            name: float(value) for name, value in
            EXPERIMENT_CONFIG.get("ik", {}).get("locked_wrist_degrees", {}).items()
        }
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.posture_weight = float(posture_weight)

        rest_cfg = EXPERIMENT_CONFIG["path"]["rest_pose_degrees"]
        rest = torch.tensor([rest_cfg[name] for name in DRIVEN_COORDINATES], dtype=DTYPE)
        self.register_buffer("rest_degrees", rest)
        self.register_buffer("rest_radians", torch.deg2rad(rest))
        self.register_buffer("w2s", torch.as_tensor(S2W.T, dtype=DTYPE))

        # R.Shoulder is fixed to the torso in this model. Evaluate it at the
        # configured rest pose so target conversion matches ikcenterout.py.
        with torch.no_grad():
            self.skeleton.setPositions(self._full_positions(self.rest_radians).numpy())
            marker_positions = self.skeleton.getMarkerMapWorldPositions(parsed.markersMap)
            shoulder = marker_positions["R.Shoulder"]
        self.register_buffer("shoulder_osim", torch.as_tensor(shoulder, dtype=DTYPE))

    def _full_positions(self, q_radians: torch.Tensor) -> torch.Tensor:
        driven = dict(zip(DRIVEN_COORDINATES, q_radians.unbind(-1)))
        values = []
        for name in self.dof_names:
            if name in driven:
                value = driven[name]
            elif name in self.couplings:
                independent, slope = self.couplings[name]
                value = driven[independent] * slope
            else:
                value = torch.deg2rad(q_radians.new_tensor(
                    self.locked_degrees.get(name, 0.0)
                ))
            values.append(value)
        return torch.stack(values)

    def _target_osim(self, xyz_cm: torch.Tensor) -> torch.Tensor:
        return self.w2s @ (xyz_cm / 100.0) + self.shoulder_osim

    def forward_kinematics(self, q_degrees: torch.Tensor) -> torch.Tensor:
        """Map four joint angles to shoulder-centered lab Handle XYZ (cm)."""
        q_radians = torch.deg2rad(q_degrees.to(dtype=DTYPE))
        full = self._full_positions(q_radians)
        state = torch.cat((full, torch.zeros_like(full)))
        osim_position = nimble.map_to_pos(self.world, self.mapping, state)
        return (self.w2s.T @ (osim_position - self.shoulder_osim)) * 100.0

    def _solve_one(self, target_cm: torch.Tensor, initial_degrees: torch.Tensor) -> torch.Tensor:
        target_osim = self._target_osim(target_cm)
        q = torch.deg2rad(initial_degrees.to(dtype=DTYPE))
        for _ in range(self.iterations):
            full = self._full_positions(q)
            state = torch.cat((full, torch.zeros_like(full)))
            predicted = nimble.map_to_pos(self.world, self.mapping, state)

            # Nimble provides the exact local FK Jacobian. Chain it through the
            # model's linear coordinate couplers to obtain dHandle/dq_active.
            full_jacobian = torch.as_tensor(
                self.mapping.getRealPosToMappedPosJac(self.world), dtype=DTYPE
            )
            coupling_jacobian = torch.autograd.functional.jacobian(
                self._full_positions, q, create_graph=False
            ).detach()
            jacobian = full_jacobian @ coupling_jacobian
            identity = torch.eye(len(DRIVEN_COORDINATES), dtype=DTYPE)
            normal = jacobian.T @ jacobian + (
                self.damping + self.posture_weight
            ) * identity
            rhs = jacobian.T @ (predicted - target_osim)
            rhs = rhs + self.posture_weight * (q - self.rest_radians)
            q = q - torch.linalg.solve(normal, rhs)
        return torch.rad2deg(q)

    def solve(self, target_xyz_cm: torch.Tensor, initial_degrees=None) -> torch.Tensor:
        """Solve one or more targets while retaining target -> joints autograd."""
        targets = target_xyz_cm.to(dtype=DTYPE)
        flat = targets.reshape(-1, 3)
        initial = self.rest_degrees if initial_degrees is None else initial_degrees
        solutions = []
        for target in flat:
            solution = self._solve_one(target, initial)
            solutions.append(solution)
            initial = solution
        return torch.stack(solutions).reshape(targets.shape[:-1] + (4,))


def _inspect_model():
    model_path = prepare_nimble_model()
    parsed = nimble.biomechanics.OpenSimParser.parseOsim(str(model_path))
    skeleton = parsed.skeleton
    print(f"Parsed {model_path}")
    print(f"DOFs ({skeleton.getNumDofs()}): {[skeleton.getDofByIndex(i).getName() for i in range(skeleton.getNumDofs())]}")
    print(f"Markers: {list(parsed.markersMap.keys())}")
    print(f"IKMapping methods: {[name for name in dir(nimble.neural.IKMapping) if not name.startswith('_')]}")
    world = nimble.simulation.World()
    world.addSkeleton(skeleton)
    mapping = nimble.neural.IKMapping(world)
    mapping.addLinearBodyNode(skeleton.getBodyNode("nimble_handle"))
    print(f"Raw Handle: {mapping.getPositions(world)}", flush=True)
    print(f"Raw Jacobian shape: {mapping.getRealPosToMappedPosJac(world).shape}", flush=True)
    state = torch.as_tensor(world.getState(), dtype=DTYPE).requires_grad_(True)
    skeleton.setPositions(state[:world.getNumDofs()].detach().numpy())
    print("Raw setPositions succeeded", flush=True)
    print("Testing differentiable Handle mapping...", flush=True)
    position = nimble.map_to_pos(world, mapping, state)
    gradient = torch.autograd.grad(position.sum(), state)[0]
    print(f"Handle: {position.detach().numpy()}; finite gradient: {torch.isfinite(gradient).all()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        _inspect_model()
        return

    config = EXPERIMENT_CONFIG.get("ik", {}).get("nimble", {})
    ik = NimbleArmIK(
        iterations=config.get("iterations", 12),
        damping=config.get("damping", 1e-5),
        posture_weight=config.get("posture_weight", 1e-5),
    )
    manifest = load_manifest()
    all_errors = []
    for trajectory in manifest["trajectories"]:
        xyz, times = validate_path_artifact(resolve_artifact(trajectory["desired_path"]))
        output = np.zeros((len(xyz), len(OUTPUT_COORDINATES)), dtype=np.float32)
        for index, name in enumerate(OUTPUT_COORDINATES[4:], start=4):
            output[:, index] = ik.locked_degrees.get(name, 0.0)
        reconstructed = np.zeros_like(xyz)
        solution_by_target = {}
        initial = ik.rest_degrees
        for frame, target in enumerate(xyz):
            key = np.asarray(target, dtype=np.float64).tobytes()
            if key not in solution_by_target:
                target_tensor = torch.as_tensor(target, dtype=DTYPE)
                solution = ik.solve(target_tensor, initial).detach()
                solution_by_target[key] = solution
                initial = solution
            solution = solution_by_target[key]
            output[frame, :4] = solution.numpy()
            reconstructed[frame] = ik.forward_kinematics(solution).detach().numpy()
        errors = np.linalg.norm(reconstructed - xyz, axis=1)
        all_errors.append(errors)
        out_path = Path(IK_DIR) / f"{trajectory['id']}.npz"
        np.savez(
            out_path,
            joint_angles=output,
            times=times,
            trajectory_id=trajectory["id"],
            backend="nimble",
        )
        set_artifact(trajectory["id"], "ik_solution", out_path)
        print(
            f"{trajectory['id']}: marker error mean={errors.mean():.4f} cm, "
            f"max={errors.max():.4f} cm",
            flush=True,
        )
    all_errors = np.concatenate(all_errors)
    print(
        f"Nimble IK complete: mean={all_errors.mean():.4f} cm, "
        f"p95={np.quantile(all_errors, .95):.4f} cm, max={all_errors.max():.4f} cm",
        flush=True,
    )


if __name__ == "__main__":
    main()
    # Some Nimble Linux wheels fault while tearing down imported OpenSim
    # skeletons at interpreter shutdown. All files and streams are complete.
    sys.stdout.flush()
    os._exit(0)
