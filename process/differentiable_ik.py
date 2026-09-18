"""Experimental differentiable replacement for ``ikcenterout.py``.

The production IK stage uses OpenSim's file-oriented InverseKinematicsTool,
which cannot participate in a PyTorch autograd graph.  This module instead:

1. fits a small polynomial forward-kinematics surrogate ``q -> wrist_xyz``
   from existing OpenSim results;
2. solves inverse kinematics with a fixed number of unrolled gradient steps;
3. keeps the resulting ``target_xyz -> q`` computation differentiable.

It is intentionally separate from the experiment manifest and writes beneath
``outputs/<experiment>/differentiable_ik``.  The fitted surrogate is local to
the joint-space region represented by the reference OpenSim trajectories; it
is not a general replacement for MoBL-ARMS outside that region.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROCESS_DIR = Path(__file__).resolve().parent
REPO_DIR = PROCESS_DIR.parent
if str(PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESS_DIR))

from experiment import load_manifest, resolve_artifact, validate_path_artifact
from paths import CENTEROUT_DIR, EXPERIMENT_CONFIG


DTYPE = torch.float64
DRIVEN_COORDINATES = (
    "elv_angle",
    "shoulder_elv",
    "shoulder_rot",
    "elbow_flexion",
)


def _powers(n_inputs: int, degree: int) -> torch.Tensor:
    """Return exponent vectors for all monomials up to ``degree``."""
    rows = [torch.zeros(n_inputs, dtype=torch.int64)]
    for order in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(range(n_inputs), order):
            power = torch.zeros(n_inputs, dtype=torch.int64)
            for index in indices:
                power[index] += 1
            rows.append(power)
    return torch.stack(rows)


class PolynomialForwardKinematics(nn.Module):
    """Differentiable local approximation of OpenSim marker kinematics."""

    def __init__(self, q_mean, q_scale, powers, coefficients):
        super().__init__()
        self.register_buffer("q_mean", torch.as_tensor(q_mean, dtype=DTYPE))
        self.register_buffer("q_scale", torch.as_tensor(q_scale, dtype=DTYPE))
        self.register_buffer("powers", torch.as_tensor(powers, dtype=torch.int64))
        self.register_buffer("coefficients", torch.as_tensor(coefficients, dtype=DTYPE))

    def features(self, q_degrees: torch.Tensor) -> torch.Tensor:
        z = (q_degrees - self.q_mean) / self.q_scale
        return torch.prod(z.unsqueeze(-2) ** self.powers, dim=-1)

    def forward(self, q_degrees: torch.Tensor) -> torch.Tensor:
        """Map ``(..., 4)`` joint angles in degrees to ``(..., 3)`` cm."""
        return self.features(q_degrees) @ self.coefficients


class ReferencePoseInitializer(nn.Module):
    """Choose the closest known local IK branch before differentiable refinement.

    The discrete branch selection is piecewise constant. Gradients with respect
    to the target are supplied by the subsequent unrolled optimization steps.
    """

    def __init__(self, xyz_reference, q_reference):
        super().__init__()
        self.register_buffer("xyz_reference", torch.as_tensor(xyz_reference, dtype=DTYPE))
        self.register_buffer("q_reference", torch.as_tensor(q_reference, dtype=DTYPE))

    def forward(self, target_xyz_cm: torch.Tensor) -> torch.Tensor:
        flat = target_xyz_cm.reshape(-1, 3)
        nearest = torch.cdist(flat, self.xyz_reference).argmin(dim=1)
        return self.q_reference[nearest].reshape(target_xyz_cm.shape[:-1] + (4,))


def fit_forward_kinematics(
    q_degrees: np.ndarray,
    wrist_xyz_cm: np.ndarray,
    degree: int = 4,
    ridge: float = 1e-8,
) -> PolynomialForwardKinematics:
    """Fit a polynomial FK model by regularized linear least squares."""
    q = torch.as_tensor(q_degrees, dtype=DTYPE)
    xyz = torch.as_tensor(wrist_xyz_cm, dtype=DTYPE)
    q_mean = q.mean(dim=0)
    q_scale = q.std(dim=0).clamp_min(1.0)
    powers = _powers(q.shape[-1], degree)

    normalized = (q - q_mean) / q_scale
    design = torch.prod(normalized.unsqueeze(1) ** powers, dim=-1)
    regularizer = ridge * torch.eye(design.shape[1], dtype=DTYPE)
    # Do not penalize the constant coefficient.
    regularizer[0, 0] = 0.0
    coefficients = torch.linalg.solve(
        design.T @ design + regularizer,
        design.T @ xyz,
    )
    return PolynomialForwardKinematics(q_mean, q_scale, powers, coefficients)


class DifferentiableIK(nn.Module):
    """Fixed-step, unrolled IK whose output remains connected to its target."""

    def __init__(
        self,
        forward_kinematics: nn.Module,
        inverse_initializer: nn.Module,
        rest_degrees: torch.Tensor,
        lower_degrees: torch.Tensor,
        upper_degrees: torch.Tensor,
        iterations: int = 80,
        learning_rate: float = 0.08,
        posture_weight: float = 1e-4,
    ):
        super().__init__()
        self.forward_kinematics = forward_kinematics
        self.inverse_initializer = inverse_initializer
        self.register_buffer("rest_degrees", torch.as_tensor(rest_degrees, dtype=DTYPE))
        self.register_buffer("lower_degrees", torch.as_tensor(lower_degrees, dtype=DTYPE))
        self.register_buffer("upper_degrees", torch.as_tensor(upper_degrees, dtype=DTYPE))
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.posture_weight = posture_weight

    def forward(self, target_xyz_cm: torch.Tensor, initial_q=None) -> torch.Tensor:
        target = target_xyz_cm.to(dtype=DTYPE)
        if initial_q is None:
            # Select the local posture branch, then differentiate through the
            # fixed sequence of refinement steps below.
            q = self.inverse_initializer(target)
        else:
            q = initial_q.to(dtype=DTYPE)

        # Functional (non-in-place) updates plus create_graph=True preserve the
        # target -> solution graph for a later loss.backward().
        for _ in range(self.iterations):
            q = q.requires_grad_(True)
            predicted = self.forward_kinematics(q)
            marker_loss = ((predicted - target) ** 2).sum()
            posture_loss = self.posture_weight * ((q - self.rest_degrees) ** 2).sum()
            gradient = torch.autograd.grad(
                marker_loss + posture_loss,
                q,
                create_graph=True,
            )[0]
            q = q - self.learning_rate * gradient
            q = torch.clamp(q, self.lower_degrees, self.upper_degrees)
        return q


def load_reference_pairs(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load joint/marker pairs previously evaluated by OpenSim."""
    q_all, xyz_all = [], []
    for trajectory in manifest["trajectories"]:
        if "ik_solution" not in trajectory or "muscle_data" not in trajectory:
            continue
        ik = np.load(resolve_artifact(trajectory["ik_solution"]))
        muscles = np.load(resolve_artifact(trajectory["muscle_data"]))
        q_all.append(np.asarray(ik["joint_angles"][:, :4], dtype=np.float64))
        xyz_all.append(np.asarray(muscles["wrist_xyz_world"], dtype=np.float64))
    if not q_all:
        raise RuntimeError(
            "No paired ik_solution and muscle_data artifacts were found. Run the "
            "existing OpenSim pipeline through muscle-lengths first."
        )
    return np.concatenate(q_all), np.concatenate(xyz_all)


def _coordinate_bounds(q_reference: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Use configured/observed bounds, with a small margin around observations."""
    # These broad anatomical limits match the four-coordinate arm convention
    # used by the paper and contain all current OpenSim solutions.
    anatomical_lower = np.array([-95.0, 0.0, -90.0, 0.0])
    anatomical_upper = np.array([130.0, 180.0, 120.0, 130.0])
    observed_lower = q_reference.min(axis=0) - 5.0
    observed_upper = q_reference.max(axis=0) + 5.0
    lower = np.maximum(anatomical_lower, observed_lower)
    upper = np.minimum(anatomical_upper, observed_upper)
    return torch.as_tensor(lower, dtype=DTYPE), torch.as_tensor(upper, dtype=DTYPE)


def _rest_pose() -> torch.Tensor:
    configured = EXPERIMENT_CONFIG["path"]["rest_pose_degrees"]
    return torch.tensor([configured[name] for name in DRIVEN_COORDINATES], dtype=DTYPE)


def verify_autograd(ik: DifferentiableIK, target: torch.Tensor) -> dict[str, float]:
    """Compare one autograd derivative with a central finite difference."""
    sample = target.detach().clone().requires_grad_(True)
    q = ik(sample)
    scalar = q[..., 0].sum()
    derivative = torch.autograd.grad(scalar, sample)[0]

    epsilon = 1e-4
    plus = target.detach().clone()
    minus = target.detach().clone()
    plus[..., 0] += epsilon
    minus[..., 0] -= epsilon
    finite_difference = (ik(plus)[..., 0] - ik(minus)[..., 0]) / (2 * epsilon)
    auto = derivative[..., 0]
    return {
        "autograd": float(auto.mean()),
        "finite_difference": float(finite_difference.detach().mean()),
        "absolute_difference": float((auto - finite_difference).abs().detach().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    args = parser.parse_args()

    torch.manual_seed(0)
    manifest = load_manifest()
    q_reference, xyz_reference = load_reference_pairs(manifest)

    # Remove repeated hold frames before splitting so identical samples cannot
    # leak into both the fit and the held-out interpolation check.
    _, unique_pose_indices = np.unique(q_reference, axis=0, return_index=True)
    q_fit = q_reference[unique_pose_indices]
    xyz_fit = xyz_reference[unique_pose_indices]

    # Hold out every fifth unique pose. This measures interpolation in the
    # current workspace; it is deliberately not an out-of-workspace test.
    indices = np.arange(len(q_fit))
    train_mask = indices % 5 != 0
    test_mask = ~train_mask
    fk = fit_forward_kinematics(
        q_fit[train_mask],
        xyz_fit[train_mask],
        degree=args.degree,
    )
    # Subsample exact duplicates to keep the nearest-reference lookup compact.
    _, unique_indices = np.unique(xyz_fit, axis=0, return_index=True)
    inverse_initializer = ReferencePoseInitializer(
        xyz_fit[unique_indices],
        q_fit[unique_indices],
    )
    with torch.no_grad():
        held_out_prediction = fk(torch.as_tensor(q_fit[test_mask], dtype=DTYPE))
        held_out_target = torch.as_tensor(xyz_fit[test_mask], dtype=DTYPE)
        fk_errors = torch.linalg.vector_norm(held_out_prediction - held_out_target, dim=-1)

    lower, upper = _coordinate_bounds(q_reference)
    ik = DifferentiableIK(
        fk,
        inverse_initializer,
        _rest_pose(),
        lower,
        upper,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )

    output_dir = Path(CENTEROUT_DIR) / "differentiable_ik"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"FK held-out marker error: mean={fk_errors.mean():.4f} cm, "
        f"p95={torch.quantile(fk_errors, 0.95):.4f} cm"
    )

    joint_differences, marker_errors = [], []
    first_target = None
    for trajectory in manifest["trajectories"]:
        desired_xyz, times = validate_path_artifact(resolve_artifact(trajectory["desired_path"]))
        target = torch.as_tensor(desired_xyz, dtype=DTYPE)
        if first_target is None:
            first_target = target[len(target) // 2 : len(target) // 2 + 1]
        solved = ik(target)
        with torch.no_grad():
            reconstructed = fk(solved)
            marker_error = torch.linalg.vector_norm(reconstructed - target, dim=-1)
        marker_errors.append(marker_error.numpy())

        if "ik_solution" in trajectory:
            reference = np.load(resolve_artifact(trajectory["ik_solution"]))["joint_angles"][:, :4]
            joint_differences.append(solved.detach().numpy() - reference)

        np.savez(
            output_dir / f"{trajectory['id']}.npz",
            joint_angles=solved.detach().numpy().astype(np.float32),
            reconstructed_xyz_cm=reconstructed.numpy().astype(np.float32),
            desired_xyz_cm=desired_xyz.astype(np.float32),
            times=times,
            coordinate_names=np.array(DRIVEN_COORDINATES),
        )

    marker_errors = np.concatenate(marker_errors)
    print(
        f"Differentiable IK marker error: mean={marker_errors.mean():.4f} cm, "
        f"p95={np.quantile(marker_errors, 0.95):.4f} cm, "
        f"max={marker_errors.max():.4f} cm"
    )
    if joint_differences:
        differences = np.concatenate(joint_differences)
        rmse = np.sqrt(np.mean(differences**2, axis=0))
        print("Joint RMSE versus OpenSim (degrees):")
        for name, value in zip(DRIVEN_COORDINATES, rmse):
            print(f"  {name}: {value:.3f}")

    check = verify_autograd(ik, first_target)
    print(
        "Gradient check dq[elv_angle]/dtarget[x]: "
        f"autograd={check['autograd']:.6f}, "
        f"finite-difference={check['finite_difference']:.6f}, "
        f"abs diff={check['absolute_difference']:.3e}"
    )
    print(f"Saved comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
