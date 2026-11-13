"""Run all 4 valid seed models on the MyoArm pilot and analyze all 3 readouts:
1. Direct XYZ (primary)
2. Analytic FK (diagnostic)
3. MyoArm Physical FK (diagnostic)
"""

import os
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import yaml
import torch

ROOT = Path(r"c:\Users\brayk\OneDrive\Documents\CMU-Classes\PhiLab\MotorMetamersuPNC-myosuite")
sys.path.insert(0, str(ROOT))

from mujoco_pipeline.experiment_config import arm_config
from mujoco_pipeline.models import load_arm, set_joint_value, apply_joint_equalities, end_effector_position
from mujoco_pipeline.reaching import apply_rest_pose, shoulder_axes, to_shoulder_cm
from utils.visualize_sample import get_shoulder_elbow_wrist_loc
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset
from mujoco_pipeline.spindles import MOBL_OPTIMAL_FIBER_LENGTH_MM, firing_rates, spindle_coefficients

EXPERIMENT = os.environ.get("MYOARM_EXPERIMENT", "wang_pilot_myoarm")
CONFIG_PATH = ROOT / os.environ.get(
    "MYOARM_CONFIG", "experiments/wang_pilot_myoarm.yaml"
)
ARTIFACT_DIR = ROOT / "outputs" / EXPERIMENT
OUTPUT_DIR = ROOT / "dataexp" / EXPERIMENT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    base_config = yaml.safe_load(f)

SEEDS = [(coefficient, training) for coefficient in range(5) for training in (0, 1, 2, 9)]
TARGETS = sorted(path.stem for path in (ARTIFACT_DIR / "paths").glob("*.npz"))
if os.environ.get("MYOARM_REQUIRE_PAIRED_IK", "0") == "1":
    tolerance_cm = float(os.environ.get("MYOARM_IK_TOLERANCE_CM", "0.5"))
    passed = {}
    for target in TARGETS:
        height, location = target.split("_", 1)
        ik = np.load(ARTIFACT_DIR / "ik" / f"{target}.npz")
        passed[(height, location)] = float(np.max(ik["ik_error_cm"])) <= tolerance_cm
    valid_locations = {
        location for _, location in passed
        if passed.get(("lower", location), False)
        and passed.get(("raised", location), False)
    }
    TARGETS = [
        target for target in TARGETS if target.split("_", 1)[1] in valid_locations
    ]
    print(
        f"Paired IK filter: {len(valid_locations)} locations / "
        f"{len(passed) // 2}; {len(TARGETS)} trajectories at <= {tolerance_cm:.3f} cm"
    )
RATE_HZ = 240
PRE = round(1.65 * RATE_HZ)
REACH = round(0.50 * RATE_HZ)
ARRIVAL = PRE + REACH - 1
WINDOWS = {
    "arrival": [ARRIVAL],
    "plus_100ms": [ARRIVAL + round(0.100 * RATE_HZ)],
    "plus_250ms": [ARRIVAL + round(0.250 * RATE_HZ)],
    "plus_500ms": [ARRIVAL + round(0.500 * RATE_HZ)],
    "late_hold_250ms": list(range(ARRIVAL + round(0.900 * RATE_HZ), ARRIVAL + round(1.150 * RATE_HZ))),
}

def analytic_wrist(joint_angles_deg):
    count = len(joint_angles_deg)
    labels = np.zeros((count, 7), dtype=np.float32)
    labels[:, 3:7] = joint_angles_deg[:, :4]
    return get_shoulder_elbow_wrist_loc(labels)[2]

def run():
    arm_cfg = arm_config(base_config)
    arm = load_arm("myosuite_corrected", arm_cfg)
    apply_rest_pose(arm, arm_cfg)
    axes, shoulder = shoulder_axes(arm)

    # Preload spindle data and paths
    spindle_data = {seed: {} for seed in range(5)}
    path_data = {}
    ik_data = {}
    for target in TARGETS:
        sp = np.load(ARTIFACT_DIR / "spindles" / f"{target}.npz")
        p = np.load(ARTIFACT_DIR / "paths" / f"{target}.npz")
        ik = np.load(ARTIFACT_DIR / "ik" / f"{target}.npz")
        spindle_data[0][target] = sp
        path_data[target] = p
        ik_data[target] = ik
        muscle = np.load(ARTIFACT_DIR / "muscles" / f"{target}.npz")
        for coefficient_seed in range(1, 5):
            spindle_cfg, coefficients, sampled, muscles = spindle_coefficients(coefficient_seed, 5)
            rates, velocity, acceleration = firing_rates(
                muscle["fiber_lengths"], muscle["times"], MOBL_OPTIMAL_FIBER_LENGTH_MM,
                spindle_cfg, coefficients, sampled, muscles,
            )
            spindle_data[coefficient_seed][target] = {
                "firing_rates": rates, "joint_angles": muscle["joint_angles"], "times": muscle["times"],
            }

    all_rows = []

    for coef_seed, train_seed in SEEDS:
        model_name = f"{coef_seed}_{train_seed}"
        model_dir = ROOT / f"trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints/spatiotemporal_4_8-8-32-64_7171_{model_name}"
        print(f"\n--- Running model {model_name} ---")

        with open(model_dir / "config.yaml", "r", encoding="utf-8") as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)
        model_cfg = {k: parse_config_value(v) for k, v in model_cfg.items()}
        model_cfg["padding_mode"] = "replicate"

        # Run inference for each target
        model_rows = []
        for target in TARGETS:
            sp = spindle_data[coef_seed][target]
            ik = ik_data[target]
            p = path_data[target]
            rates = sp["firing_rates"].astype(np.float32) # (1, 10, 25, T)
            joint_angles = sp["joint_angles"].astype(np.float32) # (T, 7)
            t = sp["times"].astype(np.float64)
            T = len(t)

            true_analytic_wrist = analytic_wrist(joint_angles)
            labels = np.zeros((1, T, 7), dtype=np.float32)
            labels[0, :, 0:3] = true_analytic_wrist
            labels[0, :, 3:7] = joint_angles[:, :4]

            tmp_h5 = OUTPUT_DIR / f"_tmp_{target}_{model_name}.h5"
            with h5py.File(tmp_h5, "w") as hf:
                hf.create_dataset("data", data=rates)
                hf.create_dataset("labels", data=labels)

            dataset = SpindleDataset(
                str(tmp_h5), dataset_type="test", key="spindle_info",
                task="letter_reconstruction_joints", aclass=None,
                need_muscles=False, new_size=model_cfg["input_shape"][-1],
            )
            tester = load_model(
                model_cfg, str(model_dir), "letter_reconstruction_joints",
                torch.device("cpu"), dataset, causal=True, save_dir=ROOT,
            )
            preds, _ = tester.get_predictions()
            pred = preds[0].cpu().detach().numpy() # (T, 7)
            tmp_h5.unlink()

            # Save prediction CSV if model is 0_9 (or all models)
            pred_df = pd.DataFrame({
                "time_s": t,
                "true_wrist_X_cm": labels[0, :, 0], "true_wrist_Y_cm": labels[0, :, 1], "true_wrist_Z_cm": labels[0, :, 2],
                "true_elv_angle_deg": labels[0, :, 3], "true_shoulder_elv_deg": labels[0, :, 4],
                "true_shoulder_rot_deg": labels[0, :, 5], "true_elbow_flexion_deg": labels[0, :, 6],
                "pred_wrist_X_cm": pred[:, 0], "pred_wrist_Y_cm": pred[:, 1], "pred_wrist_Z_cm": pred[:, 2],
                "pred_elv_angle_deg": pred[:, 3], "pred_shoulder_elv_deg": pred[:, 4],
                "pred_shoulder_rot_deg": pred[:, 5], "pred_elbow_flexion_deg": pred[:, 6],
            })
            pred_df.to_csv(OUTPUT_DIR / f"predictions_{model_name}_{target}.csv", index=False)

            # Compute analytic FK of predicted joints
            pred_analytic_wrist = analytic_wrist(pred[:, 3:7])

            # Compute physical FK using MyoArm
            pred_physical_wrist = np.zeros((T, 3), dtype=np.float64)
            true_physical_wrist = ik["achieved_xyz"]

            # Compute physical FK for frames in windows
            needed_frames = sorted({idx for indices in WINDOWS.values() for idx in indices})
            for idx in needed_frames:
                pred_deg = {
                    "elv_angle_r": pred[idx, 3],
                    "shoulder_elv_r": pred[idx, 4],
                    "shoulder_rot_r": pred[idx, 5],
                    "elbow_flexion_r": pred[idx, 6],
                }
                for jname, jid in zip(arm.free_joint_names, arm.free_joint_ids):
                    set_joint_value(arm, jid, np.deg2rad(pred_deg[jname]))
                for jname, jid in zip(arm.locked_joint_names, arm.locked_joint_ids):
                    set_joint_value(arm, jid, 0.0)
                apply_joint_equalities(arm)
                pred_physical_wrist[idx] = to_shoulder_cm(end_effector_position(arm), shoulder, axes)

            target_body = p["target_xyz_body"] if "target_xyz_body" in p else np.zeros(3)

            for win_name, indices in WINDOWS.items():
                # 1. Direct XYZ (primary)
                # Convention: direct_right = -(pred_X - true_X), direct_forward = -(pred_Y - true_Y), direct_up = +(pred_Z - true_Z)
                dx_raw = (pred[indices, 0] - labels[0, indices, 0]).mean()
                dy_raw = (pred[indices, 1] - labels[0, indices, 1]).mean()
                dz_raw = (pred[indices, 2] - labels[0, indices, 2]).mean()
                direct_right = -float(dx_raw)
                direct_forward = -float(dy_raw)
                direct_up = float(dz_raw)
                direct_l2 = float(np.sqrt(dx_raw**2 + dy_raw**2 + dz_raw**2))

                # 2. Analytic FK (diagnostic)
                afk_dx_raw = (pred_analytic_wrist[indices, 0] - labels[0, indices, 0]).mean()
                afk_dy_raw = (pred_analytic_wrist[indices, 1] - labels[0, indices, 1]).mean()
                afk_dz_raw = (pred_analytic_wrist[indices, 2] - labels[0, indices, 2]).mean()
                afk_right = -float(afk_dx_raw)
                afk_forward = -float(afk_dy_raw)
                afk_up = float(afk_dz_raw)

                # 3. Physical FK (diagnostic)
                # In MuJoCo shoulder frame: error_right = -(pred_x - true_x), error_forward = -(pred_y - true_y), error_up = +(pred_z - true_z)
                phys_dx = (pred_physical_wrist[indices, 0] - true_physical_wrist[indices, 0]).mean()
                phys_dy = (pred_physical_wrist[indices, 1] - true_physical_wrist[indices, 1]).mean()
                phys_dz = (pred_physical_wrist[indices, 2] - true_physical_wrist[indices, 2]).mean()
                phys_outward = -float(phys_dx)
                phys_forward = -float(phys_dy)
                phys_up = float(phys_dz)
                phys_l2 = float(np.sqrt(phys_dx**2 + phys_dy**2 + phys_dz**2))

                row = {
                    "model": model_name,
                    "coefficient_seed": coef_seed,
                    "training_seed": train_seed,
                    "target": target,
                    "window": win_name,
                    # Direct XYZ (Primary)
                    "direct_right_cm": direct_right,
                    "direct_forward_cm": direct_forward,
                    "direct_up_cm": direct_up,
                    "direct_l2_cm": direct_l2,
                    # Analytic FK (Diagnostic)
                    "analytic_joint_fk_right_cm": afk_right,
                    "analytic_joint_fk_forward_cm": afk_forward,
                    "analytic_joint_fk_up_cm": afk_up,
                    # Physical FK (Diagnostic)
                    "physical_outward_cm": phys_outward,
                    "physical_forward_cm": phys_forward,
                    "physical_up_cm": phys_up,
                    "physical_l2_cm": phys_l2,
                    # Consistency
                    "direct_vs_analytic_fk_cm": float(np.linalg.norm([dx_raw - afk_dx_raw, dy_raw - afk_dy_raw, dz_raw - afk_dz_raw])),
                }
                model_rows.append(row)
                all_rows.append(row)

        model_df = pd.DataFrame(model_rows)
        model_df.to_csv(OUTPUT_DIR / f"pilot_model_{model_name}_bias.csv", index=False)
        late = model_df[model_df["window"] == "late_hold_250ms"]
        print(f"Model {model_name} Late-Hold Summary:")
        print(late[["target", "direct_right_cm", "analytic_joint_fk_right_cm", "physical_outward_cm"]].to_string(index=False))

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(OUTPUT_DIR / "pilot_20_models_bias.csv", index=False)

    # Ensemble summary across all 20 matched coefficient/training-seed models.
    late_all = all_df[all_df["window"] == "late_hold_250ms"]
    ensemble_summary = late_all.groupby("target", sort=False).agg(
        model_count=("model", "nunique"),
        # Direct XYZ
        direct_right_mean=("direct_right_cm", "mean"),
        direct_right_std=("direct_right_cm", "std"),
        direct_fraction_outward=("direct_right_cm", lambda x: float((x > 0).mean())),
        direct_forward_mean=("direct_forward_cm", "mean"),
        direct_l2_mean=("direct_l2_cm", "mean"),
        # Analytic FK
        analytic_right_mean=("analytic_joint_fk_right_cm", "mean"),
        analytic_fraction_outward=("analytic_joint_fk_right_cm", lambda x: float((x > 0).mean())),
        # Physical FK
        physical_outward_mean=("physical_outward_cm", "mean"),
        physical_outward_std=("physical_outward_cm", "std"),
        physical_fraction_outward=("physical_outward_cm", lambda x: float((x > 0).mean())),
        physical_forward_mean=("physical_forward_cm", "mean"),
        physical_l2_mean=("physical_l2_cm", "mean"),
    ).reset_index()
    ensemble_summary.to_csv(OUTPUT_DIR / "pilot_20_model_ensemble_summary.csv", index=False)
    print("\n=== FINAL ENSEMBLE SUMMARY (Late Hold) ===")
    print(ensemble_summary.to_string(index=False))

if __name__ == "__main__":
    run()
