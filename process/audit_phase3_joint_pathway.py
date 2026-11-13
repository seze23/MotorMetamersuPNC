"""Localize Phase 3 signal mismatch and audit the joint-to-physical-FK pathway."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OPEN = ROOT / "outputs/wang_pilot"
MYO = ROOT / "outputs/wang_pilot_myoarm"
OUT = ROOT / "dataexp/wang_pilot_myoarm/joint_pathway_audit"
MUSCLES = [
    "CORB", "DELT1", "DELT2", "DELT3", "INFSP", "LAT1", "LAT2", "LAT3",
    "PECM1", "PECM2", "PECM3", "SUBSC", "SUPSP", "TMAJ", "TMIN", "ANC",
    "BIClong", "BICshort", "BRA", "BRD", "ECRL", "PT", "TRIlat", "TRIlong", "TRImed",
]
JOINTS = ["elv_angle_r", "shoulder_elv_r", "shoulder_rot_r", "elbow_flexion_r"]
PRED_COLS = ["pred_elv_angle_deg", "pred_shoulder_elv_deg", "pred_shoulder_rot_deg", "pred_elbow_flexion_deg"]
TRUE_COLS = ["true_elv_angle_deg", "true_shoulder_elv_deg", "true_shoulder_rot_deg", "true_elbow_flexion_deg"]
TARGETS = ["center", "near_center", "far_center", "center_left", "center_right"]
MODELS = [f"{coefficient}_{training}" for coefficient in range(5) for training in (0, 1, 2, 9)]
ARRIVAL = round(1.65 * 240) + round(0.50 * 240) - 1
LATE = np.arange(ARRIVAL + round(0.900 * 240), ARRIVAL + round(1.150 * 240))


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else np.nan


def main():
    from mujoco_pipeline.experiment_config import arm_config
    from mujoco_pipeline.models import load_arm, set_joint_value, apply_joint_equalities, end_effector_position
    from mujoco_pipeline.reaching import apply_rest_pose, shoulder_axes, to_shoulder_cm
    from utils.visualize_sample import get_shoulder_elbow_wrist_loc

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "experiments/wang_pilot_myoarm.yaml").read_text())
    arm_cfg = arm_config(cfg)
    arm = load_arm("myosuite_corrected", arm_cfg)
    apply_rest_pose(arm, arm_cfg)
    axes, shoulder = shoulder_axes(arm)

    signal_rows = []
    for target in TARGETS:
        a = np.load(OPEN / "spindles" / f"{target}.npz")["firing_rates"]
        b = np.load(MYO / "spindles" / f"{target}.npz")["firing_rates"]
        for idx, muscle in enumerate(MUSCLES):
            av, bv = a[:, :, idx, :], b[:, :, idx, :]
            scale = float(np.percentile(av, 99) - np.percentile(av, 1))
            rmse = float(np.sqrt(np.mean((bv - av) ** 2)))
            signal_rows.append({"target": target, "muscle": muscle, "correlation": corr(av, bv),
                                "rmse_hz": rmse, "nrmse_p1_p99": rmse / scale if scale else np.nan,
                                "max_abs_hz": float(np.max(np.abs(bv - av)))})
    signals = pd.DataFrame(signal_rows)
    signals.to_csv(OUT / "signal_agreement_by_target_muscle.csv", index=False)
    muscle_summary = signals.groupby("muscle").agg(mean_correlation=("correlation", "mean"),
        min_correlation=("correlation", "min"), mean_nrmse=("nrmse_p1_p99", "mean"),
        max_nrmse=("nrmse_p1_p99", "max"), max_abs_hz=("max_abs_hz", "max")).reset_index()
    muscle_summary.sort_values("mean_correlation").to_csv(OUT / "signal_agreement_by_muscle.csv", index=False)

    def physical(joint_deg):
        for value, jid in zip(joint_deg, arm.free_joint_ids):
            set_joint_value(arm, jid, np.deg2rad(value))
        for jid in arm.locked_joint_ids:
            set_joint_value(arm, jid, 0.0)
        apply_joint_equalities(arm)
        return to_shoulder_cm(end_effector_position(arm), shoulder, axes)

    fk_rows, decomposition = [], []
    for target in TARGETS:
        ik = np.load(MYO / "ik" / f"{target}.npz")
        sp = np.load(MYO / "spindles" / f"{target}.npz")
        joints = sp["joint_angles"][:, :4]
        labels = np.zeros((len(joints), 7), np.float32); labels[:, 3:7] = joints
        analytic = get_shoulder_elbow_wrist_loc(labels)[2]
        physical_true = np.stack([physical(joints[i]) for i in LATE])
        achieved = ik["achieved_xyz"][LATE]
        fk_rows.append({"target": target,
            "true_physical_vs_achieved_rmse_cm": float(np.sqrt(np.mean((physical_true-achieved)**2))),
            "true_physical_vs_achieved_max_cm": float(np.max(np.linalg.norm(physical_true-achieved, axis=1))),
            "analytic_vs_physical_true_mean_offset_cm": float(np.linalg.norm((analytic[LATE]-physical_true).mean(axis=0))),
            "analytic_vs_physical_true_rmse_cm": float(np.sqrt(np.mean((analytic[LATE]-physical_true)**2)))})

        for model in MODELS:
            pred = pd.read_csv(ROOT / f"dataexp/wang_pilot_myoarm/predictions_{model}_{target}.csv")
            true_j = pred.loc[LATE, TRUE_COLS].to_numpy()
            pred_j = pred.loc[LATE, PRED_COLS].to_numpy()
            true_xyz = np.stack([physical(x) for x in true_j])
            base_xyz = np.stack([physical(x) for x in pred_j])
            base_right = float(-(base_xyz[:, 0]-true_xyz[:, 0]).mean())
            decomposition.append({"model": model, "target": target, "restored_joint": "none",
                                  "physical_right_cm": base_right, "change_from_all_predicted_cm": 0.0})
            for j, name in enumerate(JOINTS):
                mixed = pred_j.copy(); mixed[:, j] = true_j[:, j]
                xyz = np.stack([physical(x) for x in mixed])
                right = float(-(xyz[:, 0]-true_xyz[:, 0]).mean())
                decomposition.append({"model": model, "target": target, "restored_joint": name,
                                      "physical_right_cm": right, "change_from_all_predicted_cm": right-base_right})
    pd.DataFrame(fk_rows).to_csv(OUT / "true_joint_fk_validation.csv", index=False)
    dec = pd.DataFrame(decomposition)
    dec.to_csv(OUT / "one_joint_restoration.csv", index=False)
    dec.groupby("restored_joint").agg(mean_physical_right_cm=("physical_right_cm", "mean"),
        mean_change_cm=("change_from_all_predicted_cm", "mean"),
        max_abs_change_cm=("change_from_all_predicted_cm", lambda x: float(np.max(np.abs(x))))).reset_index().to_csv(OUT / "one_joint_restoration_summary.csv", index=False)

    summary = {"worst_signal_muscles": muscle_summary.sort_values("mean_correlation").head(5).to_dict("records"),
               "true_fk": fk_rows,
               "decomposition": pd.read_csv(OUT / "one_joint_restoration_summary.csv").to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
