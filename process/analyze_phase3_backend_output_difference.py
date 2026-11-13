"""Quantify decoder-output differences between paired OpenSim and MyoArm pilots."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataexp/wang_pilot_myoarm/joint_pathway_audit"
TARGETS = ["center", "near_center", "far_center", "center_left", "center_right"]
PRED = ["pred_wrist_X_cm", "pred_wrist_Y_cm", "pred_wrist_Z_cm", "pred_elv_angle_deg",
        "pred_shoulder_elv_deg", "pred_shoulder_rot_deg", "pred_elbow_flexion_deg"]
ARRIVAL = round(1.65 * 240) + round(0.50 * 240) - 1
LATE = np.arange(ARRIVAL + round(.9 * 240), ARRIVAL + round(1.15 * 240))

rows = []
for target in TARGETS:
    opensim = pd.read_csv(ROOT / f"outputs/wang_pilot/predictions/{target}.csv")
    myoarm = pd.read_csv(ROOT / f"dataexp/wang_pilot_myoarm/predictions_0_9_{target}.csv")
    delta = myoarm[PRED].to_numpy() - opensim[PRED].to_numpy()
    row = {"target": target}
    for index, name in enumerate(PRED):
        row[f"{name}_trajectory_rmse"] = float(np.sqrt(np.mean(delta[:, index] ** 2)))
        row[f"{name}_late_mean_difference"] = float(delta[LATE, index].mean())
    for prefix, frame in (("opensim", opensim), ("myoarm", myoarm)):
        row[f"{prefix}_direct_right_cm"] = float(-(frame.loc[LATE, "pred_wrist_X_cm"] - frame.loc[LATE, "true_wrist_X_cm"]).mean())
    row["direct_right_difference_cm"] = row["myoarm_direct_right_cm"] - row["opensim_direct_right_cm"]
    rows.append(row)

table = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT / "backend_decoder_output_difference_model_0_9.csv", index=False)
summary = {
    "scope": "Paired target names, model 0_9; includes backend-specific IK/trajectory and spindle differences",
    "mean_abs_direct_right_difference_cm": float(table.direct_right_difference_cm.abs().mean()),
    "max_abs_direct_right_difference_cm": float(table.direct_right_difference_cm.abs().max()),
    "mean_direct_right_difference_cm": float(table.direct_right_difference_cm.mean()),
    "target_rows": rows,
}
(OUT / "backend_decoder_output_difference_model_0_9.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({key: value for key, value in summary.items() if key != "target_rows"}, indent=2))
