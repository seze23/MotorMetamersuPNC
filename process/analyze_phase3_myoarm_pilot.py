"""Paired signal and direct-XYZ audit for the five-target corrected-MyoArm pilot."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OPEN = ROOT / "outputs/wang_pilot"
MYO = ROOT / "outputs/wang_pilot_myoarm"
OUT = ROOT / "dataexp/wang_pilot_myoarm"
ARRIVAL = round(1.65 * 240) + round(0.50 * 240) - 1
LATE = list(range(ARRIVAL + round(0.900 * 240), ARRIVAL + round(1.150 * 240)))


def correlation(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = yaml.safe_load((MYO / "manifest.yaml").read_text(encoding="utf-8"))
    rows = []
    all_open, all_myo = [], []
    for item in manifest["trajectories"]:
        target = item["id"]
        open_rates = np.load(OPEN / "spindles" / f"{target}.npz")["firing_rates"]
        myo_rates = np.load(MYO / "spindles" / f"{target}.npz")["firing_rates"]
        all_open.append(open_rates)
        all_myo.append(myo_rates)
        difference = myo_rates - open_rates
        reference_range = float(np.percentile(open_rates, 99) - np.percentile(open_rates, 1))
        myo_motion = np.load(MYO / "ik" / f"{target}.npz")
        pred = pd.read_csv(MYO / "predictions" / f"{target}.csv")
        direct_right = float((-(pred.loc[LATE, "pred_wrist_X_cm"] - pred.loc[LATE, "true_wrist_X_cm"])).mean())
        rows.append({
            "target": target,
            "spindle_correlation": correlation(open_rates, myo_rates),
            "spindle_rmse_hz": float(np.sqrt(np.mean(difference**2))),
            "spindle_nrmse_p1_p99": float(np.sqrt(np.mean(difference**2)) / reference_range),
            "max_abs_spindle_difference_hz": float(np.max(np.abs(difference))),
            "myo_max_ik_error_cm": float(myo_motion["ik_error_cm"].max()),
            "myo_direct_right_cm": direct_right,
            "myo_tmin_rate_max_hz": float(myo_rates[:, :, 14, :].max()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "target_summary.csv", index=False)
    open_all = np.stack(all_open)
    myo_all = np.stack(all_myo)
    diff = myo_all - open_all
    reference_range = float(np.percentile(open_all, 99) - np.percentile(open_all, 1))
    summary = {
        "aggregate_spindle_correlation": correlation(open_all, myo_all),
        "aggregate_spindle_rmse_hz": float(np.sqrt(np.mean(diff**2))),
        "aggregate_spindle_nrmse_p1_p99": float(np.sqrt(np.mean(diff**2)) / reference_range),
        "minimum_target_correlation": float(result["spindle_correlation"].min()),
        "maximum_target_nrmse": float(result["spindle_nrmse_p1_p99"].max()),
        "maximum_ik_error_cm": float(result["myo_max_ik_error_cm"].max()),
        "mean_direct_right_cm": float(result["myo_direct_right_cm"].mean()),
        "fraction_targets_outward": float((result["myo_direct_right_cm"] > 0).mean()),
        "maximum_tmin_rate_hz": float(result["myo_tmin_rate_max_hz"].max()),
        "gate_correlation_at_least_0_99": bool(correlation(open_all, myo_all) >= 0.99),
        "gate_nrmse_at_most_0_05": bool(np.sqrt(np.mean(diff**2)) / reference_range <= 0.05),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
