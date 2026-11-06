"""Aggregate the 20-model Wang grid and save comparison statistics."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataexp" / "wang_grid"


def main():
    files = sorted(DATA_DIR.glob("model_*_bias.csv"))
    if len(files) != 20:
        raise ValueError(f"Expected 20 model files, found {len(files)}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if len(data) != 20 * 100 * 5:
        raise ValueError(f"Unexpected row count: {len(data)}")

    summary = data.groupby(["target", "grid_row", "grid_column", "window"]).agg(
        model_count=("model", "nunique"),
        unscaled_right_cm=("unscaled_right_cm", "mean"),
        unscaled_forward_cm=("unscaled_forward_cm", "mean"),
        actual_right_cm=("actual_right_cm", "mean"),
        actual_forward_cm=("actual_forward_cm", "mean"),
        decoded_right_cm=("decoded_right_cm", "mean"),
        decoded_forward_cm=("decoded_forward_cm", "mean"),
        mean_outward_cm=("error_outward_cm", "mean"),
        std_outward_cm=("error_outward_cm", "std"),
        fraction_outward=("error_outward_cm", lambda values: (values > 0).mean()),
        mean_forward_cm=("error_forward_cm", "mean"),
        std_forward_cm=("error_forward_cm", "std"),
        fraction_forward=("error_forward_cm", lambda values: (values > 0).mean()),
        mean_up_cm=("error_up_cm", "mean"),
        mean_l2_cm=("physical_l2_cm", "mean"),
        maximum_ik_error_cm=("requested_achieved_l2_cm", "max"),
    ).reset_index()
    summary.to_csv(DATA_DIR / "ensemble_target_window_summary.csv", index=False)

    late = data[data["window"] == "late_hold_250ms"].copy()
    model_summary = late.groupby(["model", "coefficient_seed", "training_seed"]).agg(
        mean_outward_cm=("error_outward_cm", "mean"),
        fraction_outward=("error_outward_cm", lambda values: (values > 0).mean()),
        mean_forward_cm=("error_forward_cm", "mean"),
        fraction_forward=("error_forward_cm", lambda values: (values > 0).mean()),
        mean_l2_cm=("physical_l2_cm", "mean"),
    ).reset_index()
    model_summary.to_csv(DATA_DIR / "model_late_hold_summary.csv", index=False)

    late_targets = summary[summary["window"] == "late_hold_250ms"].copy()
    left = late[late["unscaled_right_cm"] < 0]
    right = late[late["unscaled_right_cm"] > 0]
    near = late[late["unscaled_forward_cm"] < 33.455]
    far = late[late["unscaled_forward_cm"] > 33.455]
    comparisons = [
        ("new_all_model_targets", len(late), late.error_outward_cm.mean(),
         (late.error_outward_cm > 0).mean(), late.error_forward_cm.mean(),
         (late.error_forward_cm > 0).mean()),
        ("new_ensemble_mean_targets", len(late_targets),
         late_targets.mean_outward_cm.mean(),
         (late_targets.mean_outward_cm > 0).mean(),
         late_targets.mean_forward_cm.mean(),
         (late_targets.mean_forward_cm > 0).mean()),
        ("new_left_half", len(left), left.error_outward_cm.mean(),
         (left.error_outward_cm > 0).mean(), left.error_forward_cm.mean(),
         (left.error_forward_cm > 0).mean()),
        ("new_right_half", len(right), right.error_outward_cm.mean(),
         (right.error_outward_cm > 0).mean(), right.error_forward_cm.mean(),
         (right.error_forward_cm > 0).mean()),
        ("new_near_rows", len(near), near.error_outward_cm.mean(),
         (near.error_outward_cm > 0).mean(), near.error_forward_cm.mean(),
         (near.error_forward_cm > 0).mean()),
        ("new_far_rows", len(far), far.error_outward_cm.mean(),
         (far.error_outward_cm > 0).mean(), far.error_forward_cm.mean(),
         (far.error_forward_cm > 0).mean()),
        # Old raw analytic X was labeled rightward in that experiment.
        ("old_model_0_9_polar_raw", 72, -0.6792389861, 1 - 0.6944444444,
         -0.3943409291, 1 - 0.875),
        ("old_valid4_point_means_raw", 72, -0.8611531007, 0.0,
         -0.1717104504, 1 - 0.7083333333),
    ]
    pd.DataFrame(comparisons, columns=[
        "subset", "n", "mean_right_or_outward_cm", "fraction_right_or_outward",
        "mean_forward_cm", "fraction_forward",
    ]).to_csv(DATA_DIR / "behavior_old_model_comparison.csv", index=False)

    print(model_summary.to_string(index=False))
    print("\nOverall late-hold outward fraction:",
          (late.error_outward_cm > 0).mean())
    print("Targets with outward ensemble mean:",
          (late_targets.mean_outward_cm > 0).sum(), "/ 100")


if __name__ == "__main__":
    main()
