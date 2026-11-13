"""Paired static-endpoint poster multiring comparison at +4 and -23.2 cm."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import process.recreate_poster_multiring_corrected_tmin as recovered
import process.run_poster_to_wang_posture_bridge as bridge
import process.run_wang_workspace_relocation_pilot as inference
from process.solve_fixed_target_posture_pilot import Solver, NAMES

OUT = ROOT / "dataexp/poster_bias_investigation/multiring_height_pair_static"
TABLE = ROOT / "dataexp/poster_bias_investigation/endpoint_factor_analysis/poster_endpoint_physical_table.csv"
HEIGHTS = {"low": -23.2, "high": 4.0}
WANG_REFERENCE = np.array([27.729233, 26.808939, 45.29328, 97.10674])
POSTER_REFERENCE = np.array([45.0, 70.0, 25.0, 90.0])
PILOT_TARGETS = tuple(f"g{angle:03d}_r08" for angle in range(0, 360, 45))
PILOT_MODELS = ((0, 9), (1, 0), (2, 1), (4, 2))
N = 816


def archived_endpoint_joints():
    result = {}
    for path in recovered.coefficient_files()[0]:
        name = Path(path).name.split("_spindles_coefseed", 1)[0].replace("center_out_", "")
        result[name] = recovered.git_npz(path)["joint_angles"][recovered.PEAK, :4].astype(float)
    return result


def solve_one(target, condition, right, forward, original):
    target_dir = OUT / "ik"
    cache = target_dir / f"{condition}_{target}.npz"
    if cache.exists():
        saved = np.load(cache)
        if float(saved["error_cm"]) < 0.05:
            return target, condition, float(saved["error_cm"]), saved["q"], saved["physical"], saved["desired"]
    height = HEIGHTS[condition]
    desired = np.array([right, forward, height])
    reference = original if condition == "high" else WANG_REFERENCE
    seeds = [reference, original, WANG_REFERENCE, POSTER_REFERENCE, (original + WANG_REFERENCE) / 2]
    seeds += [np.array([e, s, r, f]) for e in (20, 45, 70) for s in (25, 55, 85)
              for r in (-60, 0, 45, 90) for f in (60, 90, 120)]
    solver = Solver()
    candidates = []
    for index, seed in enumerate(seeds):
        item = solver.solve(desired, seed, reference, 200 if index < 5 else 100)
        candidates.append(item)
        # Seeds are ordered by the desired redundant-posture reference.  Once
        # that preferred branch is physically feasible, extra branches cannot
        # improve the predeclared posture objective enough to justify solving.
        if item[1] < 0.05:
            break
    feasible = [item for item in candidates if item[1] < 0.05]
    best = min(feasible or candidates, key=lambda item: (item[1] >= 0.05, item[2], item[1]))
    physical = solver.position(np.deg2rad(best[0]))
    temporary = cache.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, target=target, condition=condition, q=best[0],
                        physical=physical, desired=desired, error_cm=best[1])
    os.replace(temporary, cache)
    return target, condition, best[1], best[0], physical, desired


def solve(targets):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ik").mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TABLE).set_index("target")
    original = archived_endpoint_joints()
    jobs = [(target, condition, table.loc[target, "physical_right_cm"],
             table.loc[target, "physical_forward_cm"], original[target])
            for target in targets for condition in HEIGHTS]
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(solve_one, *job) for job in jobs]
        for future in as_completed(futures):
            target, condition, error, q, physical, desired = future.result()
            rows.append({"target": target, "condition": condition, "error_cm": error,
                         "physical_right_cm": physical[0], "physical_forward_cm": physical[1],
                         "physical_up_cm": physical[2], "desired_right_cm": desired[0],
                         "desired_forward_cm": desired[1], "desired_up_cm": desired[2],
                         "pass_0_05_cm": error < 0.05,
                         **{f"{name}_deg": q[i] for i, name in enumerate(NAMES)}})
            print(target, condition, "error", error, flush=True)
    return pd.DataFrame(rows).sort_values(["target", "condition"])


def evaluate(audit, models, prefix):
    valid = audit.groupby("target").pass_0_05_cm.all()
    targets = valid[valid].index.tolist()
    conditions = [(target, condition) for target in targets for condition in HEIGHTS]
    q, physical = [], []
    indexed = audit.set_index(["target", "condition"])
    for target, condition in conditions:
        saved = np.load(OUT / "ik" / f"{condition}_{target}.npz")
        q.append(saved["q"]); physical.append(saved["physical"])
    q = np.asarray(q); physical = np.asarray(physical)
    trajectories = np.zeros((len(q), N, 7), dtype=np.float32)
    trajectories[:, :, :4] = q[:, None, :]
    labels = bridge.labels_from_trajectories(trajectories)
    length_path = OUT / f"{prefix}corrected_muscle_lengths.npy"
    lengths = np.load(length_path) if length_path.exists() else bridge.extract_lengths(trajectories)
    if not length_path.exists(): np.save(length_path, lengths)
    rows, distribution, rate_cache = [], [], {}
    for coefficient_seed, training_seed in models:
        if coefficient_seed not in rate_cache:
            rate_cache[coefficient_seed] = bridge.rates_from_lengths(lengths, coefficient_seed)
        rates = rate_cache[coefficient_seed]
        pad = (-len(labels)) % 25
        rates_eval = np.concatenate([rates, np.repeat(rates[-1:], pad, axis=0)]) if pad else rates
        labels_eval = np.concatenate([labels, np.repeat(labels[-1:], pad, axis=0)]) if pad else labels
        model, prediction = inference.predict(rates_eval, labels_eval, coefficient_seed, training_seed)
        prediction = prediction[:len(labels)]
        config = yaml.safe_load((bridge.MODELS / f"spatiotemporal_4_8-8-32-64_7171_{coefficient_seed}_0/config.yaml").read_text())
        label_mean = np.asarray(config["label_mean"])[0, 0, :3]
        label_std = np.asarray(config["label_std"])[0, 0, :3]
        for i, (target, condition) in enumerate(conditions):
            true = labels[i, inference.LATE, :3].mean(0)
            pred = prediction[i, inference.LATE, :3].mean(0)
            raw = pred - true
            rows.append({"model": model, "coefficient_seed": coefficient_seed,
                         "training_seed": training_seed, "target": target, "condition": condition,
                         "outward_cm": -float(raw[0]), "forward_cm": -float(raw[1]),
                         "up_cm": float(raw[2]), "l2_cm": float(np.linalg.norm(raw)),
                         "physical_right_cm": physical[i, 0], "physical_forward_cm": physical[i, 1],
                         "physical_up_cm": physical[i, 2]})
            distribution.append({"model": model, "target": target, "condition": condition,
                                 "max_abs_label_z": float(np.abs((true-label_mean)/label_std).max())})
        print("completed", model, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"{prefix}model_target_results.csv", index=False)
    pd.DataFrame(distribution).to_csv(OUT / f"{prefix}distribution_audit.csv", index=False)
    pivot = result.pivot(index=["model", "target"], columns="condition", values="outward_cm")
    pivot["high_minus_low_cm"] = pivot.high - pivot.low
    target_summary = pivot.groupby("target").agg(
        low_outward_cm=("low", "mean"), high_outward_cm=("high", "mean"),
        high_minus_low_cm=("high_minus_low_cm", "mean"),
        models_shifted_outward=("high_minus_low_cm", lambda value: int((value > 0).sum())),
    ).reset_index()
    target_summary.to_csv(OUT / f"{prefix}target_summary.csv", index=False)
    summary = {
        "paired_valid_targets": len(targets), "models": len(models),
        "low_mean_outward_cm": float(pivot.low.mean()),
        "high_mean_outward_cm": float(pivot.high.mean()),
        "high_minus_low_cm": float(pivot.high_minus_low_cm.mean()),
        "model_target_pairs_shifted_outward": int((pivot.high_minus_low_cm > 0).sum()),
        "model_target_pairs": len(pivot),
        "target_means_shifted_outward": int((target_summary.high_minus_low_cm > 0).sum()),
        "tmin_min_mm": float(lengths[:, :, 14].min()), "tmin_max_mm": float(lengths[:, :, 14].max()),
        "max_abs_label_z": float(pd.DataFrame(distribution).max_abs_label_z.max()),
    }
    (OUT / f"{prefix}result_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    targets = PILOT_TARGETS if args.pilot else tuple(pd.read_csv(TABLE).target)
    audit_path = OUT / ("pilot_ik_audit.csv" if args.pilot else "ik_audit.csv")
    audit = solve(targets)
    audit.to_csv(audit_path, index=False)
    models = PILOT_MODELS if args.pilot else tuple((c, t) for c in range(5) for t in (0, 1, 2, 9))
    evaluate(audit, models, "pilot_" if args.pilot else "")


if __name__ == "__main__":
    main()
