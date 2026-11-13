"""Relocate an unchanged poster joint-space reach into Wang-like baseline postures."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import process.run_poster_to_wang_posture_bridge as bridge

OUT = ROOT / "dataexp/poster_bias_investigation/poster_workspace_relocation_pilot"
CONDITIONS = {
    "poster_workspace": (),
    "relocate_elv_angle": (0,),
    "relocate_shoulder_elv": (1,),
    "relocate_shoulder_rot": (2,),
    "relocate_elbow_flexion": (3,),
    "relocate_elevation_pair": (0, 1),
    "relocate_shoulder_all": (0, 1, 2),
    "relocate_all_arm": (0, 1, 2, 3),
}


def make_conditions():
    _, poster_npz = bridge.make_trajectories()
    poster = poster_npz["joint_angles"][:bridge.N].astype(np.float32)
    wang = np.load(bridge.WANG_IK)["joint_angles"].astype(np.float32)
    offset = wang[0] - poster[0]
    trajectories, metadata = [], []
    for name, indices in CONDITIONS.items():
        trajectory = poster.copy()
        trajectory[:, list(indices)] += offset[list(indices)] if indices else 0
        trajectories.append(trajectory)
        row = {"condition": name}
        for index, coordinate in enumerate(bridge.COORDS):
            row[f"offset_{coordinate}_deg"] = offset[index] if index in indices else 0
            row[f"start_{coordinate}_deg"] = trajectory[0, index]
            row[f"endpoint_{coordinate}_deg"] = trajectory[bridge.PEAK, index]
        metadata.append(row)
    return np.stack(trajectories), pd.DataFrame(metadata)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trajectories, metadata = make_conditions()
    metadata.to_csv(OUT / "conditions.csv", index=False)
    labels = bridge.labels_from_trajectories(trajectories)
    lengths_path = OUT / "corrected_muscle_lengths.npy"
    if lengths_path.exists():
        lengths = np.load(lengths_path)
    else:
        lengths = bridge.extract_lengths(trajectories)
        np.save(lengths_path, lengths)
    rows, names = [], list(CONDITIONS)
    for coefficient_seed in range(5):
        rates = bridge.rates_from_lengths(lengths, coefficient_seed)
        for training_seed in (0, 1, 2, 9):
            rows.extend(bridge.infer(rates, labels, coefficient_seed, training_seed, names))
            print("model", coefficient_seed, training_seed, flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "model_results.csv", index=False)
    summary = results.groupby("condition", sort=False).agg(
        true_x=("true_x", "mean"), true_y=("true_y", "mean"), true_z=("true_z", "mean"),
        mean_outward_cm=("outward_cm", "mean"), std_outward_cm=("outward_cm", "std"),
        fraction_outward=("outward_cm", lambda value: float((value > 0).mean())),
    ).reset_index()
    summary.to_csv(OUT / "ensemble_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
