"""Compare suspect OpenSim/MyoArm fiber lengths at identical joint postures."""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import opensim as osm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = ROOT / "MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
OUT = ROOT / "dataexp/wang_pilot_myoarm/joint_pathway_audit"
TARGETS = ["center", "near_center", "far_center", "center_left", "center_right"]
MUSCLES = ["PECM1", "TMAJ", "LAT2", "LAT3", "TMIN"]
COORDS = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]


def main():
    model = osm.Model(str(MODEL)); state = model.initSystem()
    coordinates = [model.getCoordinateSet().get(name) for name in COORDS]
    muscles = [model.getMuscles().get(name) for name in MUSCLES]
    rows = []
    for target in TARGETS:
        myo = np.load(ROOT / f"outputs/wang_pilot_myoarm/muscles/{target}.npz")
        joint = myo["joint_angles"]
        myo_names = list(myo["muscle_names"].astype(str))
        indices = np.unique(np.linspace(0, len(joint)-1, 81).round().astype(int))
        for frame in indices:
            for coordinate, degrees in zip(coordinates, joint[frame]):
                coordinate.setValue(state, np.deg2rad(float(degrees)))
            model.equilibrateMuscles(state)
            for muscle, name in zip(muscles, MUSCLES):
                rows.append({"target": target, "frame": int(frame), "muscle": name,
                    "opensim_fiber_mm": float(muscle.getFiberLength(state) * 1000),
                    "myoarm_fiber_mm": float(myo["fiber_lengths"][frame, myo_names.index(name)])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "matched_posture_suspect_muscle_lengths.csv", index=False)
    summaries = []
    for name, group in table.groupby("muscle"):
        x = group.opensim_fiber_mm.to_numpy(); y = group.myoarm_fiber_mm.to_numpy()
        raw_rmse = float(np.sqrt(np.mean((y-x)**2)))
        offset = float(np.mean(x-y)); offset_residual = y + offset - x
        design = np.column_stack([np.ones(len(y)), y])
        intercept, slope = np.linalg.lstsq(design, x, rcond=None)[0]
        affine_residual = intercept + slope*y - x
        summaries.append({"muscle": name, "correlation": float(np.corrcoef(x,y)[0,1]),
            "raw_rmse_mm": raw_rmse, "required_offset_to_myoarm_mm": offset,
            "offset_only_rmse_mm": float(np.sqrt(np.mean(offset_residual**2))),
            "affine_intercept_mm": float(intercept), "affine_slope": float(slope),
            "affine_rmse_mm": float(np.sqrt(np.mean(affine_residual**2))),
            "opensim_range_mm": float(np.ptp(x)), "myoarm_range_mm": float(np.ptp(y))})
    summary = pd.DataFrame(summaries).sort_values("offset_only_rmse_mm", ascending=False)
    summary.to_csv(OUT / "matched_posture_suspect_muscle_summary.csv", index=False)
    (OUT / "matched_posture_suspect_muscle_summary.json").write_text(json.dumps(summary.to_dict("records"), indent=2))
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
