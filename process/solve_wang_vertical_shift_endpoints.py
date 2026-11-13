"""Solve the Wang endpoint grid after isolating a poster-height translation."""

from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from process.solve_fixed_target_posture_pilot import Solver,NAMES

SOURCE=ROOT/"outputs/wang_grid_full_scale_refined/ik"
FEATURES=ROOT/"dataexp/poster_bias_investigation/endpoint_factor_analysis/endpoint_feature_table.csv"
OUT=ROOT/"dataexp/poster_bias_investigation/wang_vertical_shift_full"
POSTER=np.array([45.,70.,25.,90.]);HIGH_UP_CM=4.0
PILOT=("r00_c00","r00_c10","r00_c19","r02_c00","r02_c10","r02_c19","r04_c00","r04_c10","r04_c19")


def solve_one(target):
    cache=OUT/"ik"/f"{target}.npz"
    if cache.exists():
        with np.load(cache) as z:
            cached=(float(z["error_cm"]),z["q"].copy(),z["physical"].copy(),z["desired"].copy())
        if cached[0]<.05:return target,*cached
    features=pd.read_csv(FEATURES).set_index("target");row=features.loc[target]
    desired=np.array([row.physical_right_cm,row.physical_forward_cm,HIGH_UP_CM])
    original=np.load(SOURCE/f"{target}.npz")["joint_angles"][515,:4]
    neighbor_seeds=[];row=int(target[1:3]);column=int(target[5:7])
    for neighbor_column in (column-1,column+1):
        neighbor=OUT/"ik"/f"r{row:02d}_c{neighbor_column:02d}.npz"
        if neighbor.exists():
            with np.load(neighbor) as saved:
                if float(saved["error_cm"])<.05:neighbor_seeds.append(saved["q"].copy())
    solver=Solver();first=solver.solve(desired,neighbor_seeds[0] if neighbor_seeds else POSTER,POSTER,200)
    if first[1]<.05:candidates=[first]
    else:
        seeds=neighbor_seeds[1:]+[original,(POSTER+original)/2]+[np.array([40,55,r,90.]) for r in (-60,-30,0,25,50,80,110)]
        candidates=[first]+[solver.solve(desired,seed,POSTER,150) for seed in seeds]
    feasible=[item for item in candidates if item[1]<.05];best=min(feasible or candidates,key=lambda item:(item[1]>=.05,item[2],item[1]))
    physical=solver.position(np.deg2rad(best[0]));temporary=cache.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,q=best[0],physical=physical,desired=desired,error_cm=best[1],target=target);os.replace(temporary,cache)
    return target,best[1],best[0],physical,desired


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--pilot",action="store_true");args=parser.parse_args()
    (OUT/"ik").mkdir(parents=True,exist_ok=True)
    targets=PILOT if args.pilot else tuple(f"r{r:02d}_c{c:02d}" for r in range(5) for c in range(20))
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(solve_one,target):target for target in targets}
        for future in as_completed(futures):
            item=future.result();results.append(item);print(item[0],"error",item[1],flush=True)
    rows=[]
    for target,error,q,physical,desired in sorted(results):
        rows.append({"target":target,"error_cm":error,"physical_right_cm":physical[0],"physical_forward_cm":physical[1],
            "physical_up_cm":physical[2],"desired_up_cm":desired[2],"pass_0_05_cm":error<.05,
            **{f"{name}_deg":q[i] for i,name in enumerate(NAMES)}})
    audit=pd.DataFrame(rows);name="pilot_reachability.csv" if args.pilot else "endpoint_ik_audit.csv";audit.to_csv(OUT/name,index=False)
    summary={"target_count":len(audit),"passing_targets":int(audit.pass_0_05_cm.sum()),"maximum_error_cm":float(audit.error_cm.max()),
        "height_cm":HIGH_UP_CM,"all_joint_limits_pass":bool(((audit[[f'{name}_deg' for name in NAMES]].to_numpy()>=np.array([-95,0,-90,0]))&(audit[[f'{name}_deg' for name in NAMES]].to_numpy()<=np.array([130,180,120,130]))).all())}
    (OUT/("pilot_reachability.json" if args.pilot else "endpoint_ik_summary.json")).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
