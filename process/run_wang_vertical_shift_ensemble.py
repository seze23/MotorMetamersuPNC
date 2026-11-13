"""Pilot/full static endpoint evaluation of the height-shifted Wang grid."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import process.run_poster_to_wang_posture_bridge as bridge
import process.run_wang_workspace_relocation_pilot as inference
import process.solve_wang_vertical_shift_endpoints as solve

OUT=solve.OUT;N=816
PILOT_MODELS=((0,9),(1,0),(2,1),(4,2))


def original_result(coefficient_seed,training_seed,targets):
    path=ROOT/f"dataexp/wang_grid_full_scale_corrected_tmin/model_{coefficient_seed}_{training_seed}_direct_xyz.csv"
    data=pd.read_csv(path);data=data[(data.window=="late_hold_250ms")&data.target.isin(targets)]
    return data.set_index("target")[["direct_outward_cm", "direct_forward_cm", "direct_up_cm"]]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--pilot",action="store_true");args=parser.parse_args()
    if args.pilot:targets=list(solve.PILOT)
    else:
        audit=pd.read_csv(OUT/"endpoint_ik_audit.csv");targets=audit.loc[audit.pass_0_05_cm,"target"].tolist()
    models=PILOT_MODELS if args.pilot else tuple((c,t) for c in range(5) for t in (0,1,2,9))
    q=[];physical=[]
    for target in targets:
        z=np.load(OUT/"ik"/f"{target}.npz");q.append(z["q"]);physical.append(z["physical"])
    q=np.asarray(q);physical=np.asarray(physical);trajectories=np.zeros((len(q),N,7),np.float32);trajectories[:,:,:4]=q[:,None,:]
    labels=bridge.labels_from_trajectories(trajectories)
    cache=OUT/("pilot_corrected_muscle_lengths.npy" if args.pilot else "corrected_muscle_lengths.npy")
    if cache.exists():lengths=np.load(cache)
    else:lengths=bridge.extract_lengths(trajectories);np.save(cache,lengths)
    rows=[];distribution=[];rate_cache={}
    for coefficient_seed,training_seed in models:
        if coefficient_seed not in rate_cache:rate_cache[coefficient_seed]=bridge.rates_from_lengths(lengths,coefficient_seed)
        rates=rate_cache[coefficient_seed];pad=(-len(labels))%25 if len(labels)>25 else 0
        if pad:
            rates_eval=np.concatenate([rates,np.repeat(rates[-1:],pad,axis=0)]);labels_eval=np.concatenate([labels,np.repeat(labels[-1:],pad,axis=0)])
        else:rates_eval,labels_eval=rates,labels
        name,prediction=inference.predict(rates_eval,labels_eval,coefficient_seed,training_seed);prediction=prediction[:len(labels)]
        baseline=original_result(coefficient_seed,training_seed,targets)
        config=yaml.safe_load((bridge.MODELS/f"spatiotemporal_4_8-8-32-64_7171_{coefficient_seed}_0/config.yaml").read_text())
        label_mean=np.asarray(config["label_mean"])[0,0,:3];label_std=np.asarray(config["label_std"])[0,0,:3]
        for i,target in enumerate(targets):
            true=labels[i,inference.LATE,:3].mean(0);pred=prediction[i,inference.LATE,:3].mean(0);raw=pred-true
            outward=float(-raw[0]);forward=float(-raw[1]);up=float(raw[2])
            base=baseline.loc[target]
            rows.append({"model":name,"target":target,
                "original_outward_cm":base.direct_outward_cm,"shifted_outward_cm":outward,"outward_shift_cm":outward-base.direct_outward_cm,
                "original_forward_cm":base.direct_forward_cm,"shifted_forward_cm":forward,"forward_shift_cm":forward-base.direct_forward_cm,
                "original_up_cm":base.direct_up_cm,"shifted_up_cm":up,"up_shift_cm":up-base.direct_up_cm,
                "physical_right_cm":physical[i,0],"physical_forward_cm":physical[i,1],"physical_up_cm":physical[i,2]})
            distribution.append({"model":name,"target":target,"max_abs_label_z":float(np.abs((true-label_mean)/label_std).max())})
        print("completed",name,flush=True)
    result=pd.DataFrame(rows);prefix="pilot_" if args.pilot else ""
    result.to_csv(OUT/f"{prefix}model_target_results.csv",index=False);pd.DataFrame(distribution).to_csv(OUT/f"{prefix}distribution_audit.csv",index=False)
    target=result.groupby("target",sort=False).agg(original_outward_cm=("original_outward_cm","mean"),shifted_outward_cm=("shifted_outward_cm","mean"),
        outward_shift_cm=("outward_shift_cm","mean"),models_shifted_outward=("outward_shift_cm",lambda v:int((v>0).sum())),
        original_forward_cm=("original_forward_cm","mean"),shifted_forward_cm=("shifted_forward_cm","mean"),forward_shift_cm=("forward_shift_cm","mean"),
        original_up_cm=("original_up_cm","mean"),shifted_up_cm=("shifted_up_cm","mean"),up_shift_cm=("up_shift_cm","mean")).reset_index()
    target.to_csv(OUT/f"{prefix}target_summary.csv",index=False)
    original_distribution=pd.read_csv(ROOT/"dataexp/poster_bias_investigation/wang_workspace_relocation_full/paired_target_summary.csv").set_index("target").original_max_abs_label_z
    shifted_distribution=pd.DataFrame(distribution).groupby("target").max_abs_label_z.max()
    target["original_max_abs_label_z"]=target.target.map(original_distribution);target["shifted_max_abs_label_z"]=target.target.map(shifted_distribution)
    target["shared_in_distribution_abs_z_le_3"]=target[["original_max_abs_label_z","shifted_max_abs_label_z"]].max(axis=1)<=3
    target.to_csv(OUT/f"{prefix}target_summary.csv",index=False)
    shared=target[target.shared_in_distribution_abs_z_le_3]
    summary={"targets":len(target),"models":len(models),"original_mean_outward_cm":float(result.original_outward_cm.mean()),
        "shifted_mean_outward_cm":float(result.shifted_outward_cm.mean()),"mean_outward_shift_cm":float(result.outward_shift_cm.mean()),
        "targets_shifted_outward":int((target.outward_shift_cm>0).sum()),"target_means_outward_original":int((target.original_outward_cm>0).sum()),
        "target_means_outward_shifted":int((target.shifted_outward_cm>0).sum()),"max_abs_label_z":float(pd.DataFrame(distribution).max_abs_label_z.max()),
        "tmin_min_mm":float(lengths[:,:,14].min()),"tmin_max_mm":float(lengths[:,:,14].max())}
    summary.update({"shared_in_distribution_targets":len(shared),"shared_original_mean_outward_cm":float(shared.original_outward_cm.mean()),
        "shared_shifted_mean_outward_cm":float(shared.shifted_outward_cm.mean()),"shared_mean_outward_shift_cm":float(shared.outward_shift_cm.mean()),
        "shared_targets_shifted_outward":int((shared.outward_shift_cm>0).sum()),"excluded_unreachable_targets":["r00_c15","r00_c16"] if not args.pilot else []})
    (OUT/f"{prefix}result_summary.json").write_text(json.dumps(summary,indent=2));print(target.to_string(index=False));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
