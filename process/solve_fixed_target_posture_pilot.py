"""Framewise fixed-Cartesian Wang pilot on Wang-like and poster-like IK branches."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import sys

import numpy as np
import opensim as osm
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "process"))
from body_coordinates import opensim_to_body_cm

SOURCE = ROOT / "outputs/wang_grid_full_scale_refined"
OUT = ROOT / "dataexp/poster_bias_investigation/fixed_target_posture_pilot"
MODEL = ROOT / "MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
NAMES = ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion")
POSTER_REFERENCE = np.array([45.0, 70.0, 25.0, 90.0])
TARGETS = ("r00_c10", "r00_c19", "r01_c05", "r02_c09", "r02_c10", "r03_c15", "r04_c10", "r04_c19")
POSTURE_WEIGHT_CM_PER_RAD = 0.01


def vec3(value): return np.array([value.get(i) for i in range(3)])


class Solver:
    def __init__(self):
        self.model = osm.Model(str(MODEL)); self.state = self.model.initSystem()
        self.coordinates = [self.model.getCoordinateSet().get(name) for name in NAMES]
        self.handle = self.model.getMarkerSet().get("Handle"); self.c7 = self.model.getMarkerSet().get("C7")
        self.lower = np.array([item.getRangeMin() for item in self.coordinates])+1e-8
        self.upper = np.array([item.getRangeMax() for item in self.coordinates])-1e-8

    def position(self, q_rad):
        for coordinate, value in zip(self.coordinates, q_rad): coordinate.setValue(self.state,float(value))
        self.model.realizePosition(self.state)
        return opensim_to_body_cm(vec3(self.handle.getLocationInGround(self.state)),vec3(self.c7.getLocationInGround(self.state)))

    def solve(self,target,initial_deg,reference_deg,max_nfev=20):
        initial=np.clip(np.deg2rad(initial_deg),self.lower,self.upper); reference=np.deg2rad(reference_deg)
        def residual(q): return np.r_[self.position(q)-target,POSTURE_WEIGHT_CM_PER_RAD*(q-reference)]
        result=least_squares(residual,initial,bounds=(self.lower,self.upper),xtol=1e-7,ftol=1e-7,gtol=1e-7,max_nfev=max_nfev)
        q=np.rad2deg(result.x); error=float(np.linalg.norm(self.position(result.x)-target)); distance=float(np.linalg.norm(q-reference_deg))
        return q,error,distance,result.nfev


def solve_branch(solver, desired, existing, reference_kind):
    unique_start=395; unique_end=516
    reference0=existing[0] if reference_kind=="wang_like" else POSTER_REFERENCE
    seeds=[existing[0],POSTER_REFERENCE]
    seeds += [np.array([45,70,rotation,90],float) for rotation in (-60,-30,0,25,45,60,90)]
    candidates=[solver.solve(desired[0],seed,reference0,80) for seed in seeds]
    feasible=[item for item in candidates if item[1]<.05]
    first=min(feasible or candidates,key=lambda item:(item[1]>=.05,item[2],item[1]))
    center,target=desired[0],desired[unique_end-1];delta=target-center;denominator=float(delta@delta)
    progress=np.clip(((desired-center)@delta)/denominator,0,1)
    final=None
    for node_count in (9,17,33):
        node_progress=np.linspace(0,1,node_count);node_q=[first[0]];errors=[first[1]];nfevs=[first[3]]
        for value in node_progress[1:]:
            point=center+value*delta
            reference=(existing[0]+value*(existing[unique_end-1]-existing[0])) if reference_kind=="wang_like" else POSTER_REFERENCE
            solved=solver.solve(point,node_q[-1],reference,30)
            node_q.append(solved[0]);errors.append(solved[1]);nfevs.append(solved[3])
        interpolation=PchipInterpolator(node_progress,np.asarray(node_q),axis=0)
        q=interpolation(progress).astype(np.float32)
        moving=np.arange(unique_start,unique_end)
        achieved_moving=np.stack([solver.position(np.deg2rad(q[index])) for index in moving])
        moving_error=np.linalg.norm(achieved_moving-desired[moving],axis=1)
        final=(q,achieved_moving,np.asarray(errors),np.asarray(nfevs),node_count,float(moving_error.max()))
        if moving_error.max()<.05: break
    q,achieved_moving,errors,nfevs,node_count,_=final
    achieved=np.empty_like(desired);achieved[:unique_start]=achieved_moving[0];achieved[unique_start:unique_end]=achieved_moving;achieved[unique_end:]=achieved_moving[-1]
    return q,achieved,errors,nfevs,candidates,node_count


def solve_target(target):
    solver=Solver(); path=np.load(SOURCE/"paths"/f"{target}.npz"); desired=path["xyz_body"].astype(float)
    existing=np.load(SOURCE/"ik"/f"{target}.npz")["joint_angles"][:,:4].astype(float)
    result={}
    for condition in ("wang_like","poster_like"):
        q,achieved,errors,nfevs,candidates,node_count=solve_branch(solver,desired,existing,condition)
        result[condition]={"q":q,"achieved":achieved,"errors":errors,"nfevs":nfevs,"candidates":candidates,"node_count":node_count}
    return target,desired,path["times"],result


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"ik").mkdir(exist_ok=True)
    solved={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(solve_target,target):target for target in TARGETS}
        for future in as_completed(futures):
            target,desired,times,result=future.result();solved[target]=(desired,times,result);print("solved",target,flush=True)
    rows=[];center_rows=[]
    for target in TARGETS:
        desired,times,result=solved[target]
        paired=np.linalg.norm(result["wang_like"]["achieved"]-result["poster_like"]["achieved"],axis=1)
        for condition in ("wang_like","poster_like"):
            item=result[condition];joints=np.zeros((len(times),7),np.float32);joints[:,:4]=item["q"]
            np.savez_compressed(OUT/"ik"/f"{target}_{condition}.npz",joint_angles=joints,times=times,desired_xyz_body=desired,achieved_xyz_body=item["achieved"],condition=condition,target=target)
            endpoint=item["q"][-1];start=item["q"][0]
            rows.append({"target":target,"condition":condition,"adaptive_node_count":item["node_count"],"max_cartesian_error_cm":float(np.linalg.norm(item["achieved"]-desired,axis=1).max()),"mean_cartesian_error_cm":float(np.linalg.norm(item["achieved"]-desired,axis=1).mean()),"max_paired_path_difference_cm":float(paired.max()),"max_joint_step_deg":float(np.abs(np.diff(item["q"],axis=0)).max()),**{f"start_{name}_deg":start[i] for i,name in enumerate(NAMES)},**{f"endpoint_{name}_deg":endpoint[i] for i,name in enumerate(NAMES)}})
            for candidate in item["candidates"]: center_rows.append({"target":target,"condition":condition,"error_cm":candidate[1],"reference_distance_deg":candidate[2],"nfev":candidate[3],**{f"{name}_deg":candidate[0][i] for i,name in enumerate(NAMES)}})
    audit=pd.DataFrame(rows);audit.to_csv(OUT/"trajectory_audit.csv",index=False);pd.DataFrame(center_rows).to_csv(OUT/"center_multistart_candidates.csv",index=False)
    pivot=audit.pivot(index="target",columns="condition",values=[f"start_{name}_deg" for name in NAMES]+[f"endpoint_{name}_deg" for name in NAMES])
    summary={"targets":len(TARGETS),"maximum_cartesian_error_cm":float(audit.max_cartesian_error_cm.max()),"maximum_paired_path_difference_cm":float(audit.max_paired_path_difference_cm.max()),"maximum_joint_step_deg":float(audit.max_joint_step_deg.max()),"all_paths_pass_0_05_cm":bool((audit.max_cartesian_error_cm<.05).all())}
    (OUT/"ik_summary.json").write_text(json.dumps(summary,indent=2));print(audit.to_string(index=False));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
