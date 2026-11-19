"""OpenSim marker IK with bounded, warm-started SciPy least squares.

Uses the same model, joint bounds, brace and desired-path manifest as the
standard OpenSim IK stage. Repeated spatial targets reuse exact solutions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import opensim as osm
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from experiment import load_manifest, resolve_artifact, set_artifact, validate_path_artifact
from paths import EXPERIMENT_CONFIG, MODEL_PATH, IK_DIR
from utils.opensim_pose import CoupledPose

NAMES = ['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion']
DISTAL = ['pro_sup','deviation','flexion']
ROTATION = np.array([[0,0,-1],[-1,0,0],[0,1,0]])

def marker(markers, state, name):
    v=markers.get(name).getLocationInGround(state)
    return np.array([v.get(i) for i in range(3)])

def main():
    cfg=EXPERIMENT_CONFIG
    if not cfg.get('ik',{}).get('lock_wrist',True):
        raise ValueError('The bounded four-coordinate IK backend requires lock_wrist=true.')
    model=osm.Model(MODEL_PATH); state=model.initSystem()
    coords=model.getCoordinateSet(); markers=model.getMarkerSet()
    rest=cfg['path']['rest_pose_degrees']
    brace=cfg['ik'].get('locked_wrist_degrees',{})
    pose=CoupledPose(model,MODEL_PATH)
    pose.set_radians(state,{name:np.deg2rad(value) for name,value in {**rest,**brace}.items()})
    model.realizePosition(state)
    origin=marker(markers,state,'R.Shoulder')
    free=[coords.get(n) for n in NAMES]
    lower=np.array([np.rad2deg(c.getRangeMin()) for c in free])+1e-5
    upper=np.array([np.rad2deg(c.getRangeMax()) for c in free])-1e-5
    initial=np.array([rest[n] for n in NAMES])
    tolerance=float(cfg.get('ik',{}).get('max_marker_error_cm',.1))
    cache={}
    for item in load_manifest()['trajectories']:
        xyz,times=validate_path_artifact(resolve_artifact(item['desired_path']))
        angles=[]; actual=[]; previous=initial.copy()
        stride=int(cfg.get('ik',{}).get('sample_stride',1))
        indices=np.unique(np.r_[np.arange(0,len(xyz),stride),len(xyz)-1])
        for frame_index in indices:
            target=xyz[frame_index]
            key=tuple(np.round(target,5))
            if key not in cache:
                def forward(q):
                    pose.set_radians(state,dict(zip(NAMES,np.deg2rad(q))))
                    model.realizePosition(state)
                    return ROTATION@(marker(markers,state,'Handle')-origin)*100
                def residual(q):
                    return np.r_[forward(q)-target, 1e-5*(q-previous)]
                candidates=[]
                seeds=[previous,initial,np.array([-60,100,60,60]),np.array([0,90,90,60]),np.array([90,70,20,30])]
                for seed in seeds:
                    fit=least_squares(residual,np.clip(seed,lower,upper),bounds=(lower,upper),
                                      diff_step=1e-4,max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
                    hand=forward(fit.x)
                    candidates.append((np.linalg.norm(hand-target),fit.x.copy(),hand.copy()))
                    if candidates[-1][0] < tolerance/10: break
                error,q,hand=min(candidates,key=lambda r:r[0])
                cache[key]=(q,hand,error)
            q,hand,error=cache[key]; previous=q.copy()
            angles.append([*q,*[brace.get(n,0.) for n in DISTAL]]);actual.append(hand)
        angles=CubicSpline(times[indices],np.array(angles),axis=0)(times)
        actual=[]
        for q in angles[:,:4]:
            pose.set_radians(state,dict(zip(NAMES,np.deg2rad(q))))
            model.realizePosition(state)
            actual.append(ROTATION@(marker(markers,state,'Handle')-origin)*100)
        actual=np.array(actual);errors=np.linalg.norm(actual-xyz,axis=1)
        output=Path(IK_DIR)/f"{item['id']}.npz"
        np.savez(output,joint_angles=np.array(angles,dtype=np.float32),times=times,
                 trajectory_id=item['id'],reference_origin_ground_m=origin,
                 wrist_xyz_reference_world=actual,marker_error_cm=errors)
        set_artifact(item['id'],'ik_solution',output)
        print(f"{item['id']}: max marker error {errors.max():.5f} cm",flush=True)
        if errors.max()>tolerance:
            raise RuntimeError(f"{item['id']} exceeds marker tolerance; inspect IK before continuing.")

if __name__=='__main__':main()
