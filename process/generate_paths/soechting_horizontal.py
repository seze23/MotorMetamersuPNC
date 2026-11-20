"""Soechting & Poizner 2005 Eqs. 2–4, horizontal area-preserving paths.
Placement is explicitly a model assumption: the paper gives no torso placement.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd

def ar_from_alpha(alpha):
    return 1+alpha if alpha>=0 else 1/(1-alpha)
def alpha_from_ar(ar):
    return ar-1 if ar>=1 else 1-1/ar
def trajectory(alpha,condition,experiment,radius=15.,fs=240,buffer=.25):
    period=4. if experiment==1 else 2.5;cycles=3 if experiment==1 else 5
    c1,c2=(.0991,.0118) if experiment==1 else (.1558,.0321)
    if condition=='N':c1,c2=0.,0.
    elif condition=='FS':c1=-c1
    elif condition!='FT':raise ValueError(condition)
    ar=ar_from_alpha(alpha);A=radius*np.sqrt(ar);B=radius/np.sqrt(ar)
    # Repeat identical sampled cycles exactly; no resampling of spindle inputs.
    n=round(period*fs);u=2*np.pi*np.arange(n)/n
    theta=u+c1*np.sin(2*u)+c2*np.sin(4*u)
    omega=2*np.pi/period;tdot=omega*(1+2*c1*np.cos(2*u)+4*c2*np.cos(4*u))
    xy=np.column_stack([A*np.sin(theta),B*np.cos(theta)])
    speed=tdot*np.sqrt((A*np.cos(theta))**2+(B*np.sin(theta))**2)
    idx=np.arange(cycles*n+round(buffer*fs))%n
    return dict(times=np.arange(len(idx))/fs,paper_xy=xy[idx],phase=theta[idx],speed_cm_s=speed[idx],A=A,B=B,AR=ar,period=period,cycles=cycles,analysis_start_index=(cycles-1)*n,analysis_end_index=cycles*n)

def main():
    import opensim as osm
    from paths import EXPERIMENT_CONFIG,MODEL_PATH,PATHS_DIR,CENTEROUT_DIR
    from experiment import create_manifest,relative_artifact,validate_path_artifact
    from utils.opensim_pose import CoupledPose
    cfg=EXPERIMENT_CONFIG;pc=cfg['path'];opt=pc['horizontal_ellipses'];model=osm.Model(MODEL_PATH);state=model.initSystem();pose=CoupledPose(model,MODEL_PATH)
    pose.set_radians(state,{n:np.deg2rad(v) for n,v in {**pc['rest_pose_degrees'],**cfg['ik']['locked_wrist_degrees']}.items()});model.realizePosition(state)
    def marker(n):
        v=model.getMarkerSet().get(n).getLocationInGround(state);return np.array([v.get(j) for j in range(3)])
    R=np.array([[0,0,-1],[-1,0,0],[0,1,0]]);origin=marker('R.Shoulder');center=np.asarray(opt['center_world_cm'],dtype=float) if 'center_world_cm' in opt else R@(marker('Handle')-origin)*100;c7=R@(marker('C7')-origin)*100
    rows=[];entries=[]
    for exp in opt['experiments']:
      for alpha in opt['alphas']:
       for condition in opt['conditions']:
        p=trajectory(float(alpha),condition,int(exp),opt['radius_cm'],pc['sample_rate_hz'],opt['buffer_s']);ident=f'E{exp}_{condition}_{alpha:+.2f}'.replace('+','p').replace('-','m').replace('.','_')
        xy=p.pop('paper_xy');xyz=np.repeat(center[None],len(xy),axis=0);xyz[:,:2]-=xy
        dest=Path(PATHS_DIR)/f'{ident}.npz';np.savez_compressed(dest,**p,xyz=xyz,center_xyz=center,c7_xyz_world=c7,position_units='cm',coordinate_frame='shoulder_centered_world',sample_rate_hz=pc['sample_rate_hz'],paper_xy=xy,alpha=alpha,condition=condition,study_experiment=exp)
        validate_path_artifact(dest,pc);entries.append(dict(id=ident,desired_path=relative_artifact(dest),samples=len(xyz),sample_rate_hz=pc['sample_rate_hz']))
        rows.append(dict(id=ident,study_experiment=exp,condition=condition,alpha=alpha,AR=p['AR'],A_cm=p['A'],B_cm=p['B'],period_s=p['period'],cycles=p['cycles'],correct='circle' if alpha==0 else 'wide' if alpha>0 else 'narrow'))
    create_manifest(entries);pd.DataFrame(rows).to_csv(Path(CENTEROUT_DIR)/'stimuli.csv',index=False)
    print('Generated',len(rows),'horizontal paths. Center in pipeline frame (cm):',center,flush=True)
if __name__=='__main__':main()
