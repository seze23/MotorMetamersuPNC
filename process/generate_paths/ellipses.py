"""Generate Viviani-style frontoparallel ellipses for the shared Pipeline."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import ellipe
from scipy.integrate import cumulative_trapezoid
from experiment import create_manifest,relative_artifact,validate_path_artifact
from paths import EXPERIMENT_CONFIG,PATHS_DIR,CENTEROUT_DIR

def ellipse_path(e,condition,perimeter_cm=160.,period_s=2.5,fs=240,cycles=3,kinematic_e=.9,direction='clockwise'):
    if direction not in {'clockwise','counterclockwise'}:
        raise ValueError('Unsupported movement direction')
    vertical_sign=1 if direction=='clockwise' else -1
    def axes(e):
        a=perimeter_cm/(4*ellipe(e**2));b=a*np.sqrt(1-e**2)
        return (a,b) if e>=0 else (b,a)
    ax,az=axes(e);phi=np.linspace(0,2*np.pi,30001)
    x,z=-ax*np.cos(phi),vertical_sign*az*np.sin(phi)
    ds=np.sqrt((ax*np.sin(phi))**2+(az*np.cos(phi))**2)
    if condition=='A':speed=np.ones_like(phi)
    elif condition in ['B','C']:
        a,b=axes(kinematic_e);theta=0 if condition=='B' else np.pi/2
        u,v=x*np.cos(theta)+z*np.sin(theta),-x*np.sin(theta)+z*np.cos(theta)
        psi=np.arctan2(u/a,v/b)
        speed=(2*np.pi/period_s)*np.sqrt((a*np.cos(psi))**2+(b*np.sin(psi))**2)
    else:raise ValueError('Condition must be A, B or C')
    clock=cumulative_trapezoid(ds/speed,phi,initial=0);clock*=period_s/clock[-1]
    times=np.arange(round(cycles*period_s*fs))/fs
    phase=np.interp(times%period_s,clock,phi)
    return times,np.column_stack((-ax*np.cos(phase),vertical_sign*az*np.sin(phase)))

def study_plane_center(path_config,ellipse_config):
    """Shoulder-height plane 45 cm anterior to the model's thorax reference.

    Ground X is anterior, Y superior and Z toward the right arm. The pipeline
    transform maps these to (-Z,-X,Y), so participant-right is pipeline -X.
    The model has no chest-surface marker: torso-reference X=0 is explicit.
    """
    import opensim as osm
    from paths import MODEL_PATH
    model=osm.Model(MODEL_PATH);state=model.initSystem();coordinates=model.getCoordinateSet()
    pose=dict(path_config['rest_pose_degrees'])
    pose.update(EXPERIMENT_CONFIG['ik']['locked_wrist_degrees'])
    for name,degrees in pose.items():coordinates.get(name).setValue(state,float(np.deg2rad(degrees)))
    model.realizePosition(state)
    point=model.getMarkerSet().get('R.Shoulder').getLocationInGround(state)
    shoulder=np.array([point.get(i) for i in range(3)])
    center=shoulder.copy()
    center[0]=float(ellipse_config.get('chest_reference_ground_x_m',0))+float(ellipse_config.get('plane_distance_cm',45))/100
    rotation=np.array([[0,0,-1],[-1,0,0],[0,1,0]])
    return rotation@(center-shoulder)*100

def main():
    cfg=EXPERIMENT_CONFIG['path'];e_cfg=cfg['ellipse']
    if e_cfg.get('center_source') == 'study_plane':
        center=study_plane_center(cfg,e_cfg)
    elif e_cfg.get('center_source') == 'rest_pose':
        from process.generate_paths.generatereachpath import model_rest_center
        center=model_rest_center(cfg['rest_pose_degrees'])
    else:
        center=np.asarray(e_cfg.get('center_world_cm',[0,45,0]),dtype=float)
    rows=[];entries=[]
    for e in e_cfg['eccentricities']:
        e=float(e)
        for condition in 'ABC':
            direction=e_cfg.get('direction','clockwise')
            times,xy=ellipse_path(e,condition,e_cfg['perimeter_cm'],e_cfg['period_s'],cfg['sample_rate_hz'],e_cfg['cycles'],e_cfg['kinematic_e'],direction)
            if e_cfg.get('horizontal_axis')=='participant_right':
                xy[:,0]*=-1
            ident=f'{condition}_{e:+.2f}'.replace('+','p').replace('-','m').replace('.','_')
            xyz=np.repeat(center[None,:],len(times),axis=0);xyz[:,[0,2]]+=xy
            dest=Path(PATHS_DIR)/f'{ident}.npz'
            np.savez(dest,xyz=xyz,times=times,center_xyz=center,position_units='cm',coordinate_frame='shoulder_centered_world',sample_rate_hz=cfg['sample_rate_hz'],direction=direction,horizontal_axis=e_cfg.get('horizontal_axis','pipeline_x'))
            validate_path_artifact(dest,cfg)
            entries.append(dict(id=ident,desired_path=relative_artifact(dest),samples=len(times),sample_rate_hz=cfg['sample_rate_hz']))
            rows.append(dict(id=ident,condition=condition,e=e,correct='true circle' if e==0 else 'horizontal ellipse' if e>0 else 'vertical ellipse'))
    create_manifest(entries)
    pd.DataFrame(rows).to_csv(Path(CENTEROUT_DIR)/'stimuli.csv',index=False)
    print(f'Generated {len(rows)} ellipse/velocity stimuli.')

if __name__=='__main__':main()
