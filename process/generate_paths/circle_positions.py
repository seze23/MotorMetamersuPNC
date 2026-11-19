"""Vertical circles at C7-referenced workspace positions, from a common start."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import opensim as osm
from paths import EXPERIMENT_CONFIG,MODEL_PATH,PATHS_DIR,CENTEROUT_DIR
from experiment import create_manifest,relative_artifact,validate_path_artifact
R=np.array([[0,0,-1],[-1,0,0],[0,1,0]])

def make_trial(common,center,radius,fs=240,period=2.5,cycles=3,hold=.5,reach=1.,settle=.5,buffer=.25):
    """Circles start on participant-left and move counterclockwise in X-right/Z-up.

    World pipeline X is negative participant-right. A continuing-motion buffer
    protects the last analyzed cycle from end-of-sequence filtering effects.
    """
    common=np.asarray(common,dtype=float);center=np.asarray(center,dtype=float)
    n_hold=round(hold*fs);n_reach=round(reach*fs);n_settle=round(settle*fs)
    n_cycle=round(period*fs);n_circle=cycles*n_cycle+round(buffer*fs)
    left=center+np.array([radius,0.,0.])
    phase=np.linspace(0,1,n_reach);progress=10*phase**3-15*phase**4+6*phase**5
    approach=common+(left-common)*progress[:,None]
    angles=2*np.pi*np.arange(n_circle)/(period*fs)
    circle=np.repeat(center[None,:],n_circle,axis=0)
    circle[:,0]+=radius*np.cos(angles);circle[:,2]-=radius*np.sin(angles)
    xyz=np.vstack([np.repeat(common[None,:],n_hold,axis=0),approach,np.repeat(left[None,:],n_settle,axis=0),circle])
    first=n_hold+n_reach+n_settle
    return dict(xyz=xyz,times=np.arange(len(xyz))/fs,
                circle_start_index=first,circle_end_index=first+cycles*n_cycle,
                analysis_start_index=first+(cycles-1)*n_cycle,analysis_end_index=first+cycles*n_cycle,
                center_xyz=center,common_start_xyz=common,circle_left_xyz=left)

def main():
    path=EXPERIMENT_CONFIG['path'];cfg=path['circle_positions']
    root=Path(__file__).resolve().parents[2]
    grid=pd.read_csv(root/cfg['grid_csv'])
    if cfg.get('selected_ids'):grid=grid[grid.id.isin(cfg['selected_ids'])]
    model=osm.Model(MODEL_PATH);state=model.initSystem();cs=model.getCoordinateSet()
    pose=dict(path['rest_pose_degrees']);pose.update(EXPERIMENT_CONFIG['ik']['locked_wrist_degrees'])
    for name,degrees in pose.items():cs.get(name).setValue(state,float(np.deg2rad(degrees)))
    model.realizePosition(state);markers=model.getMarkerSet()
    def marker(name):
        v=markers.get(name).getLocationInGround(state);return np.array([v.get(i) for i in range(3)])
    shoulder=marker('R.Shoulder');c7=marker('C7')
    def position(right,forward):
        ground=c7+np.array([forward,0.,right])/100
        ground[1]=shoulder[1]+float(cfg.get('height_from_shoulder_cm',0))/100
        return R@(ground-shoulder)*100
    common=position(*cfg['common_start_right_forward_cm'])
    radius=float(cfg['radius_cm']);entries=[];rows=[]
    for row in grid.itertuples(index=False):
        center=position(row.right_cm,row.forward_cm)
        trial=make_trial(common,center,radius,path['sample_rate_hz'],cfg['period_s'],cfg['cycles'],cfg['hold_start_s'],cfg['reach_s'],cfg['settle_s'],cfg['buffer_s'])
        dest=Path(PATHS_DIR)/f'{row.id}.npz'
        np.savez(dest,**trial,position_units='cm',coordinate_frame='shoulder_centered_world',sample_rate_hz=path['sample_rate_hz'],radius_cm=radius,right_from_c7_cm=row.right_cm,forward_from_c7_cm=row.forward_cm,c7_xyz_world=R@(c7-shoulder)*100)
        validate_path_artifact(dest,path)
        entries.append(dict(id=row.id,desired_path=relative_artifact(dest),samples=len(trial['times']),sample_rate_hz=path['sample_rate_hz']))
        rows.append(dict(id=row.id,row=row.row,column=row.column,right_cm=row.right_cm,forward_cm=row.forward_cm,radius_cm=radius,correct='circle'))
    create_manifest(entries);pd.DataFrame(rows).to_csv(Path(CENTEROUT_DIR)/'positions.csv',index=False)
    print(f'Generated {len(rows)} constant-speed vertical circles with one common hand start.',flush=True)

if __name__=='__main__':main()
