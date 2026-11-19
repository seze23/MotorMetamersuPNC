"""Read CNN joint predictions in OpenSim's physical frontoparallel plane."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import opensim as osm
import yaml
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.opensim_pose import CoupledPose
ROOT=Path(__file__).resolve().parents[1]
R=np.array([[0,0,-1],[-1,0,0],[0,1,0]])
NAMES=['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion']
CLASSES=['true circle','horizontal ellipse','vertical ellipse']

def shape(xy,tolerance):
    extent=np.ptp(xy,axis=0)
    if not np.isfinite(extent).all() or min(extent)<1e-5:return 'invalid',np.nan,np.nan
    ratio=extent[0]/extent[1]
    e=np.sign(ratio-1)*np.sqrt(1-min(ratio,1/ratio)**2)
    label=CLASSES[0] if 1/tolerance<=ratio<=tolerance else CLASSES[1 if ratio>1 else 2]
    return label,ratio,e

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tolerance',type=float,default=1.05)
    args=parser.parse_args();out=args.output.resolve()
    cfg=yaml.safe_load((out/'experiment.yaml').read_text());inventory=pd.read_csv(out/'batch/inventory.csv')
    stimuli=pd.read_csv(out/'stimuli.csv').set_index('id')
    model=osm.Model(str(ROOT/'MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim'))
    state=model.initSystem();cs=model.getCoordinateSet();mk=model.getMarkerSet()
    def marker(name):
        v=mk.get(name).getLocationInGround(state);return np.array([v.get(i) for i in range(3)])
    pose=CoupledPose(model,ROOT/'MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim')
    pose.set_radians(state,{name:np.deg2rad(value) for name,value in {**cfg['path']['rest_pose_degrees'],**cfg['ik']['locked_wrist_degrees']}.items()})
    model.realizePosition(state);origin=marker('R.Shoulder');free=[cs.get(name) for name in NAMES]
    lower=np.array([c.getRangeMin() for c in free]);upper=np.array([c.getRangeMax() for c in free])
    rows=[];quality=[]
    for ident in stimuli.index:
        muscle=np.load(out/'muscles'/f'{ident}.npz');desired=np.load(out/'paths'/f'{ident}.npz')
        actual=muscle['wrist_xyz_reference_world'];error=np.linalg.norm(actual-desired['xyz'],axis=1)
        label,ratio,e=shape(actual[-600:,[0,2]],args.tolerance)
        quality.append(dict(id=ident,max_ik_error_cm=error.max(),mean_ik_error_cm=error.mean(),delivered_class=label,delivered_e=e))
    for item in inventory.itertuples(index=False):
        raw=np.load(out/item.predictions);pred=raw['predictions'];times=raw['times'];mask=times>=times[-1]-2.5+1/240
        indices=np.flatnonzero(mask)[::4];angles=pred[indices,3:7];xyz=[]
        for q in angles:
            pose.set_radians(state,dict(zip(NAMES,np.deg2rad(q))))
            model.realizePosition(state);xyz.append(R@(marker('Handle')-origin)*100)
        xyz=np.asarray(xyz);label,ratio,e=shape(xyz[:,[0,2]],args.tolerance)
        invalid_bounds=np.any((np.deg2rad(angles)<lower)|(np.deg2rad(angles)>upper),axis=1).mean()
        dest=(out/item.predictions).with_name(item.id+'_physical.npz')
        np.savez_compressed(dest,xyz=xyz,times=times[indices])
        rows.append(dict(id=item.id,condition=item.condition,e=item.e,coefficient_seed=item.coefficient_seed,
                         training_seed=item.training_seed,correct=stimuli.loc[item.id,'correct'],predicted=label,
                         ratio=ratio,predicted_e=e,joint_out_of_bounds_fraction=invalid_bounds,
                         physical_prediction=str(dest.relative_to(out))))
        if len(rows)%21==0:print(f'Read out {len(rows)} predictions',flush=True)
    pd.DataFrame(rows).to_csv(out/'batch/shape_predictions.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'batch/quality.csv',index=False)
    (out/'batch/readout.json').write_text(json.dumps(dict(tolerance=args.tolerance,source_signature=json.loads((out/'generation.json').read_text())['signature']),indent=2))

if __name__=='__main__':main()
