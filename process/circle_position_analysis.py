"""Validate workspace circles and read one configured CNN's circle predictions."""
from pathlib import Path
import argparse,json,sys,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import yaml
import opensim as osm
from utils.opensim_pose import CoupledPose
from utils.trajectory_shape import extent_shape,fit_ellipse
ROOT=Path(__file__).resolve().parents[1]
R=np.array([[0,0,-1],[-1,0,0],[0,1,0]])
NAMES=['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--quality-only',action='store_true');args=parser.parse_args()
    out=args.output.resolve();cfg=yaml.safe_load((out/'experiment.yaml').read_text());positions=pd.read_csv(out/'positions.csv').set_index('id');manifest=yaml.safe_load((out/'manifest.yaml').read_text())
    optimal=np.asarray(yaml.safe_load((ROOT/'extract_data/configs/train_test_data_spindles_extended.yaml').read_text())['optimal_lengths'])
    quality=[];starts=[];items={r['id']:r for r in manifest['trajectories']}
    for ident in positions.index:
        item=items[ident];p=np.load(out/item['desired_path']);m=np.load(out/item['muscle_data']);s=np.load(out/item['spindle_data']);lo,hi=int(p['analysis_start_index']),int(p['analysis_end_index']);actual=m['wrist_xyz_reference_world'][lo:hi];target=p['xyz'][lo:hi];xy=actual[:,[0,2]]*[-1,1]
        label,ratio=extent_shape(xy);ellipse=fit_ellipse(xy)
        lengths=m['fiber_lengths'];velocity=np.abs(np.gradient(lengths,1/float(cfg['path']['sample_rate_hz']),axis=0)).max();error=np.linalg.norm(actual-target,axis=1).max()
        finite=bool(np.isfinite(lengths).all() and np.isfinite(s['firing_rates']).all())
        passed=bool(finite and (lengths/optimal).min()>.1 and (lengths/optimal).max()<3 and velocity<500 and error<.2 and label=='circle' and ellipse['axis_ratio']<=1.05)
        starts.append(m['wrist_xyz_reference_world'][0])
        quality.append(dict(id=ident,passed=passed,max_circle_error_cm=error,actual_extent_ratio=ratio,actual_fit_axis_ratio=ellipse['axis_ratio'],max_fiber_mm=lengths.max(),min_fiber_mm=lengths.min(),peak_fiber_velocity_mm_s=velocity))
    quality=pd.DataFrame(quality);quality.to_csv(out/'circle_position_quality.csv',index=False)
    start_spread=float(np.linalg.norm(np.asarray(starts)-starts[0],axis=1).max())
    (out/'common_start_check.json').write_text(json.dumps(dict(max_achieved_start_difference_cm=start_spread,requested_common_start_right_forward_cm=cfg['path']['circle_positions']['common_start_right_forward_cm']),indent=2))
    if args.quality_only:
        if not quality.passed.all() or start_spread>.05:raise ValueError('Circle validation failed; inspect circle_position_quality.csv and common_start_check.json.')
        print(f'All {len(quality)} circles valid; shared achieved start spread {start_spread:.6f} cm.');return
    model_path=ROOT/'MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim';model=osm.Model(str(model_path));state=model.initSystem();pose=CoupledPose(model,model_path);cs=model.getCoordinateSet();mk=model.getMarkerSet()
    pose.set_radians(state,{n:np.deg2rad(v) for n,v in {**cfg['path']['rest_pose_degrees'],**cfg['ik']['locked_wrist_degrees']}.items()});model.realizePosition(state)
    def marker(name):
        v=mk.get(name).getLocationInGround(state);return np.array([v.get(i) for i in range(3)])
    origin=marker('R.Shoulder');lower=np.array([cs.get(n).getRangeMin() for n in NAMES]);upper=np.array([cs.get(n).getRangeMax() for n in NAMES]);rows=[]
    for ident in positions.index:
        info=positions.loc[ident];item=items[ident];q=quality[quality.id==ident].iloc[0];p=np.load(out/item['desired_path']);m=np.load(out/item['muscle_data']);lo,hi=int(p['analysis_start_index']),int(p['analysis_end_index']);indices=np.arange(lo,hi,4)
        if not q.passed or 'raw_predictions' not in item:
            rows.append(dict(id=ident,predicted='invalid input',**info.to_dict()));continue
        raw=np.load(out/item['raw_predictions']);pred=raw['predictions'];xyz=[]
        assert len(pred)==len(p['times']) and np.isfinite(pred).all()
        angles=pred[indices,3:7]
        for a in angles:
            pose.set_radians(state,dict(zip(NAMES,np.deg2rad(a))));model.realizePosition(state);xyz.append(R@(marker('Handle')-origin)*100)
        xyz=np.asarray(xyz);actual=m['wrist_xyz_reference_world'][indices];xy=xyz[:,[0,2]]*[-1,1];label,ratio=extent_shape(xy);ellipse=fit_ellipse(xy);true_center=actual.mean(axis=0);decoded_center=xyz.mean(axis=0);c7=p['c7_xyz_world']
        center_error=np.linalg.norm(decoded_center-true_center);delta=xyz-actual;point_error=np.linalg.norm(delta,axis=1)
        dest=out/'predictions'/f'{ident}_circle_readout.npz';np.savez_compressed(dest,decoded_xyz=xyz,actual_xyz=actual,times=p['times'][indices])
        rows.append(dict(id=ident,**info.to_dict(),predicted=label,extent_ratio=ratio,actual_extent_ratio=q.actual_extent_ratio,actual_fitted_axis_ratio=q.actual_fit_axis_ratio,fitted_axis_ratio=ellipse['axis_ratio'],fitted_angle_deg=ellipse['angle_deg'],ellipse_fit_rmse_cm=ellipse['fit_rmse_cm'],decoded_major_radius_cm=ellipse['major_radius_cm'],decoded_minor_radius_cm=ellipse['minor_radius_cm'],actual_right_cm=-(true_center[0]-c7[0]),actual_forward_cm=-(true_center[1]-c7[1]),decoded_right_cm=-(decoded_center[0]-c7[0]),decoded_forward_cm=-(decoded_center[1]-c7[1]),centroid_error_cm=center_error,rmse_3d_cm=np.sqrt(np.mean(np.sum(delta**2,axis=1))),mean_point_error_cm=point_error.mean(),joint_out_of_bounds_fraction=np.any((np.deg2rad(angles)<lower)|(np.deg2rad(angles)>upper),axis=1).mean(),readout=str(dest.relative_to(out))))
    pd.DataFrame(rows).to_csv(out/'circle_position_predictions.csv',index=False)
    checkpoint=ROOT/cfg['proprioception']['model_path'];(out/'single_model_provenance.json').write_text(json.dumps(dict(n_models=1,checkpoint_path=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256((checkpoint/'model.ckpt').read_bytes()).hexdigest(),config_signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),grid_csv_sha256=hashlib.sha256((ROOT/cfg['path']['circle_positions']['grid_csv']).read_bytes()).hexdigest(),source_grid='Approximate uniform grid from supplied screenshot; no original numeric grid available'),indent=2))
    print(f'Read out {len(rows)} circle predictions using one checkpoint.')

if __name__=='__main__':main()
