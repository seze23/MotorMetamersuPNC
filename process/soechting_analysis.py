"""Quality, matched 20-checkpoint inference and physical readout for horizontal ellipses."""
from pathlib import Path
import sys,json,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/soechting_2005';BASE=ROOT/'trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints';NAMES=['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion'];R=np.array([[0,0,-1],[-1,0,0],[0,1,0]])
def config():return yaml.safe_load((ROOT/'experiments/soechting_2005.yaml').read_text())
def entries():return yaml.safe_load((OUT/'manifest.yaml').read_text())['trajectories']
def quality():
 import xml.etree.ElementTree as ET
 c=config();xml=ET.parse(ROOT/'MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim');coord_names=[*NAMES,'pro_sup','deviation','flexion'];ranges={e.attrib['name']:np.rad2deg(np.fromstring(e.findtext('range'),sep=' ')) for e in xml.findall('.//Coordinate') if e.find('range') is not None};lower=np.array([ranges[n][0] for n in coord_names]);upper=np.array([ranges[n][1] for n in coord_names]);opt=np.array(yaml.safe_load((ROOT/'extract_data/configs/train_test_data_spindles_extended.yaml').read_text())['optimal_lengths']);rows=[]
 for e in entries():
  p=np.load(OUT/e['desired_path']);m=np.load(OUT/e['muscle_data']);l=m['fiber_lengths'];q=m['joint_angles'];dt=1/c['path']['sample_rate_hz'];actual=m['wrist_xyz_reference_world'];error=np.linalg.norm(actual-p['xyz'],axis=1);lo=int(p['analysis_start_index']);hi=int(p['analysis_end_index']);span=np.ptp(actual[lo:hi,:2],axis=0);ar=span[0]/span[1]
  speed=np.abs(np.gradient(l,dt,axis=0)).max();joint_violations=int(np.any((q<lower-1e-3)|(q>upper+1e-3),axis=1).sum());finite=bool(np.isfinite(l).all() and np.isfinite(q).all() and np.isfinite(actual).all());passed=finite and error.max()<.2 and (l/opt).min()>.1 and (l/opt).max()<3 and speed<500 and joint_violations==0 and abs(ar-float(p['AR']))<.03 and np.ptp(actual[:,2])<.2
  rows.append(dict(id=e['id'],passed=passed,finite=finite,input_joint_limit_violation_frames=joint_violations,max_tracking_error_cm=error.max(),max_height_drift_cm=np.ptp(actual[:,2]),actual_AR=ar,requested_AR=float(p['AR']),min_fiber_mm=l.min(),max_fiber_mm=l.max(),min_normalized_fiber=(l/opt).min(),max_normalized_fiber=(l/opt).max(),peak_fiber_speed_mm_s=speed,elbow_min_deg=q[:,3].min(),elbow_max_deg=q[:,3].max(),max_joint_speed_deg_s=np.abs(np.gradient(q,dt,axis=0)).max()))
 df=pd.DataFrame(rows);df.to_csv(OUT/'quality.csv',index=False);print(df.to_string(index=False))
 if not df.passed.all():raise RuntimeError('Input-quality screen failed; do not infer until reviewed.')

def inference():
 import torch
 from scipy.signal import savgol_filter
 from utils.spindle_FR_helper import normalize,load_coefficients,get_sampled_coefficients
 from extract_data.generate_train_test_data import process_chunk
 from inference.test_model_utils_new import load_model,parse_config_value
 assert pd.read_csv(OUT/'quality.csv').passed.all()
 torch.set_num_threads(2);c=yaml.safe_load((ROOT/'extract_data/configs/train_test_data_spindles_extended.yaml').read_text());normalized={};windows={};times={};metadata={}
 for e in entries():
  m=np.load(OUT/e['muscle_data']);p=np.load(OUT/e['desired_path']);l=m['fiber_lengths'].T[None].astype(np.float32);dt=np.median(np.diff(m['times']));v=savgol_filter(np.gradient(l,dt,axis=2),31,1,axis=2).astype(np.float32);a=np.gradient(v,dt,axis=2).astype(np.float32);normalized[e['id']]=normalize(l,v,a,c['optimal_lengths']);indices=np.arange(int(p['analysis_start_index']),int(p['analysis_end_index']),4);windows[e['id']]=indices;times[e['id']]=m['times'];metadata[e['id']]=e
 rows=[];training=[];provenance=[];(OUT/'variants').mkdir(exist_ok=True);(OUT/'matched_spindles').mkdir(exist_ok=True)
 for coef in range(5):
  c['seed']=coef
  for kind in ['i_a','ii']:
   base=ROOT/'data/extended_spindle_coefficients'/kind/'linear';c[kind+'_coeff_path']=str(base/'coefficients.csv');c[kind+'_sampled_coeff_path']=str(base/f'sampled_coefficients_{kind}_5_{coef}.csv')
  coefficients={k:load_coefficients(c[k+'_coeff_path']) for k in ['i_a','ii']};sampled=get_sampled_coefficients(c,[5,5],c['muscles'],coefficients);rates={}
  caps=np.zeros((10,25))
  for ki,kind in enumerate(['i_a','ii']):
   for muscle in c['muscles']:
    for j,index in enumerate(sampled[kind][muscle]):caps[ki*5+j,muscle]=coefficients[kind][muscle]['max_rate'][index]
  for key,data in normalized.items():
   r=process_chunk(data,coefficients,[5,5],c['muscles'],chunk_size=1,sampled_coefficients=sampled).astype(np.float32);rates[key]=r
   assert np.isfinite(r).all() and (r>=0).all() and (r<=caps[None,:,:,None]+1e-4).all()
   np.savez_compressed(OUT/'matched_spindles'/f'{coef}_{key}.npz',firing_rates=r,times=times[key],coefficient_seed=coef)
   for kind,sl in [('Ia',slice(0,5)),('II',slice(5,10))]:
    v=r[0,sl];rows.append(dict(id=key,coefficient_seed=coef,type=kind,min_Hz=v.min(),median_Hz=np.median(v),p99_Hz=np.quantile(v,.99),max_Hz=v.max(),zero_fraction=(v==0).mean(),upper_cap_fraction=np.isclose(v,caps[sl,:,None],atol=1e-4,rtol=1e-6).mean()))
  for rep in [0,1,2,9]:
   ident=f'{coef}_{rep}';path=BASE/f'spatiotemporal_4_8-8-32-64_7171_{ident}';mc={k:parse_config_value(v) for k,v in yaml.safe_load((path/'config.yaml').read_text()).items()};mc['padding_mode']='replicate';tester=load_model(mc,str(path),'letter_reconstruction_joints',torch.device('cpu'),None,save_dir=str(ROOT));tester.model.eval();dest=OUT/'variants'/ident;dest.mkdir(exist_ok=True)
   mean=np.asarray(mc['train_mean']).reshape(1,10,25,1);std=np.asarray(mc['train_std']).reshape(1,10,25,1)
   with torch.inference_mode():
    for key,r in rates.items():
     x,_=tester._normalize_batch(torch.from_numpy(r),torch.zeros((1,r.shape[-1],7)));s,_=tester._compute_scores(x);pred=s[0].cpu().numpy();indices=windows[key];assert np.isfinite(pred).all();np.savez_compressed(dest/f'{key}.npz',predictions=pred[indices],indices=indices,times=times[key][indices]);z=np.abs((r[:,:,:,indices]-mean)/(std+1e-8))
     training.append(dict(id=key,variant=ident,abs_z_over_3_fraction=(z>3).mean(),abs_z_over_5_fraction=(z>5).mean(),max_abs_z=z.max()))
   provenance.append(dict(variant=ident,coefficient_seed=coef,replicate_id=rep,config_recorded_seed=mc.get('seed'),checkpoint_sha256=hashlib.sha256((path/'model.ckpt').read_bytes()).hexdigest()));print('Completed',ident,len(rates),'trajectories',flush=True)
 pd.DataFrame(rows).to_csv(OUT/'spindle_quality.csv',index=False);pd.DataFrame(training).to_csv(OUT/'training_range.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps(dict(n_models=20,n_trajectories=len(normalized),config_sha256=hashlib.sha256((ROOT/'experiments/soechting_2005.yaml').read_bytes()).hexdigest(),temporal_padding='replicate',preprocessing='original 31-point first-order Savitzky-Golay fiber velocity smoothing; no rate resampling',variants=provenance),indent=2))

def readout():
 import opensim as osm
 from utils.opensim_pose import CoupledPose
 from utils.trajectory_shape import fit_ellipse
 from process.generate_paths.soechting_horizontal import alpha_from_ar
 from process.ellipse_error_analysis import crossing_bracket
 c=config();source=ROOT/'MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim';model=osm.Model(str(source));state=model.initSystem();pose=CoupledPose(model,source);cs=model.getCoordinateSet();mk=model.getMarkerSet()
 def marker(n):
  v=mk.get(n).getLocationInGround(state);return np.array([v.get(j) for j in range(3)])
 pose.set_radians(state,{n:np.deg2rad(v) for n,v in {**c['path']['rest_pose_degrees'],**c['ik']['locked_wrist_degrees']}.items()});model.realizePosition(state);origin=marker('R.Shoulder');lower=np.array([cs.get(n).getRangeMin() for n in NAMES]);upper=np.array([cs.get(n).getRangeMax() for n in NAMES]);stimuli=pd.read_csv(OUT/'stimuli.csv').set_index('id');rows=[]
 (OUT/'readouts').mkdir(exist_ok=True)
 for entry in entries():
  ident=entry['id'];info=stimuli.loc[ident];actual=np.load(OUT/entry['muscle_data'])['wrist_xyz_reference_world'];all_xyz=[]
  for coef in range(5):
   for rep in [0,1,2,9]:
    variant=f'{coef}_{rep}';saved=np.load(OUT/'variants'/variant/f'{ident}.npz');angles=saved['predictions'][:,3:7];xyz=[]
    for a in angles:
     pose.set_radians(state,dict(zip(NAMES,np.deg2rad(a))));model.realizePosition(state);xyz.append(R@(marker('Handle')-origin)*100)
    xyz=np.array(xyz);all_xyz.append(xyz);truth=actual[saved['indices']];delta=xyz-truth;xy=-xyz[:,:2];span=np.ptp(xy,axis=0);ar=span[0]/span[1];pa=alpha_from_ar(ar);ellipse=fit_ellipse(xy);centered=delta-delta.mean(axis=0);bound=np.any((np.deg2rad(angles)<lower)|(np.deg2rad(angles)>upper),axis=1).mean()
    rows.append(dict(id=ident,**info.to_dict(),variant=variant,coefficient_seed=coef,replicate_id=rep,predicted_AR=ar,predicted_alpha=pa,predicted='circle' if 1/1.05<=ar<=1.05 else 'wide' if ar>1 else 'narrow',rmse_cm=np.sqrt(np.mean(np.sum(delta**2,axis=1))),centered_rmse_cm=np.sqrt(np.mean(np.sum(centered**2,axis=1))),right_bias_cm=-delta[:,0].mean(),forward_bias_cm=-delta[:,1].mean(),height_bias_cm=delta[:,2].mean(),height_excursion_cm=np.ptp(xyz[:,2]),outside_joint_fraction=bound,fit_axis_ratio=ellipse['axis_ratio'],fit_angle_deg=ellipse['angle_deg'],fit_residual_cm=ellipse['fit_rmse_cm']))
  np.savez_compressed(OUT/'readouts'/f'{ident}.npz',predicted_xyz=np.array(all_xyz),actual_xyz=truth,times=saved['times'],variant_ids=np.array([f'{i}_{j}' for i in range(5) for j in [0,1,2,9]]))
  print('Read out',ident,flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'all_predictions.csv',index=False);pses=[]
 for (exp,variant,condition),g in df.groupby(['study_experiment','variant','condition']):
  root,lo,hi=crossing_bracket(g.alpha,g.predicted_alpha);pses.append(dict(study_experiment=exp,variant=variant,condition=condition,pse=root,bracket_low=lo,bracket_high=hi))
 pd.DataFrame(pses).to_csv(OUT/'model_pses.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['quality','inference','readout']);a=p.parse_args();globals()[a.stage]()
