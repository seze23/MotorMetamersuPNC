"""Separate, matched comparison of all 20 archived CNN variants; no training.
Inference runs in the Torch environment; physical readout runs in OpenSim.
Folder replicate identifiers are not asserted to be verified training seeds.
"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/research_audit/checkpoint_comparison'
FOLDERS=['viviani_aligned','viviani_aligned_refine','circle_position_grid']
BASE=ROOT/'trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints'
NAMES=['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion']
def inputs():
 for folder in FOLDERS:
  base=ROOT/'outputs'/folder;cfg=yaml.safe_load((base/'experiment.yaml').read_text())
  for entry in yaml.safe_load((base/'manifest.yaml').read_text())['trajectories']:
   yield folder,base,cfg,entry

def inference(audit_only=False):
 import torch
 from scipy.signal import savgol_filter
 from inference.test_model_utils_new import load_model,parse_config_value
 from utils.spindle_FR_helper import normalize,load_coefficients,get_sampled_coefficients
 from extract_data.generate_train_test_data import process_chunk
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
 cfg=yaml.safe_load((ROOT/'extract_data/configs/train_test_data_spindles_extended.yaml').read_text())
 normalized={};windows={};sources={}
 for folder,base,c,e in inputs():
  m=np.load(base/e['muscle_data']);p=np.load(base/e['desired_path']);t=m['times'];dt=np.median(np.diff(t));l=m['fiber_lengths'].T[None].astype(np.float32)
  v=savgol_filter(np.gradient(l,dt,axis=2),31,1,axis=2).astype(np.float32);a=np.gradient(v,dt,axis=2).astype(np.float32)
  key=folder+'/'+e['id'];normalized[key]=normalize(l,v,a,cfg['optimal_lengths'])
  lo=int(p['analysis_start_index']) if 'analysis_start_index' in p else len(t)-600
  hi=int(p['analysis_end_index']) if 'analysis_end_index' in p else len(t)
  windows[key]=np.arange(lo,hi,4);sources[key]=(base,e)
 provenance=[];signal_checks=[]
 for coef in range(5):
  cfg['seed']=coef
  for kind in ['i_a','ii']:
   b=ROOT/'data/extended_spindle_coefficients'/kind/'linear'
   cfg[kind+'_coeff_path']=str(b/'coefficients.csv');cfg[kind+'_sampled_coeff_path']=str(b/f'sampled_coefficients_{kind}_5_{coef}.csv')
  coefficients={k:load_coefficients(cfg[k+'_coeff_path']) for k in ['i_a','ii']}
  sampled=get_sampled_coefficients(cfg,[5,5],cfg['muscles'],coefficients)
  rates={k:process_chunk(v,coefficients,[5,5],cfg['muscles'],chunk_size=1,sampled_coefficients=sampled).astype(np.float32) for k,v in normalized.items()}
  caps=np.zeros((10,25))
  for ki,kind in enumerate(['i_a','ii']):
   for muscle in cfg['muscles']:
    for j,index in enumerate(sampled[kind][muscle]):caps[ki*5+j,muscle]=coefficients[kind][muscle]['max_rate'][index]
  for key,r in rates.items():
   assert np.isfinite(r).all() and r.min()>=0
   assert (r<=caps[None,:,:,None]+1e-4).all()
   signal_checks.append(dict(coefficient_seed=coef,trajectory=key,min_Hz=float(r.min()),max_Hz=float(r.max()),zero_fraction=float((r==0).mean()),upper_cap_fraction=float(np.isclose(r,caps[None,:,:,None],atol=1e-4,rtol=1e-6).mean())))
  if audit_only:continue
  if coef==0:
   for k,r in rates.items():
    b,e=sources[k];saved_rates=np.load(b/e['spindle_data'])['firing_rates'];np.testing.assert_allclose(r,saved_rates,rtol=1e-4,atol=1e-4);rates[k]=saved_rates
  for replicate in [0,1,2,9]:
   ident=f'{coef}_{replicate}';path=BASE/f'spatiotemporal_4_8-8-32-64_7171_{ident}';c={k:parse_config_value(v) for k,v in yaml.safe_load((path/'config.yaml').read_text()).items()};c['padding_mode']='replicate'
   tester=load_model(c,str(path),'letter_reconstruction_joints',torch.device('cpu'),None,save_dir=str(ROOT));tester.model.eval()
   dest=OUT/ident;dest.mkdir(exist_ok=True)
   with torch.inference_mode():
    for key,r in rates.items():
     x,_=tester._normalize_batch(torch.from_numpy(r),torch.zeros((1,r.shape[-1],7)));s,_=tester._compute_scores(x);pred=s[0].cpu().numpy();indices=windows[key]
     if ident=='0_9':
      b,e=sources[key];np.testing.assert_allclose(pred,np.load(b/e['raw_predictions'])['predictions'],rtol=2e-4,atol=2e-4)
     np.savez_compressed(dest/(key.replace('/','__')+'.npz'),predictions=pred[indices],indices=indices)
   provenance.append(dict(variant=ident,coefficient_seed=coef,replicate_id=replicate,config_seed=c.get('seed'),checkpoint_sha256=hashlib.sha256((path/'model.ckpt').read_bytes()).hexdigest()))
   print('Completed checkpoint',ident,flush=True)
 pd.DataFrame(signal_checks).to_csv(OUT/'all_coefficient_signal_checks.csv',index=False)
 if audit_only:return
 (OUT/'provenance.json').write_text(json.dumps(dict(n_checkpoints=20,n_trajectories=len(normalized),padding='replicate',selection='all 5 coefficient groups x all 4 available replicate folders',variants=provenance),indent=2))

def readout():
 import opensim as osm
 from utils.opensim_pose import CoupledPose
 from process.ellipse_readout import shape
 from process.ellipse_error_analysis import crossing_bracket
 path=ROOT/'MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim'
 model=osm.Model(str(path));state=model.initSystem();pose=CoupledPose(model,path);R=np.array([[0,0,-1],[-1,0,0],[0,1,0]])
 def marker(n):
  v=model.getMarkerSet().get(n).getLocationInGround(state);return np.array([v.get(i) for i in range(3)])
 lower=np.array([model.getCoordinateSet().get(n).getRangeMin() for n in NAMES]);upper=np.array([model.getCoordinateSet().get(n).getRangeMax() for n in NAMES])
 rows=[]
 for folder,base,c,e in inputs():
  pose.set_radians(state,{n:np.deg2rad(v) for n,v in {**c['path']['rest_pose_degrees'],**c['ik']['locked_wrist_degrees']}.items()});model.realizePosition(state);origin=marker('R.Shoulder')
  m=np.load(base/e['muscle_data']);
  info=None if folder=='circle_position_grid' else pd.read_csv(base/'stimuli.csv').set_index('id').loc[e['id']]
  for coef in range(5):
   for rep in [0,1,2,9]:
    ident=f'{coef}_{rep}';saved=np.load(OUT/ident/(folder+'__'+e['id']+'.npz'));angles=saved['predictions'][:,3:7];xyz=[]
    for a in angles:
     pose.set_radians(state,dict(zip(NAMES,np.deg2rad(a))));model.realizePosition(state);xyz.append(R@(marker('Handle')-origin)*100)
    xyz=np.array(xyz);actual=m['wrist_xyz_reference_world'][saved['indices']];label,ratio,pe=shape(xyz[:,[0,2]],1.05);delta=xyz-actual
    rows.append(dict(experiment=folder,id=e['id'],variant=ident,coefficient_seed=coef,replicate_id=rep,condition='A' if info is None else info.condition,e=0 if info is None else info.e,correct='true circle' if info is None else info.correct,predicted=label,ratio=ratio,predicted_e=pe,rmse_cm=np.sqrt(np.mean(np.sum(delta**2,axis=1))),right_bias_cm=-delta[:,0].mean(),forward_bias_cm=-delta[:,1].mean(),outside_joint_fraction=np.any((np.deg2rad(angles)<lower)|(np.deg2rad(angles)>upper),axis=1).mean()))
  print('Read out',folder,e['id'],flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'all_variant_results.csv',index=False);pses=[]
 for (variant,condition),g in df[df.experiment!='circle_position_grid'].groupby(['variant','condition']):
  pse,lo,hi=crossing_bracket(g.e,g.predicted_e);pses.append(dict(variant=variant,condition=condition,pse=pse,bracket_low=lo,bracket_high=hi))
 pd.DataFrame(pses).to_csv(OUT/'variant_pses.csv',index=False)
 print('Completed',len(df),'physical readouts')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['inference','readout','audit-signals']);a=p.parse_args();inference(audit_only=True) if a.stage=='audit-signals' else globals()[a.stage]()
