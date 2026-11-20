"""Bounded OpenSim marker IK on a periodic phase grid, validated at every frame.
Same model/couplers as the original pipeline; spline interpolation only accelerates
repeated geometry across speed conditions. It never interpolates spindle rates.
"""
from pathlib import Path
import sys,os,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,yaml,opensim as osm
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from utils.opensim_pose import CoupledPose
NAMES=['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion'];R=np.array([[0,0,-1],[-1,0,0],[0,1,0]])
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='experiments/soechting_2005.yaml');a=p.parse_args();os.environ['MOTOR_META_CONFIG']=str(Path(a.config).resolve())
 from paths import EXPERIMENT_CONFIG as cfg,MODEL_PATH,IK_DIR,CENTEROUT_DIR
 from experiment import load_manifest,resolve_artifact,set_artifact
 model=osm.Model(MODEL_PATH);state=model.initSystem();pose=CoupledPose(model,MODEL_PATH);cs=model.getCoordinateSet();mk=model.getMarkerSet();opt=cfg['path']['horizontal_ellipses'];rest=np.array([cfg['path']['rest_pose_degrees'][n] for n in NAMES]);brace=cfg['ik']['locked_wrist_degrees']
 pose.set_radians(state,{**dict(zip(NAMES,np.deg2rad(rest))),**{n:np.deg2rad(v) for n,v in brace.items()}});model.realizePosition(state)
 def marker(n):
  v=mk.get(n).getLocationInGround(state);return np.array([v.get(j) for j in range(3)])
 origin=marker('R.Shoulder');lower=np.rad2deg([cs.get(n).getRangeMin() for n in NAMES])+1e-5;upper=np.rad2deg([cs.get(n).getRangeMax() for n in NAMES])-1e-5
 def forward(q):
  pose.set_radians(state,dict(zip(NAMES,np.deg2rad(q))));model.realizePosition(state);return R@(marker('Handle')-origin)*100
 manifest=load_manifest();cache={};knots=np.linspace(0,2*np.pi,int(opt['ik_phase_knots'])+1);rows=[]
 muscles=['CORB','DELT1','DELT2','DELT3','INFSP','LAT1','LAT2','LAT3','PECM1','PECM2','PECM3','SUBSC','SUPSP','TMAJ','TMIN','ANC','BIClong','BICshort','BRA','BRD','ECRL','PT','TRIlat','TRIlong','TRImed']
 optimal=np.array(yaml.safe_load(Path('extract_data/configs/train_test_data_spindles_extended.yaml').read_text())['optimal_lengths']);ms=model.getMuscles()
 for e in manifest['trajectories']:
  desired=np.load(resolve_artifact(e['desired_path']));alpha=float(desired['alpha']);A=float(desired['A']);B=float(desired['B']);center=desired['center_xyz'];q0=rest.copy()
  if alpha not in cache:
   solved=[];fiber=[];max_error=0.
   lower[1]=max(lower[1],float(opt.get('minimum_shoulder_elevation_deg',0)))
   for phase_index,phase in enumerate(np.tile(knots[:-1],int(opt.get('ik_warmup_loops',0))+1)):
    target=center-np.array([A*np.sin(phase),B*np.cos(phase),0.])
    def residual(q):return np.r_[forward(q)-target,.001*(q-rest)]
    fit=least_squares(residual,np.clip(q0,lower,upper),bounds=(lower,upper),diff_step=1e-4,max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    q0=fit.x;error=np.linalg.norm(forward(q0)-target)
    if phase_index < len(knots[:-1])*int(opt.get('ik_warmup_loops',0)):continue
    max_error=max(max_error,error);solved.append(q0.copy());model.equilibrateMuscles(state);fiber.append([ms.get(n).getFiberLength(state)*1000 for n in muscles])
   solved=np.array(solved)
   # Fit the entire closed joint loop jointly. Periodic smoothness prevents
   # redundant-IK branch changes from producing fiber/velocity discontinuities.
   from scipy.sparse import lil_matrix
   n=len(solved);targets=center-np.column_stack([A*np.sin(knots[:-1]),B*np.cos(knots[:-1]),np.zeros(n)])
   sparsity=lil_matrix((11*n,4*n),dtype=int)
   for k in range(n):
    sparsity[3*k:3*k+3,4*k:4*k+4]=1
    for j in [(k-1)%n,k,(k+1)%n]:sparsity[3*n+4*k:3*n+4*k+4,4*j:4*j+4]=1
    sparsity[7*n+4*k:7*n+4*k+4,4*k:4*k+4]=1
   def loop_residual(flat):
    q=flat.reshape(n,4);hand=np.array([forward(v) for v in q]);smooth=np.roll(q,1,axis=0)-2*q+np.roll(q,-1,axis=0)
    return np.r_[(hand-targets).ravel(),(.1*smooth).ravel(),(.0005*(q-rest)).ravel()]
   fit=least_squares(loop_residual,np.clip(solved,lower,upper).ravel(),bounds=(np.tile(lower,n),np.tile(upper,n)),jac_sparsity=sparsity.tocsr(),diff_step=1e-4,max_nfev=200,ftol=1e-8,xtol=1e-8,gtol=1e-8)
   solved=fit.x.reshape(n,4);max_error=max(np.linalg.norm(forward(v)-t) for v,t in zip(solved,targets));fiber=[]
   for q in solved:
    forward(q);model.equilibrateMuscles(state);fiber.append([ms.get(name).getFiberLength(state)*1000 for name in muscles])
   fiber=np.array(fiber);spline=CubicSpline(knots,np.vstack([solved,solved[0]]),bc_type='periodic');cache[alpha]=spline
   np.savez_compressed(Path(IK_DIR)/f'phase_{alpha:+.2f}.npz',phase=knots[:-1],joint_angles=solved,fiber_lengths=fiber)
   if max_error>opt['tracking_tolerance_cm'] or not np.isfinite(fiber).all() or (fiber/optimal).min()<.1 or (fiber/optimal).max()>3:
    raise RuntimeError(f'Phase geometry invalid for alpha {alpha}: error {max_error:.3f} cm; fibers {fiber.min():.2f}..{fiber.max():.2f} mm, normalized {(fiber/optimal).min():.2f}..{(fiber/optimal).max():.2f}')
   print('Validated spatial loop',alpha,'max knot error',max_error,'fibers',fiber.min(),fiber.max(),flush=True)
  # Identical phases across cycles yield identical angle bit patterns.
  proximal=cache[alpha](desired['phase']);angles=np.column_stack([proximal,*[np.full(len(proximal),brace.get(n,0.)) for n in ['pro_sup','deviation','flexion']]])
  actual=np.array([forward(q) for q in proximal]);error=np.linalg.norm(actual-desired['xyz'],axis=1)
  if error.max()>opt['tracking_tolerance_cm']:raise RuntimeError('Interpolated IK tracking failed: '+e['id'])
  dest=Path(IK_DIR)/(e['id']+'.npz');np.savez_compressed(dest,joint_angles=angles,times=desired['times'],wrist_xyz_reference_world=actual,reference_origin_ground_m=origin,marker_error_cm=error)
  set_artifact(e['id'],'ik_solution',dest);rows.append(dict(id=e['id'],max_hand_error_cm=error.max(),max_joint_speed_deg_s=np.abs(np.gradient(angles,1/240,axis=0)).max(),elbow_min_deg=angles[:,3].min(),elbow_max_deg=angles[:,3].max()))
 pd.DataFrame(rows).to_csv(Path(CENTEROUT_DIR)/'ik_quality.csv',index=False)
if __name__=='__main__':main()
