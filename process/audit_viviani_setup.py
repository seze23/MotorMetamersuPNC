"""Audit cached spindle bounds and arm posture; never changes inputs or weights.
Run from the repository root with .venv-mujoco/bin/python.
Reported zero fractions include the transfer function's rectification.
Upper-cap checks use the exact saved coefficient-seed-0 channel ordering.
"""
from pathlib import Path
import csv,ast,json
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/research_audit';out.mkdir(parents=True,exist_ok=True)
limits=[]
for kind in ['i_a','ii']:
 base=ROOT/'data/extended_spindle_coefficients'/kind/'linear'
 coeff={int(r['Muscle']):r for r in csv.DictReader(open(base/'coefficients.csv'))}
 samples=list(csv.reader(open(base/f'sampled_coefficients_{kind}_5_0.csv')))[1:]
 caps=np.zeros((5,25))
 for muscle,indices in samples:
  idx=np.fromstring(indices.strip('[]'),sep=' ',dtype=int)
  caps[:,int(muscle)]=np.array(ast.literal_eval(coeff[int(muscle)]['max_rate']))[idx]
 limits.append(caps)
caps=np.concatenate(limits)
model_cfg=yaml.safe_load((ROOT/'trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints/spatiotemporal_4_8-8-32-64_7171_0_9/config.yaml').read_text())
train_mean=np.array(model_cfg['train_mean']).reshape(10,25,1)
train_std=np.array(model_cfg['train_std']).reshape(10,25,1)
rows=[];postures=[]
for folder in ['viviani_aligned','viviani_aligned_refine','circle_position_grid']:
 base=ROOT/'outputs'/folder;manifest=yaml.safe_load((base/'manifest.yaml').read_text())
 for item in manifest['trajectories']:
  m=np.load(base/item['muscle_data']);p=np.load(base/item['desired_path']);s=np.load(base/item['spindle_data'])
  rates=s['firing_rates'][0];n=rates.shape[-1]
  lo=int(p['analysis_start_index']) if 'analysis_start_index' in p else n-600
  hi=int(p['analysis_end_index']) if 'analysis_end_index' in p else n
  for window,a in [('full',rates),('readout',rates[:,:,lo:hi])]:
   for kind,sl in [('Ia',slice(0,5)),('II',slice(5,10))]:
    v=a[sl];cap=caps[sl,:,None];z=(v-train_mean[sl])/(train_std[sl]+1e-8);qs=np.quantile(v,[.01,.5,.95,.99])
    rows.append(dict(experiment=folder,id=item['id'],window=window,type=kind,minimum=float(v.min()),p01=qs[0],median=qs[1],p95=qs[2],p99=qs[3],maximum=float(v.max()),zero_fraction=float((v<=0).mean()),abs_z_over_3_fraction=float((np.abs(z)>3).mean()),max_abs_z=float(np.abs(z).max()),cap_fraction=float(np.isclose(v,cap,atol=1e-4,rtol=1e-6).mean()),above_cap=int((v>cap+1e-3).sum()),nonfinite=int((~np.isfinite(v)).sum())))
  q=m['joint_angles'];lengths=m['fiber_lengths'];err=np.linalg.norm(m['wrist_xyz_reference_world']-p['xyz'],axis=1)
  postures.append(dict(experiment=folder,id=item['id'],elbow_min=float(q[:,3].min()),elbow_max=float(q[:,3].max()),elbow_start=float(q[0,3]),forearm_min=float(q[:,4].min()),forearm_max=float(q[:,4].max()),fiber_min_mm=float(lengths.min()),fiber_max_mm=float(lengths.max()),max_hand_error_cm=float(err.max())))
df=pd.DataFrame(rows);df.to_csv(out/'spindle_range_audit.csv',index=False);pd.DataFrame(postures).to_csv(out/'arm_posture_audit.csv',index=False)
print(df.groupby(['experiment','window','type']).agg(min_Hz=('minimum','min'),max_Hz=('maximum','max'),mean_zero_fraction=('zero_fraction','mean'),mean_cap_fraction=('cap_fraction','mean'),nonfinite=('nonfinite','sum'),above_cap=('above_cap','sum')).to_string())
print(pd.DataFrame(postures).groupby('experiment').agg(elbow_min=('elbow_min','min'),elbow_max=('elbow_max','max'),fiber_min=('fiber_min_mm','min'),fiber_max=('fiber_max_mm','max'),max_hand_error_cm=('max_hand_error_cm','max')).to_string())
print('CAP RANGE',caps.min(),caps.max())
