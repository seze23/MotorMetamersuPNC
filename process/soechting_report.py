"""Scientific figures and explicit aggregate comparisons; no human data fitting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np,pandas as pd,yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from process.ellipse_error_analysis import crossing_bracket
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/soechting_2005';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
COLORS={'N':'#4477aa','FS':'#dd7744','FT':'#449977'}
def main():
 stim=pd.read_csv(OUT/'stimuli.csv');human=pd.read_csv(ROOT/'notebooks/data/soechting_2005_human_summary.csv');manifest=yaml.safe_load((OUT/'manifest.yaml').read_text());entries={e['id']:e for e in manifest['trajectories']}
 fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for c,ax in zip(['N','FS','FT'],axes):
  for alpha in [-.25,0,1]:
   row=stim[(stim.study_experiment==2)&(stim.condition==c)&(stim.alpha==alpha)].iloc[0];p=np.load(OUT/entries[row.id]['desired_path']);lo,hi=int(p['analysis_start_index']),int(p['analysis_end_index']);ax.plot(*p['paper_xy'][lo:hi].T,label=f'α={alpha:+g}')
  ax.scatter(0,15,c='k',marker='x',label='Circle start');ax.set(title=c,xlabel='Right from path center (cm)',ylabel='Forward from path center (cm)',aspect='equal');ax.grid(alpha=.2)
 axes[-1].legend(fontsize=8);fig.suptitle('Published size, horizontal plane: constant area, varying width/forward depth');fig.savefig(FIG/'stimuli.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for exp,ax in zip([1,2],axes):
  for condition in ['N','FS','FT']:
   row=stim[(stim.study_experiment==exp)&(stim.condition==condition)&(stim.alpha==0)].iloc[0];p=np.load(OUT/entries[row.id]['desired_path']);n=round(row.period_s*240);ax.plot(p['times'][:n],p['speed_cm_s'][:n],label=condition,color=COLORS[condition])
  ax.set(title=f'Experiment {exp}: circle speed',xlabel='Time in one revolution (s)',ylabel='Tangential speed (cm/s)');ax.legend()
 fig.savefig(FIG/'speeds.png',dpi=150);plt.close(fig)
 quality=pd.read_csv(OUT/'quality.csv');rates=pd.read_csv(OUT/'spindle_quality.csv');train=pd.read_csv(OUT/'training_range.csv')
 fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
 axes[0].plot(quality.max_tracking_error_cm*10,'.');axes[0].axhline(2,c='r',ls='--',label='2-mm screen');axes[0].set(ylabel='Maximum hand error (mm)',xlabel='Trajectory index',title='Endpoint tracking');axes[0].legend(fontsize=8)
 axes[1].plot(quality.peak_fiber_speed_mm_s,'.');axes[1].axhline(500,c='r',ls='--');axes[1].set(ylabel='Maximum |fiber velocity| (mm/s)',xlabel='Trajectory index',title='Fiber continuity screen')
 g=rates.groupby(['coefficient_seed','type']).max_Hz.max().unstack('type').reindex(columns=['Ia','II']);g.plot.bar(ax=axes[2],legend=False);axes[2].set(title='Maximum synthetic firing rates',ylabel='Hz',xlabel='Spindle coefficient group');axes[2].legend(['Ia','II'])
 fig.suptitle('Engineering screens, not proof of human physiological validity');fig.savefig(FIG/'quality.png',dpi=150);plt.close(fig)
 # One actual arm example reconstructed from saved physical markers and reference.
 # Joint-angle ranges are shown instead of implying unreported human angles.
 fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 row=stim[(stim.study_experiment==2)&(stim.condition=='N')&(stim.alpha==0)].iloc[0];m=np.load(OUT/entries[row.id]['muscle_data']);p=np.load(OUT/entries[row.id]['desired_path']);lo,hi=int(p['analysis_start_index']),int(p['analysis_end_index']);t=m['times'][lo:hi]-m['times'][lo]
 for i,name in enumerate(['Shoulder plane','Shoulder elevation','Shoulder rotation','Elbow flexion']):axes[0].plot(t,m['joint_angles'][lo:hi,i],label=name)
 axes[0].set(title='Actual model joint angles: example circle',xlabel='Cycle time (s)',ylabel='Degrees');axes[0].legend(fontsize=8)
 actual=m['wrist_xyz_reference_world'][lo:hi];desired=p['xyz'][lo:hi];axes[1].plot(*(-desired[:,:2]).T,'k--',label='Requested');axes[1].plot(*(-actual[:,:2]).T,label='Achieved');axes[1].scatter(*(-desired[0,:2]),c='k',marker='x',label='Start');axes[1].set(title='Horizontal hand path at published size',xlabel='Participant right from reference shoulder (cm)',ylabel='Forward from reference shoulder (cm)',aspect='equal');axes[1].legend()
 fig.savefig(FIG/'arm_and_tracking.png',dpi=150);plt.close(fig)
 df=pd.read_csv(OUT/'all_predictions.csv');pse=pd.read_csv(OUT/'model_pses.csv');ensemble=[]
 fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
 for row in stim.itertuples():
  r=np.load(OUT/'readouts'/f'{row.id}.npz');pred=r['predicted_xyz'];truth=r['actual_xyz'];mean=pred.mean(axis=0);std=pred.std(axis=0,ddof=1);span=np.ptp(mean[:,:2],axis=0);ar=span[0]/span[1];alpha=ar-1 if ar>=1 else 1-1/ar;delta=mean-truth
  ensemble.append(dict(id=row.id,study_experiment=row.study_experiment,condition=row.condition,alpha=row.alpha,predicted_AR=ar,predicted_alpha=alpha,predicted='circle' if 1/1.05<=ar<=1.05 else 'wide' if ar>1 else 'narrow',ensemble_rmse_cm=np.sqrt(np.mean(np.sum(delta**2,axis=1))),right_bias_cm=-delta[:,0].mean(),forward_bias_cm=-delta[:,1].mean(),height_bias_cm=delta[:,2].mean(),mean_pointwise_sd_cm=np.sqrt(np.sum(std**2,axis=1)).mean()))
  if row.alpha==0:
   ax=axes[row.study_experiment-1,['N','FS','FT'].index(row.condition)];xy=-pred[:,:,:2];actual=-truth[:,:2];center=actual.mean(axis=0)
   ax.plot(*(actual-center).T,'k--',lw=2,label='Actual circle')
   for v in xy:ax.plot(*(v-center).T,c=COLORS[row.condition],alpha=.12,lw=.6)
   ax.plot(*(-mean[:,:2]-center).T,c=COLORS[row.condition],lw=2,label='Mean of 20 predictions');ax.set(title=f'Experiment {row.study_experiment}: {row.condition}',xlabel='Right from actual center (cm)',ylabel='Forward from actual center (cm)',aspect='equal');ax.grid(alpha=.15)
 axes[0,-1].legend(fontsize=8);fig.suptitle('Actual circles, all variants, and the mean physical hand prediction\nOnly actual center removed from every path: prediction offsets remain visible');fig.savefig(FIG/'circle_predictions.png',dpi=150);plt.close(fig)
 ens=pd.DataFrame(ensemble);ens.to_csv(OUT/'ensemble_predictions.csv',index=False);roots=[]
 for (exp,condition),g in ens.groupby(['study_experiment','condition']):
  root,lo,hi=crossing_bracket(g.alpha,g.predicted_alpha);roots.append(dict(study_experiment=exp,condition=condition,ensemble_pse=root,ensemble_bracket_low=lo,ensemble_bracket_high=hi))
 roots=pd.DataFrame(roots);ps=pse.groupby(['study_experiment','condition']).agg(n_variants=('variant','size'),n_unique_crossings=('pse','count'),mean_variant_pse=('pse','mean'),sd_variant_pse=('pse','std')).reset_index();comp=human.merge(ps,on=['study_experiment','condition']).merge(roots,on=['study_experiment','condition']);comp.to_csv(OUT/'human_model_comparison.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
 for exp in [1,2]:
  for condition in ['N','FS','FT']:
   g=df[(df.study_experiment==exp)&(df.condition==condition)];s=g.groupby('alpha').predicted_alpha.agg(['mean','std']).reset_index();axes[exp-1,0].plot(s.alpha,s['mean'],color=COLORS[condition],label=condition);axes[exp-1,0].fill_between(s.alpha,s['mean']-s['std'],s['mean']+s['std'],color=COLORS[condition],alpha=.12)
  axes[exp-1,0].plot([-1,1],[-1,1],'k--',lw=.8);axes[exp-1,0].axhline(0,c='gray',lw=.6);axes[exp-1,0].set(title=f'Experiment {exp}: decoded shape mean ± SD',xlabel='Actual shape α (negative = narrow, positive = wide)',ylabel='Decoded shape α');axes[exp-1,0].legend()
  g=comp[comp.study_experiment==exp]
  for i,condition in enumerate(['N','FS','FT']):
   row=g[g.condition==condition].iloc[0];values=pse[(pse.study_experiment==exp)&(pse.condition==condition)].pse.dropna();axes[exp-1,1].scatter(i+np.linspace(-.1,.1,len(values)),values,c=COLORS[condition],s=15,alpha=.5)
   axes[exp-1,1].plot(i-.2,row.mean_alpha,'ko',label='Human mean (no CI available)' if i==0 else None)
   if np.isfinite(row.ensemble_pse):axes[exp-1,1].errorbar(i+.2,row.ensemble_pse,yerr=[[row.ensemble_pse-row.ensemble_bracket_low],[row.ensemble_bracket_high-row.ensemble_pse]],fmt='rD',capsize=3,label='Mean-path PSE + tested α bracket' if i==0 else None)
  axes[exp-1,1].axhline(0,c='gray',lw=.6);axes[exp-1,1].set(xticks=range(3),xticklabels=['N','FS','FT'],ylabel='Physical shape α at equal decoded widths',title='Human mean vs model circularity proxy\n'+('Human means derived from rounded aggregates' if exp==1 else 'Human means explicitly reported'));axes[exp-1,1].legend(fontsize=7)
 fig.savefig(FIG/'human_comparison.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained');x=np.arange(3)
 for exp,off in [(1,-.15),(2,.15)]:
  g=human[human.study_experiment==exp].set_index('condition').loc[['N','FS','FT']];axes[0].bar(x+off,g.difference_threshold_2beta,width=.3,label=f'Human exp. {exp}')
 
  if exp==2:
   for j,row in enumerate(g.itertuples()):
    if np.isfinite(row.beta_ci_low):axes[0].errorbar(j+off,row.difference_threshold_2beta,yerr=[[row.difference_threshold_2beta-2*row.beta_ci_low],[2*row.beta_ci_high-row.difference_threshold_2beta]],fmt='none',ecolor='k',capsize=3)
 axes[0].set(xticks=x,xticklabels=['N','FS','FT'],ylabel='Difference threshold = 2β',title='Published human sensitivity (no CNN JND)');axes[0].legend(fontsize=8)
 for exp in [1,2]:
  g=ens[ens.study_experiment==exp].groupby('condition').ensemble_rmse_cm.mean().reindex(['N','FS','FT']);axes[1].plot(x,g,'o-',label=f'Exp. {exp}: error of mean path')
 axes[1].set(xticks=x,xticklabels=['N','FS','FT'],ylabel='3D RMSE (cm)',title='Reconstruction accuracy');axes[1].legend(fontsize=7)
 for exp in [1,2]:
  g=df[(df.study_experiment==exp)&(df.alpha==.5)&(df.fit_axis_ratio>1.1)];axes[2].scatter(g.condition.map({'N':0,'FS':1,'FT':2})+(.05 if exp==2 else -.05),g.fit_angle_deg,s=12,alpha=.6,label=f'Exp. {exp}')
 axes[2].set(xticks=x,xticklabels=['N','FS','FT'],ylabel='Fitted major-axis angle from right (degrees)',title='Tilt diagnostic: actual α=+0.5\nOnly fits with major/minor >1.1');axes[2].legend(fontsize=7)
 fig.savefig(FIG/'sensitivity_error_tilt.png',dpi=150);plt.close(fig)
 # Correct sign: dots actual center, arrow tips ensemble predictions.
 fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for exp,ax in zip([1,2],axes):
  for condition in ['N','FS','FT']:
   row=stim[(stim.study_experiment==exp)&(stim.condition==condition)&(stim.alpha==0)].iloc[0];r=np.load(OUT/'readouts'/f'{row.id}.npz');actual=-r['actual_xyz'][:,:2].mean(axis=0);decoded=-r['predicted_xyz'][:,:,:2].mean(axis=(0,1));ax.scatter(*actual,c=COLORS[condition],s=30);ax.scatter(*decoded,alpha=0);ax.quiver(*actual,*(decoded-actual),angles='xy',scale_units='xy',scale=1,color=COLORS[condition],label=condition)
  ax.set(title=f'Experiment {exp}: mean circle-center error',xlabel='Right from reference shoulder (cm)',ylabel='Forward from reference shoulder (cm)',aspect='equal');ax.autoscale();xl=ax.get_xlim();yl=ax.get_ylim();xc=np.mean(xl);yc=np.mean(yl);xr=max(np.ptp(xl),2.)*1.2;yr=max(np.ptp(yl),1.4)*1.2;ax.set_xlim(xc-xr/2,xc+xr/2);ax.set_ylim(yc-yr/2,yc+yr/2);ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Dot = actual center; arrow tip = mean CNN-predicted center');fig.savefig(FIG/'center_bias.png',dpi=150,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
 for row in stim[stim.alpha==0].itertuples():
  r=np.load(OUT/'readouts'/f'{row.id}.npz');delta=r['predicted_xyz'][:,:,2]-r['actual_xyz'][None,:,2];phase=np.arange(delta.shape[1])/delta.shape[1];ax=axes[row.study_experiment-1,['N','FS','FT'].index(row.condition)]
  for values in delta:ax.plot(phase,values,c=COLORS[row.condition],alpha=.12,lw=.6)
  ax.plot(phase,delta.mean(axis=0),c=COLORS[row.condition],lw=2);ax.axhline(0,c='k',ls='--');ax.set(title=f'Experiment {row.study_experiment}: {row.condition}',xlabel='Fraction of analyzed revolution',ylabel='Predicted − actual height (cm)')
 fig.suptitle('Out-of-plane reconstruction error: actual movements are horizontal\nThin = individual CNN variants; bold = mean of 20; dashed = accurate height')
 fig.savefig(FIG/'out_of_plane.png',dpi=150);plt.close(fig)
 # Per-variant summaries keep model variation visible, and do not call it human CI.
 df.assign(correct_label=df.correct==df.predicted).groupby(['study_experiment','variant']).agg(mean_rmse_cm=('rmse_cm','mean'),fraction_correct=('correct_label','mean'),mean_right_bias_cm=('right_bias_cm','mean'),mean_forward_bias_cm=('forward_bias_cm','mean')).to_csv(OUT/'variant_summary.csv')
 summary=dict(n_models=20,n_trajectories=len(stim),n_predictions=len(df),all_input_screens_pass=bool(quality.passed.all()),max_tracking_error_mm=float(quality.max_tracking_error_cm.max()*10),fiber_range_mm=[float(quality.min_fiber_mm.min()),float(quality.max_fiber_mm.max())],peak_fiber_speed_mm_s=float(quality.peak_fiber_speed_mm_s.max()),max_rate_Hz=float(rates.max_Hz.max()),max_upper_saturation_fraction=float(rates.upper_cap_fraction.max()),mean_abs_z_over_3_fraction=float(train.abs_z_over_3_fraction.mean()),predicted_joint_limit_violation_fraction=float(df.outside_joint_fraction.mean()),ensemble_mean_rmse_cm=float(ens.ensemble_rmse_cm.mean()),mean_individual_rmse_cm=float(df.rmse_cm.mean()),ensemble_circle_labels=ens[ens.alpha==0][['study_experiment','condition','predicted','predicted_alpha','right_bias_cm','forward_bias_cm']].to_dict('records'))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(comp.to_string(index=False))
if __name__=='__main__':main()
