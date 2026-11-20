"""Summarize all matched variants equally; never selects for human agreement."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/research_audit/checkpoint_comparison'
df=pd.read_csv(out/'all_variant_results.csv');df['fraction_correct']=(df.correct==df.predicted).astype(float)
metrics=df.groupby(['experiment','variant','coefficient_seed','replicate_id']).agg(mean_rmse_cm=('rmse_cm','mean'),fraction_correct=('fraction_correct','mean'),mean_right_bias_cm=('right_bias_cm','mean'),mean_outside_joint_fraction=('outside_joint_fraction','mean')).reset_index();metrics.to_csv(out/'per_variant_metrics.csv',index=False)
summary=metrics.groupby('experiment').agg(n_variants=('variant','nunique'),mean_rmse_cm=('mean_rmse_cm','mean'),sd_rmse_across_variants_cm=('mean_rmse_cm','std'),min_variant_rmse_cm=('mean_rmse_cm','min'),max_variant_rmse_cm=('mean_rmse_cm','max'),mean_fraction_correct=('fraction_correct','mean'),sd_fraction_correct=('fraction_correct','std'),mean_right_bias_cm=('mean_right_bias_cm','mean'),sd_right_bias_cm=('mean_right_bias_cm','std'),mean_outside_joint_fraction=('mean_outside_joint_fraction','mean')).reset_index();summary.to_csv(out/'summary_metrics.csv',index=False)
metrics.groupby(['experiment','coefficient_seed']).agg(n_replicates=('variant','nunique'),mean_rmse_cm=('mean_rmse_cm','mean'),within_group_sd_rmse_cm=('mean_rmse_cm','std'),mean_fraction_correct=('fraction_correct','mean'),mean_right_bias_cm=('mean_right_bias_cm','mean')).to_csv(out/'coefficient_group_metrics.csv')
pse=pd.read_csv(out/'variant_pses.csv');pse.groupby('condition').agg(n_total=('variant','size'),n_unique_crossings=('pse','count'),mean_pse=('pse','mean'),sd_pse_across_variants=('pse','std'),min_pse=('pse','min'),max_pse=('pse','max')).to_csv(out/'pse_summary.csv')
wide=pse.pivot(index='variant',columns='condition',values='pse');shifts=pd.DataFrame({'B-A':wide.B-wide.A,'C-A':wide.C-wide.A});shifts.to_csv(out/'variant_pse_shifts.csv')
shifts.agg(['count','mean','std','min','max']).to_csv(out/'shift_summary.csv')
# Label proportions, not a nonlinear label of an averaged path.
df.groupby(['experiment','id','predicted']).size().rename('n_variants').to_csv(out/'label_frequencies.csv')
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
shape=df[df.experiment!='circle_position_grid'];colors={'A':'#557b9e','B':'#cf7957','C':'#6b9b82'}
for condition,g in shape.groupby('condition'):
 for _,v in g.groupby('variant'):axes[0,0].plot(v.sort_values('e').e,v.sort_values('e').predicted_e,color=colors[condition],alpha=.1,lw=.7)
 s=g.groupby('e').predicted_e.agg(['mean','std']).reset_index();axes[0,0].plot(s.e,s['mean'],color=colors[condition],label=condition);axes[0,0].fill_between(s.e,s['mean']-s['std'],s['mean']+s['std'],alpha=.13,color=colors[condition])
axes[0,0].plot([-1,1],[-1,1],'k--',lw=.8);axes[0,0].set(xlabel='Physical signed eccentricity',ylabel='Decoded signed eccentricity',title='Equal-weight mean ± SD across 20 variants');axes[0,0].legend()
human=pd.read_csv(ROOT/'outputs/viviani_aligned/human_model_comparison.csv')
for i,condition in enumerate('ABC'):
 v=pse[pse.condition==condition].pse.dropna();axes[0,1].scatter(np.full(len(v),i)+np.linspace(-.12,.12,len(v)),v,c=colors[condition],s=18,alpha=.7)
 if len(v)>1:axes[0,1].errorbar(i+.2,v.mean(),yerr=v.std(ddof=1),fmt='rs',capsize=4)
 h=human[human.condition==condition].iloc[0];axes[0,1].errorbar(i-.25,h.human_pse,yerr=[[h.human_pse-h.human_ci95_low],[h.human_ci95_high-h.human_pse]],fmt='ko',capsize=4)
axes[0,1].set(xticks=range(3),xticklabels=list('ABC'),ylabel='Physical signed eccentricity at equal extents',title='Each valid model PSE; red = mean ± SD\nBlack = human mean ± 95% CI (different uncertainty)');axes[0,1].axhline(0,c='gray',lw=.7)
for i,folder in enumerate(['viviani_aligned','circle_position_grid']):
 v=metrics[metrics.experiment==folder];axes[1,0].scatter(i+np.linspace(-.14,.14,len(v)),v.mean_rmse_cm,s=20);axes[1,0].errorbar(i+.23,v.mean_rmse_cm.mean(),yerr=v.mean_rmse_cm.std(),fmt='ks',capsize=4)
axes[1,0].set(xticks=[0,1],xticklabels=['21 ellipse trials','100 grid circles'],ylabel='Mean 3D RMSE (cm)',title='Per-variant mean error; black = mean ± SD')
grid=df[df.experiment=='circle_position_grid'];bias=grid.groupby('id').right_bias_cm.agg(['mean','std']).reset_index();axes[1,1].errorbar(range(len(bias)),bias['mean'],yerr=bias['std'],fmt='.',capsize=1,ms=3);axes[1,1].axhline(0,c='gray',lw=.7);axes[1,1].set(xlabel='Grid location index (row-major order)',ylabel='Predicted − actual, participant-right (cm)',title='Grid center bias: mean ± SD across variants\nPositive = predicted too far right')
fig.suptitle('Separate robustness comparison — all 20 matched archived checkpoints\nModel spread is not a human population confidence interval',fontsize=14)
fig.savefig(out/'checkpoint_sensitivity.png',dpi=150)
print(summary.to_string(index=False));print(pse.groupby('condition').pse.agg(['count','mean','std','min','max']).to_string());print(shifts.agg(['count','mean','std']).to_string())
