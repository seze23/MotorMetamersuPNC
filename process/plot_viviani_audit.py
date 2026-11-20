from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();out=root/'outputs/research_audit';p=np.load(out/'arm_snapshots.npz')
fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
for row,folder in enumerate(p.files):
 for j,pts in enumerate(p[folder]):
  ref=pts[0];v=pts-ref
  for col,(a,b) in enumerate([(2,1),(0,1)]):
   axes[row,col].plot(v[1:,a],v[1:,b],'o-',alpha=.9 if j==0 else .35,label='Start pose' if j==0 else None)
   axes[row,col].scatter(0,0,c='k',marker='x')
 for col in range(2):
  axes[row,col].set(title=('Ellipse experiment' if row==0 else 'Grid example')+(' — front view' if col==0 else ' — side view'),xlabel='Right of C7 (cm)' if col==0 else 'Forward from C7 (cm)',ylabel='Height above C7 (cm)',aspect='equal');axes[row,col].grid(alpha=.2)
  axes[row,col].legend(fontsize=8)
fig.suptitle('Actual saved model poses: shoulder → elbow → hand\nMarkers joined by straight lines; not a rendering of bone or palm orientation')
fig.savefig(out/'arm_posture_views.png',dpi=150);plt.close(fig)
df=pd.read_csv(out/'spindle_range_audit.csv');x=df[df.window=='readout'].copy();summary=x.groupby(['experiment','type']).agg(min_Hz=('minimum','min'),max_Hz=('maximum','max'),mean_zero_fraction=('zero_fraction','mean'),mean_upper_cap_fraction=('cap_fraction','mean'),mean_abs_z_over_3_fraction=('abs_z_over_3_fraction','mean'),max_abs_z=('max_abs_z','max'),nonfinite=('nonfinite','sum'),above_cap=('above_cap','sum')).reset_index();summary.to_csv(out/'spindle_summary.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for k,(experiment,g) in enumerate(x.groupby('experiment')):
 for j,kind in enumerate(['Ia','II']):
  v=g[g.type==kind];i=k*3+j
  axes[0].bar(i,v.maximum.max(),color='#567b9e' if kind=='Ia' else '#c07858');axes[1].bar(i,100*v.abs_z_over_3_fraction.mean(),color='#567b9e' if kind=='Ia' else '#c07858')
labels=[f'{e.replace("viviani_","").replace("circle_position_grid","grid")}\n{k}' for e in sorted(x.experiment.unique()) for k in ['Ia','II']];loc=[i*3+j for i in range(3) for j in range(2)]
for ax in axes:ax.set_xticks(loc,labels,rotation=20);ax.grid(axis='y',alpha=.2)
axes[0].set(title='Maximum rate during analyzed cycle',ylabel='Hz');axes[1].set(title='Channel values far from saved training mean',ylabel='Values with |z| > 3 (%)')
fig.suptitle('Numerical spindle audit — these are not physiological acceptance limits')
fig.savefig(out/'spindle_range_checks.png',dpi=150);plt.close(fig)
print(summary.to_string(index=False))
