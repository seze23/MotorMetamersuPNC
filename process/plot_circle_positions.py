"""Scientific plots for vertical circles across an approximate workspace grid."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch,Circle,Rectangle,Patch
RIGHT_LABEL="Right of neck reference (cm)"
FORWARD_LABEL="Forward from neck reference (cm)"
BIAS_COLOR="#534fc4"
START_COLOR="#f4776d"
PALETTE={'circle':'#4c956c','horizontal ellipse':'#d98b3a','vertical ellipse':'#4f6db8','invalid':'#999999','invalid input':'#999999'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();figs=out/'figures';figs.mkdir(exist_ok=True)
    data=pd.read_csv(out/'circle_position_predictions.csv');positions=pd.read_csv(out/'positions.csv');manifest=yaml.safe_load((out/'manifest.yaml').read_text());entries={r['id']:r for r in manifest['trajectories']};cfg=yaml.safe_load((out/'experiment.yaml').read_text())
    p0=np.load(out/manifest['trajectories'][0]['desired_path']);m0=np.load(out/manifest['trajectories'][0]['muscle_data']);c7=p0['c7_xyz_world'];start=m0['wrist_xyz_reference_world'][0]
    common=np.array([-(start[0]-c7[0]),-(start[1]-c7[1]),start[2]-p0['center_xyz'][2]])
    def participant(xyz):
        xyz=np.asarray(xyz);return np.column_stack([-(xyz[:,0]-c7[0]),-(xyz[:,1]-c7[1]),xyz[:,2]-p0['center_xyz'][2]])
    selected=[positions.iloc[0].id,positions.iloc[19].id,positions.iloc[50].id,positions.iloc[-20].id,positions.iloc[-1].id]
    fig=plt.figure(figsize=(11,6));ax=fig.add_subplot(111,projection='3d')
    for ident in selected:
        p=np.load(out/entries[ident]['desired_path']);lo=int(p['circle_start_index']);hi=lo+600
        circle=participant(p['xyz'][lo:hi]);approach=participant(p['xyz'][:lo])
        ax.plot(*circle.T,label=ident);ax.plot(*approach.T,':',alpha=.4)
    ax.scatter(*common,facecolors='none',edgecolors=START_COLOR,marker='o',s=120,linewidths=2,label='Achieved common reach start')
    ax.scatter(*common,c=START_COLOR,marker='+',s=80,linewidths=2)
    ax.set(xlabel=RIGHT_LABEL,ylabel=FORWARD_LABEL,zlabel='Height from circle centers (cm)',title='Experiment layout: five examples of the 100 vertical circles\nDotted lines are approaches; only complete circle cycles are classified')
    ax.legend(fontsize=8,loc='upper left');fig.subplots_adjust(left=.02,right=.90,bottom=.07,top=.86);fig.savefig(figs/'circle_position_layout.png',dpi=150,bbox_inches='tight');plt.close(fig)
    def heat(ax,column,title,cbar_label,cmap='viridis',vmin=None,vmax=None):
        matrix=data.pivot(index='row',columns='column',values=column).sort_index().to_numpy()
        xs=np.sort(positions.right_cm.unique());ys=np.sort(positions.forward_cm.unique());dx=np.median(np.diff(xs));dy=np.median(np.diff(ys))
        im=ax.imshow(matrix,origin='lower',extent=[xs[0]-dx/2,xs[-1]+dx/2,ys[0]-dy/2,ys[-1]+dy/2],aspect='equal',cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set(xlabel=RIGHT_LABEL,ylabel=FORWARD_LABEL,title=title);fig.colorbar(im,ax=ax,label=cbar_label,shrink=.8)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for label,g in data.groupby('predicted'):
        axes[0].scatter(g.actual_right_cm,g.actual_forward_cm,c=PALETTE.get(label,'gray'),s=45,label=f'{label} (n={len(g)})')
    axes[0].scatter(common[0],common[1],facecolors='none',edgecolors=START_COLOR,marker='o',s=150,linewidths=2,label='Achieved common reach start')
    axes[0].scatter(common[0],common[1],c=START_COLOR,marker='+',s=100,linewidths=2)
    axes[0].set(xlabel=RIGHT_LABEL,ylabel=FORWARD_LABEL,title='Circle predictions by workspace location\nExisting horizontal/vertical extent rule',aspect='equal');axes[0].grid(alpha=.2);axes[0].legend(fontsize=8,bbox_to_anchor=(.5,-.18),loc='upper center',ncol=2)
    heat(axes[1],'fitted_axis_ratio','Rotation-aware ellipse fit in the vertical circle plane','Fitted major/minor axis ratio (1 = round)',vmin=1)
    fig.suptitle('Approximate 100-position grid from supplied image — one configured CNN');fig.savefig(figs/'circle_position_shape_map.png',dpi=150,bbox_inches='tight');plt.close(fig)
    rows=np.sort(positions.row.unique())[::-1];cols=np.sort(positions.column.unique());fig,axes=plt.subplots(len(rows),len(cols),figsize=(23,8),squeeze=False)
    readouts={};limit=float(cfg['path']['circle_positions']['radius_cm'])
    for item in data.itertuples():
        p=np.load(out/item.readout);actual=p['actual_xyz'][:,[0,2]]*[-1,1];decoded=p['decoded_xyz'][:,[0,2]]*[-1,1];actual-=actual.mean(axis=0);decoded-=decoded.mean(axis=0)
        readouts[item.id]=(actual,decoded);limit=max(limit,float(np.abs(actual).max()),float(np.abs(decoded).max()))
    limit*=1.08
    for i,row in enumerate(rows):
        for j,col in enumerate(cols):
            ax=axes[i,j];item=data[(data.row==row)&(data.column==col)].iloc[0];actual,decoded=readouts[item.id]
            ax.plot(*actual.T,'k--',lw=.6,alpha=.6);ax.plot(*decoded.T,c=PALETTE.get(item.predicted,'gray'),lw=1)
            ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',xticks=[],yticks=[])
            if i==0:ax.set_title(f'R {item.right_cm:.1f}',fontsize=7)
            if j==0:ax.set_ylabel(f'F {item.forward_cm:.1f} cm',fontsize=9)
            for spine in ax.spines.values():spine.set_color('#dddddd')
    handles=[Line2D([],[],c='black',ls='--',label='Actual IK circle')]+[Line2D([],[],c=c,label=k) for k,c in list(PALETTE.items())[:3]]
    fig.suptitle('Actual and decoded vertical circles at all 100 positions',fontsize=16,y=.995);fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.96),ncol=4,frameon=False)
    fig.text(.05,.015,'Paths are centered separately to show shape. Same cm scale in every panel. R = right from C7; F = forward from C7. Colors use the earlier extent-based readout.',fontsize=10)
    fig.subplots_adjust(left=.04,right=.995,bottom=.055,top=.88,wspace=.06,hspace=.12);fig.savefig(figs/'circle_position_mosaic.png',dpi=150,bbox_inches='tight');plt.close(fig)
    # Predicted centers are arrow tails;
    # actual achieved locations are arrowheads. Coordinates are C7 referenced.
    dx=data.actual_right_cm-data.decoded_right_cm;dy=data.actual_forward_cm-data.decoded_forward_cm
    fig,(ax,body)=plt.subplots(1,2,figsize=(15,6.5),gridspec_kw={'width_ratios':[3.4,1]})
    ax.quiver(data.decoded_right_cm,data.decoded_forward_cm,dx,dy,angles='xy',scale_units='xy',scale=1,color=BIAS_COLOR,width=.0028)
    ax.scatter(data.decoded_right_cm,data.decoded_forward_cm,c=BIAS_COLOR,s=14,zorder=3)
    ax.scatter(common[0],common[1],facecolors='none',edgecolors=START_COLOR,s=170,linewidths=2.5,zorder=5)
    ax.scatter(common[0],common[1],c=START_COLOR,marker='+',s=110,linewidths=2,zorder=6)
    ax.set(xlabel=RIGHT_LABEL,ylabel=FORWARD_LABEL,aspect='equal')
    ax.set_xlim(min(data.actual_right_cm.min(),data.decoded_right_cm.min())-.5,max(data.actual_right_cm.max(),data.decoded_right_cm.max())+.5)
    ax.set_ylim(min(data.actual_forward_cm.min(),data.decoded_forward_cm.min())-.5,max(data.actual_forward_cm.max(),data.decoded_forward_cm.max())+.5)
    ax.grid(alpha=.25)
    body.add_patch(FancyBboxPatch((-15,-4),30,7,boxstyle='round,pad=.2,rounding_size=3',facecolor='#dddddd',edgecolor='gray'))
    body.add_patch(Circle((0,1),3.1,facecolor='#eeeeee',edgecolor='gray'))
    body.scatter(0,0,c='gray',marker='x',s=35)
    xmin,xmax=positions.right_cm.min(),positions.right_cm.max();ymin,ymax=positions.forward_cm.min(),positions.forward_cm.max()
    body.add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,facecolor=BIAS_COLOR,edgecolor=BIAS_COLOR,alpha=.18))
    body.scatter(common[0],common[1],facecolors='none',edgecolors=START_COLOR,s=90,linewidths=2)
    body.scatter(common[0],common[1],c=START_COLOR,marker='+',s=55,linewidths=1.5)
    body.set(title='View from above',xlabel='Right (cm)',ylabel='Forward (cm)',xlim=(-19,19),ylim=(-5,39),aspect='equal',xticks=[-15,0,15],yticks=[0,15,30]);body.grid(alpha=.18)
    handles=[Line2D([],[],c=BIAS_COLOR,marker='>',markersize=8,label='Arrow: predicted center → actual center'),
             Line2D([],[],ls='none',marker='o',color=BIAS_COLOR,label='Dot: CNN-predicted circle center'),
             Line2D([],[],ls='none',marker='o',markerfacecolor='none',markeredgecolor=START_COLOR,markeredgewidth=2,markersize=9,label='Shared hand starting point'),
             Patch(facecolor='#dddddd',edgecolor='gray',label='Body / neck reference (0, 0)')]
    fig.suptitle('Where the CNN places each circle\n100 circles: predicted centers compared with actual centers',fontsize=15,y=.98)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,fontsize=10)
    fig.text(.5,.005,'Each arrow goes from the CNN’s guess to the actual center. Left arrow = guess too far right. Grid positions are approximate.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,bottom=.23,top=.83,wspace=.22)
    fig.savefig(figs/'circle_position_centroid_bias.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    heat(ax,'rmse_3d_cm','Circle reconstruction error by position','3D hand-position RMSE (cm)')
    fig.savefig(figs/'circle_position_error_map.png',dpi=150,bbox_inches='tight');plt.close(fig)
    summary=dict(n_positions=len(data),n_models=1,extent_labels=data.predicted.value_counts().to_dict(),fitted_axis_ratio_min=float(data.fitted_axis_ratio.min()),fitted_axis_ratio_max=float(data.fitted_axis_ratio.max()),mean_rmse_3d_cm=float(data.rmse_3d_cm.mean()),extent_circle_with_fitted_axis_ratio_over_1_05=int(((data.predicted=='circle')&(data.fitted_axis_ratio>1.05)).sum()))
    (out/'circle_position_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
