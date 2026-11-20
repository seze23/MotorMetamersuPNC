"""Extract saved arm poses and draw a side-view setup schematic.
The torso outline is illustrative; pose markers come from the unchanged OpenSim
model and the accepted E2 normal-speed circle. No CNN predictions are rerun.
"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/soechting_2005'
def extract():
 import opensim as osm
 from utils.opensim_pose import CoupledPose
 cfg=yaml.safe_load((ROOT/'experiments/soechting_2005.yaml').read_text());source=ROOT/'MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim';model=osm.Model(str(source));state=model.initSystem();pose=CoupledPose(model,source);markers=model.getMarkerSet()
 def marker(name):
  v=markers.get(name).getLocationInGround(state);return np.array([v.get(i) for i in range(3)])
 pose.set_radians(state,{k:np.deg2rad(v) for k,v in {**cfg['path']['rest_pose_degrees'],**cfg['ik']['locked_wrist_degrees']}.items()});model.realizePosition(state);c7=marker('C7');origin=marker('R.Shoulder')
 def participant(p):return ((np.asarray(p)-c7)*100)[...,[2,0,1]]
 e=next(e for e in yaml.safe_load((OUT/'manifest.yaml').read_text())['trajectories'] if e['id']=='E2_N_p0_00');m=np.load(OUT/e['muscle_data']);p=np.load(OUT/e['desired_path']);lo=int(p['analysis_start_index']);hi=int(p['analysis_end_index']);frames=[lo,lo+(hi-lo)//4,lo+(hi-lo)//2,lo+3*(hi-lo)//4];names=['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion','pro_sup','deviation','flexion'];poses=[]
 for frame in frames:
  pose.set_radians(state,dict(zip(names,np.deg2rad(m['joint_angles'][frame]))));model.realizePosition(state);poses.append(participant([marker(n) for n in ['R.Shoulder','R.Elbow.Lateral','Handle']]).tolist())
 R=np.array([[0,0,-1],[-1,0,0],[0,1,0]]);center_ground=p['center_xyz']/100@R+origin
 data=dict(reference_shoulder=participant(origin).tolist(),center=participant(center_ground).tolist(),poses=poses,source_trial=e['id'],frames=frames)
 (OUT/'setup_schematic_data.json').write_text(json.dumps(data,indent=2))
def plot():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Circle,FancyBboxPatch
 d=json.loads((OUT/'setup_schematic_data.json').read_text());S=np.array(d['reference_shoulder']);center=np.array(d['center']);poses=np.array(d['poses']);p=poses[0][:,[1,2]];sx,sz=S[[1,2]];cx,cz=center[[1,2]]
 fig,ax=plt.subplots(figsize=(10,6.5));ax.set_aspect('equal');ax.set(xlim=(-19,72),ylim=(-50,24));ax.axis('off')
 body=FancyBboxPatch((-6,sz-43),11,42,boxstyle='round,pad=.4,rounding_size=3',facecolor='#eeeeee',edgecolor='#929292',lw=1);ax.add_patch(body);ax.add_patch(Circle((0,sz+15),6.5,facecolor='#f5f5f5',edgecolor='#929292',lw=1));ax.plot([0,0],[sz+2,sz+8],c='#929292',lw=2)
 ax.plot([3,5],[sz+16,sz+16],c='#777777',lw=1);ax.plot([0],[0],marker='x',c='#555555',ms=7)
 ax.plot([cx-15,cx+15],[cz,cz],'--',color='#527f9a',lw=2,label='Horizontal circle seen edge-on')
 for points in poses[1:]:
  v=points[:,[1,2]];ax.plot(v[:,0],v[:,1],'-o',c='#777777',alpha=.18,lw=2,ms=4)
 ax.plot(p[:,0],p[:,1],'-o',c='#303c48',lw=5,ms=8,zorder=4);ax.scatter(*p[-1],c='#397e9e',s=95,zorder=5);ax.scatter(cx,cz,marker='+',c='#397e9e',s=85,zorder=5)
 def label(text,point,xy):
  ax.annotate(text,xy=point,xytext=xy,fontsize=11,ha='left',va='center',arrowprops=dict(arrowstyle='-',color='#777777',lw=1))
 label('Neck reference\n(C7)',(0,0),(-18,4));label('Torso\n(schematic)',(-1,sz-22),(-18,sz-19));label('Right shoulder',p[0],(12,sz-1));label('Elbow',p[1],(8,-39));label('Hand + handle\nat circle start',p[-1],(40,-17));label('Circle center',(cx,cz),(cx-4,cz-10))
 ax.annotate('',xy=(cx,7),xytext=(sx,7),arrowprops=dict(arrowstyle='<->',color='#777777',lw=1));ax.text((sx+cx)/2,9,'35 cm forward',ha='center',fontsize=11);ax.plot([sx,sx],[sz+1,6],':',c='#999999',lw=1)
 ax.annotate('',xy=(65,sz),xytext=(65,cz),arrowprops=dict(arrowstyle='<->',color='#777777',lw=1));ax.text(68,(sz+cz)/2,'30 cm below',rotation=90,ha='center',va='center',fontsize=11)
 ax.text(cx,cz-16,'A horizontal circle looks flat from the side.',ha='center',fontsize=11,color='#555555')
 fig.suptitle('Side view of the simulated arm setup',fontsize=16,y=.98)
 fig.text(.5,.915,'Solid arm: saved circle-start pose. Faint arms: later positions around the circle.',ha='center',fontsize=11)
 fig.text(.5,.035,'Placement distances are simulation assumptions, not measurements reported in the paper.\nDistances use the reference shoulder. Marker connections and the torso are schematic.',ha='center',fontsize=10,color='#555555')
 fig.subplots_adjust(top=.86,bottom=.12,left=.03,right=.96);fig.savefig(OUT/'figures/side_arm_setup.png',dpi=170,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','plot']);a=p.parse_args();globals()[a.stage]()
