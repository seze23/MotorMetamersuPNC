from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import numpy as np,yaml,opensim as osm
from utils.opensim_pose import CoupledPose
root=Path.cwd();path=root/'MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim';model=osm.Model(str(path));state=model.initSystem();pose=CoupledPose(model,path)
rows={}
for folder,ident in [('viviani_aligned','A_p0_00'),('circle_position_grid','pos_02_10')]:
 base=root/'outputs'/folder;mfest=yaml.safe_load((base/'manifest.yaml').read_text());e=next(e for e in mfest['trajectories'] if e['id']==ident);m=np.load(base/e['muscle_data']);q=m['joint_angles'];pts=[]
 for i in [0,len(q)-600,len(q)-450,len(q)-300,len(q)-150]:
  pose.set_radians(state,dict(zip(['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion','pro_sup','deviation','flexion'],np.deg2rad(q[i]))));model.realizePosition(state);p=[]
  for name in ['C7','R.Shoulder','R.Elbow.Lateral','Handle']:
   v=model.getMarkerSet().get(name).getLocationInGround(state);p.append([v.get(j)*100 for j in range(3)])
  pts.append(p)
 rows[folder]=np.array(pts)
np.savez(root/'outputs/research_audit/arm_snapshots.npz',**rows)
