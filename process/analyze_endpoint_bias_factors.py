"""Compare endpoint-level explanations of the Wang-grid outward-bias field."""

from pathlib import Path
import json
import sys

import numpy as np
import opensim as osm
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from process.body_coordinates import opensim_to_body_cm
import process.run_poster_to_wang_posture_bridge as bridge
import process.run_wang_workspace_relocation_pilot as inference

SOURCE=ROOT/"outputs/wang_grid_full_scale_refined/ik"
RESULT=ROOT/"dataexp/poster_bias_investigation/wang_workspace_relocation_full/paired_target_summary.csv"
OUT=ROOT/"dataexp/poster_bias_investigation/endpoint_factor_analysis"
TARGETS=tuple(f"r{row:02d}_c{column:02d}" for row in range(5) for column in range(20))
NAMES=("elv_angle","shoulder_elv","shoulder_rot","elbow_flexion")


def vec3(value):return np.array([value.get(i) for i in range(3)])


def endpoint_features(joints):
    model=osm.Model(str(bridge.MODEL_OSIM));state=model.initSystem()
    model.updMuscles().get("TMIN").updGeometryPath().moveUpPathWrap(state,2)
    coordinates=[model.getCoordinateSet().get(name) for name in bridge.COORDS]
    muscles=[model.getMuscles().get(name) for name in bridge.MUSCLES]
    handle=model.getMarkerSet().get("Handle");c7=model.getMarkerSet().get("C7")
    physical=[];lengths=[]
    for row in joints:
        for coordinate,degrees in zip(coordinates,row):coordinate.setValue(state,np.deg2rad(float(degrees)))
        model.equilibrateMuscles(state);model.realizePosition(state)
        physical.append(opensim_to_body_cm(vec3(handle.getLocationInGround(state)),vec3(c7.getLocationInGround(state))))
        lengths.append([muscle.getFiberLength(state)*1000 for muscle in muscles])
    return np.asarray(physical),np.asarray(lengths)


def design(values,kind):
    values=np.asarray(values,float)
    if kind!="quadratic":return values
    columns=[values]
    for i in range(values.shape[1]):
        for j in range(i,values.shape[1]):columns.append((values[:,i]*values[:,j])[:,None])
    return np.concatenate(columns,axis=1)


def ridge_predict(x_train,y_train,x_test,kind,alpha=10.0):
    mean=np.nanmean(x_train,axis=0);x_train=np.where(np.isnan(x_train),mean,x_train);x_test=np.where(np.isnan(x_test),mean,x_test)
    scale=np.nanstd(x_train,axis=0);scale[scale<1e-12]=1
    x_train=design((x_train-mean)/scale,kind);x_test=design((x_test-mean)/scale,kind)
    x_train=np.c_[np.ones(len(x_train)),x_train];x_test=np.c_[np.ones(len(x_test)),x_test]
    penalty=np.eye(x_train.shape[1])*alpha;penalty[0,0]=0
    beta=np.linalg.solve(x_train.T@x_train+penalty,x_train.T@y_train)
    return x_test@beta


def blocked_predictions(frame,features,groups,kind="linear"):
    prediction=np.full(len(frame),np.nan)
    for held_out in np.unique(groups):
        test=groups==held_out;train=~test
        prediction[test]=ridge_predict(frame.loc[train,features].to_numpy(),frame.loc[train,"bias_cm"].to_numpy(),
            frame.loc[test,features].to_numpy(),kind)
    return prediction


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    feature_cache=OUT/"endpoint_feature_table.csv"
    if feature_cache.exists():frame=pd.read_csv(feature_cache)
    else:
        trajectories=np.stack([np.load(SOURCE/f"{target}.npz")["joint_angles"] for target in TARGETS])
        endpoint_joints=trajectories[:,bridge.PEAK,:4]
        analytic=bridge.labels_from_trajectories(trajectories)[:,inference.LATE,:3].mean(1)
        physical,lengths=endpoint_features(endpoint_joints)
        result=pd.read_csv(RESULT).set_index("target").loc[list(TARGETS)].reset_index()
        frame=pd.DataFrame({"target":TARGETS,"row":result.target.str[1:3].astype(int),"column":result.target.str[5:7].astype(int),
            "bias_cm":result.original_mean_outward_cm,"in_distribution":result.shared_in_distribution_abs_z_le_3,
            "physical_right_cm":physical[:,0],"physical_forward_cm":physical[:,1],"physical_up_cm":physical[:,2],
            "analytic_x_cm":analytic[:,0],"analytic_y_cm":analytic[:,1],"analytic_z_cm":analytic[:,2],
            "max_abs_label_z":result.original_max_abs_label_z})
        frame["physical_radius_cm"]=np.linalg.norm(physical,axis=1)
        frame["physical_azimuth_deg"]=np.rad2deg(np.arctan2(physical[:,1],physical[:,0]))
        frame["physical_elevation_deg"]=np.rad2deg(np.arctan2(physical[:,2],np.linalg.norm(physical[:,:2],axis=1)))
        for index,name in enumerate(NAMES):frame[f"joint_{name}_deg"]=endpoint_joints[:,index]
        for index,name in enumerate(bridge.MUSCLES):frame[f"muscle_{name}_mm"]=lengths[:,index]
        frame.to_csv(feature_cache,index=False)

    physical_xyz=["physical_right_cm","physical_forward_cm","physical_up_cm"]
    shoulder=["physical_radius_cm","physical_azimuth_deg","physical_elevation_deg"]
    analytic_xyz=["analytic_x_cm","analytic_y_cm","analytic_z_cm"]
    joints=[f"joint_{name}_deg" for name in NAMES]
    muscles=[f"muscle_{name}_mm" for name in bridge.MUSCLES]
    specs={"physical_right_only":(["physical_right_cm"],"linear"),"physical_forward_only":(["physical_forward_cm"],"linear"),
        "physical_up_only":(["physical_up_cm"],"linear"),"physical_xyz_linear":(physical_xyz,"linear"),
        "physical_xyz_quadratic":(physical_xyz,"quadratic"),"shoulder_spherical_linear":(shoulder,"linear"),
        "shoulder_spherical_quadratic":(shoulder,"quadratic"),"analytic_xyz_linear":(analytic_xyz,"linear"),
        "analytic_xyz_quadratic":(analytic_xyz,"quadratic"),"joint_posture_linear":(joints,"linear"),
        "joint_posture_quadratic":(joints,"quadratic"),"muscle_lengths_ridge":(muscles,"linear"),
        "physical_plus_joints":(physical_xyz+joints,"linear"),"label_distribution_only":(["max_abs_label_z"],"linear")}
    for joint in joints:specs[f"{joint}_only"]=([joint],"quadratic")
    specs.update({"shoulder_elevation_pair":(joints[:2],"quadratic"),
        "elevation_plus_rotation":([joints[0],joints[2]],"quadratic"),
        "shoulder_elv_plus_rotation":([joints[1],joints[2]],"quadratic"),
        "three_shoulder_joints":(joints[:3],"quadratic")})
    data=frame[frame.in_distribution].reset_index(drop=True);rows=[];predictions=data[["target","row","column","bias_cm"]].copy()
    baseline_rmse=float(np.sqrt(np.mean((data.bias_cm-data.bias_cm.mean())**2)))
    for name,(features,kind) in specs.items():
        for scheme,groups in (("leave_column_out",data.column.to_numpy()),("leave_row_out",data.row.to_numpy())):
            pred=blocked_predictions(data,features,groups,kind);rmse=float(np.sqrt(np.mean((data.bias_cm-pred)**2)))
            rows.append({"model":name,"scheme":scheme,"targets":len(data),"rmse_cm":rmse,
                "skill_vs_global_mean":1-rmse**2/baseline_rmse**2,"correlation":float(np.corrcoef(data.bias_cm,pred)[0,1])})
            predictions[f"{name}__{scheme}"]=pred
    scores=pd.DataFrame(rows).sort_values(["scheme","rmse_cm"]);scores.to_csv(OUT/"blocked_cv_scores.csv",index=False)
    predictions.to_csv(OUT/"blocked_cv_predictions.csv",index=False)
    temporal={"sample_rate_hz":240,"conv_layers":4,"temporal_kernel_frames":7,"temporal_stride":1,
        "receptive_field_frames":25,"receptive_field_ms":25/240*1000,"arrival_frame":int(bridge.PEAK),
        "late_first_frame":int(inference.LATE[0]),"late_last_frame":int(inference.LATE[-1]),
        "minimum_late_center_delay_after_arrival_ms":float((inference.LATE[0]-bridge.PEAK)/240*1000),
        "late_receptive_field_entirely_in_hold":bool(inference.LATE[0]-12>bridge.PEAK)}
    (OUT/"temporal_receptive_field_audit.json").write_text(json.dumps(temporal,indent=2))
    print(scores.groupby("scheme",sort=False).head(8).to_string(index=False));print(json.dumps(temporal,indent=2))


if __name__=="__main__":main()
