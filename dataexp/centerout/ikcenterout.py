"""
Script 2 of 3: Inverse kinematics using OpenSim's IK tool.

For each direction:
1. Converts desired wrist XYZ (lab world frame, cm, shoulder-centered)
   back to OpenSim ground frame (meters, absolute)
2. Writes a .trc file with Handle marker positions at 1152 timepoints
3. Runs osm.InverseKinematicsTool tracking only the Handle marker
4. Reads the resulting .mot file and saves joint angles as .npz

This produces biomechanically plausible joint angle trajectories
consistent with the MoBL-ARMS model constraints -- same approach
as OpenSim's GUI IK tool.

Output: dataexp/centerout/ik_<direction>.npz
  - joint_angles: (1152, 7) degrees -- all coordinates
  - times: (1152,)
"""

import opensim as osm
import numpy as np
import os
import glob
import sys
import pandas as pd
import tempfile

REPO_DIR      = "/home/sydneyez/sydneyez/ProprioceptiveIllusions"
CENTEROUT_DIR = os.path.join(REPO_DIR, "dataexp/centerout")
MODEL_PATH    = os.path.join(REPO_DIR,
    "MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model"
    "/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim")

sys.path.insert(0, REPO_DIR)

SAMPLE_RATE = 240
N_TOTAL     = 1152

# shoulder_to_world from lab code
S2W = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
W2S = S2W.T   # world frame -> OpenSim ground frame

# Rest posture -- must match generate_centerout_xyz.py
REST = {
    'elv_angle':     20.0,
    'shoulder_elv':  40.0,
    'shoulder_rot':  25.0,
    'elbow_flexion': 85.0,
}

# All 7 MoBL-ARMS coordinates in order (for reading .mot output)
COORD_NAMES = ["elv_angle", "shoulder_elv", "shoulder_rot",
               "elbow_flexion", "pro_sup", "deviation", "flexion"]

# ----------------------------------------------------------------
# 1. Get shoulder position in OpenSim ground frame at rest posture
# ----------------------------------------------------------------
print("Loading model and computing shoulder position at rest...")
model      = osm.Model(MODEL_PATH)
init_state = model.initSystem()
coord_set  = model.getCoordinateSet()
marker_set = model.getMarkerSet()

for name, val in REST.items():
    coord_set.get(name).setValue(init_state, np.radians(val))
model.realizePosition(init_state)

def get_marker_osim(name):
    p = marker_set.get(name).getLocationInGround(init_state)
    return np.array([p.get(0), p.get(1), p.get(2)])

shoulder_osim = get_marker_osim('R.Shoulder')  # meters, OpenSim ground frame
handle_osim   = get_marker_osim('Handle')

print(f"  R.Shoulder (m): {shoulder_osim}")
print(f"  Handle at rest (m): {handle_osim}")
print()

# ----------------------------------------------------------------
# 2. Helper: write .trc file for OpenSim IK
# ----------------------------------------------------------------
def write_trc(filepath, times, handle_xyz_osim):
    """
    Write a .trc file with Handle marker positions.
    handle_xyz_osim: (N, 3) in OpenSim ground frame, meters
    """
    n = len(times)
    with open(filepath, 'w') as f:
        # TRC header
        f.write("PathFileType\t4\t(X/Y/Z)\t" + filepath + "\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
                "Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{SAMPLE_RATE}\t{SAMPLE_RATE}\t{n}\t1\t"
                f"m\t{SAMPLE_RATE}\t1\t{n}\n")
        f.write("Frame#\tTime\tHandle\t\t\n")
        f.write("\t\tX1\tY1\tZ1\n")
        f.write("\n")
        for i in range(n):
            x, y, z = handle_xyz_osim[i]
            f.write(f"{i+1}\t{times[i]:.6f}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

# ----------------------------------------------------------------
# 3. Helper: read .mot file output from IK tool
# ----------------------------------------------------------------
def load_mot_joint_angles(mot_path):
    with open(mot_path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if l.strip() == 'endheader')
    df = pd.read_csv(mot_path, sep='\t', skiprows=header_end + 1)
    df.columns = [c.strip() for c in df.columns]
    return df

# ----------------------------------------------------------------
# 4. Run IK for each direction
# ----------------------------------------------------------------
xyz_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "desired_xyz_*.npz")))
if not xyz_files:
    raise FileNotFoundError(f"No desired_xyz_*.npz in {CENTEROUT_DIR}")

print(f"Found {len(xyz_files)} directions:")
for f in xyz_files:
    print(f"  {os.path.basename(f)}")
print()

for xyz_path in xyz_files:
    name     = os.path.basename(xyz_path).replace("desired_xyz_","").replace(".npz","")
    out_npz  = os.path.join(CENTEROUT_DIR, f"ik_{name}.npz")
    trc_path = os.path.join(CENTEROUT_DIR, f"_tmp_{name}.trc")
    mot_path = os.path.join(CENTEROUT_DIR, f"_tmp_{name}.mot")

    print(f"Running IK: {name}")

    d           = np.load(xyz_path, allow_pickle=True)
    xyz_world   = d['xyz'].astype(np.float64)   # (1152, 3) world frame, cm
    times       = d['times'].astype(np.float64)  # (1152,)
    center_xyz  = d['center_xyz'].astype(np.float64)

    # Convert world frame (shoulder-centered, cm) -> OpenSim ground (meters, absolute)
    # world_pos is shoulder-centered, so add shoulder_osim after rotating
    handle_xyz_osim = np.zeros((N_TOTAL, 3))
    for i in range(N_TOTAL):
        world_cm   = xyz_world[i]              # (3,) shoulder-centered, cm
        osim_rel   = W2S @ world_cm / 100.0   # (3,) OpenSim relative, meters
        handle_xyz_osim[i] = osim_rel + shoulder_osim  # absolute OpenSim

    # Sanity check at rest frame (frame 0)
    rest_osim = handle_xyz_osim[0]
    err = np.linalg.norm(rest_osim - handle_osim)
    print(f"  Handle at rest: target={rest_osim} actual={handle_osim} err={err*100:.2f}cm")

    # Write .trc file
    write_trc(trc_path, times, handle_xyz_osim)

    # Configure and run IK tool
    ik_tool = osm.InverseKinematicsTool()
    ik_tool.setModel(model)
    ik_tool.setStartTime(float(times[0]))
    ik_tool.setEndTime(float(times[-1]))
    ik_tool.setMarkerDataFileName(trc_path)
    ik_tool.setOutputMotionFileName(mot_path)
    ik_tool.set_report_errors(False)

    # Set marker weights -- only track Handle, zero weight on all others
    marker_task_set = ik_tool.getIKTaskSet()
    # Add Handle with high weight
    handle_task = osm.IKMarkerTask()
    handle_task.setName('Handle')
    handle_task.setApply(True)
    handle_task.setWeight(100.0)
    marker_task_set.cloneAndAppend(handle_task)

    print(f"  Running OpenSim IK...")
    ik_tool.run()

    # Read result
    if not os.path.exists(mot_path):
        print(f"  ERROR: IK did not produce {mot_path}")
        continue

    df = load_mot_joint_angles(mot_path)
    print(f"  IK complete. Columns: {list(df.columns[:8])}")
    print(f"  Rows: {len(df)}")

    # Extract joint angles for the 4 driven coordinates + store all 7
    joint_angles = np.zeros((N_TOTAL, 7), dtype=np.float32)
    for j, cname in enumerate(COORD_NAMES):
        if cname in df.columns:
            vals = df[cname].to_numpy()
            # IK output may be in radians (inDegrees=no) or degrees
            # Check magnitude to determine
            if np.abs(vals).max() < 10:  # radians
                vals = np.degrees(vals)
            joint_angles[:len(vals), j] = vals[:N_TOTAL]

    # Sanity check
    print(f"  Joint angle ranges:")
    for j, cname in enumerate(COORD_NAMES[:4]):
        print(f"    {cname}: {joint_angles[:,j].min():.1f} -> "
              f"{joint_angles[:,j].max():.1f} deg")

    # Check reach amplitude using lab FK
    from utils.visualize_sample import get_shoulder_elbow_wrist_loc
    labels_fk = np.zeros((N_TOTAL, 7), dtype=np.float32)
    for j in range(4):
        labels_fk[:, j+3] = joint_angles[:, j]
    _, _, wrists = get_shoulder_elbow_wrist_loc(labels_fk)
    peak    = wrists[396+120]
    center  = wrists[396]
    delta   = peak - center
    reach_xy = np.linalg.norm(delta[:2])
    z_drift  = abs(delta[2])
    print(f"  Peak reach: XY={reach_xy:.2f}cm  Z_drift={z_drift:.2f}cm")

    # Save
    np.savez(out_npz,
             joint_angles=joint_angles,   # (N, 7) degrees, all 7 coords
             times=times,
             direction_name=name)

    # Cleanup temp files
    for tmp in [trc_path, mot_path]:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"  Saved {os.path.basename(out_npz)}")
    print()

print("All done. Next: run generate_mot_centerout.py")