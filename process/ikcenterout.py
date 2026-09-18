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

Output: outputs/<experiment>/ik/<trajectory>.npz
  - joint_angles: (N, 7) degrees -- all coordinates
  - times: (N,)
"""

import opensim as osm
import numpy as np
import os
import sys
import pandas as pd

from experiment import load_manifest, resolve_artifact, set_artifact, validate_path_artifact
from paths import REPO_DIR, IK_DIR, MODEL_PATH

sys.path.insert(0, REPO_DIR)

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
    sample_rate = 1.0 / np.median(np.diff(times))
    with open(filepath, 'w') as f:
        # TRC header
        f.write("PathFileType\t4\t(X/Y/Z)\t" + filepath + "\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
                "Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{sample_rate}\t{sample_rate}\t{n}\t1\t"
                f"m\t{sample_rate}\t1\t{n}\n")
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
manifest = load_manifest()
trajectories = manifest["trajectories"]
print(f"Found {len(trajectories)} trajectories:")
for trajectory in trajectories:
    print(f"  {trajectory['id']}")
print()

for trajectory in trajectories:
    name = trajectory["id"]
    xyz_path = resolve_artifact(trajectory["desired_path"])
    out_npz = os.path.join(IK_DIR, f"{name}.npz")
    trc_path = os.path.join(IK_DIR, f"_tmp_{name}.trc")
    mot_path = os.path.join(IK_DIR, f"_tmp_{name}.mot")

    print(f"Running IK: {name}")

    xyz_world, times = validate_path_artifact(xyz_path)
    n_total = len(times)

    # Convert world frame (shoulder-centered, cm) -> OpenSim ground (meters, absolute)
    # world_pos is shoulder-centered, so add shoulder_osim after rotating
    handle_xyz_osim = np.zeros((n_total, 3))
    for i in range(n_total):
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
    joint_angles = np.zeros((n_total, 7), dtype=np.float32)
    for j, cname in enumerate(COORD_NAMES):
        if cname in df.columns:
            vals = df[cname].to_numpy()
            # IK output may be in radians (inDegrees=no) or degrees
            # Check magnitude to determine
            if np.abs(vals).max() < 10:  # radians
                vals = np.degrees(vals)
            joint_angles[:len(vals), j] = vals[:n_total]

    # Sanity check
    print(f"  Joint angle ranges:")
    for j, cname in enumerate(COORD_NAMES[:4]):
        print(f"    {cname}: {joint_angles[:,j].min():.1f} -> "
              f"{joint_angles[:,j].max():.1f} deg")

    # Save
    np.savez(out_npz,
             joint_angles=joint_angles,   # (N, 7) degrees, all 7 coords
             times=times,
             trajectory_id=name)
    set_artifact(name, "ik_solution", out_npz)

    # Cleanup temp files
    for tmp in [trc_path, mot_path]:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"  Saved {os.path.basename(out_npz)}")
    print()

print("All done. Next: run gencenterout.py")
