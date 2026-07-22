"""
Generate .mot files from IK joint angle solutions.

Reads ik_<direction>.npz, writes center_out_<direction>.mot
All conventions match Mathis lab: radians, inDegrees=no, 240Hz, 1152 frames.

Output: dataexp/centerout/center_out_<direction>.mot
"""

import numpy as np
import os
import glob

CENTEROUT_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout"

SAMPLE_RATE = 240
N_TOTAL     = 1152
times = np.linspace(0, N_TOTAL/SAMPLE_RATE, N_TOTAL)

# Wrist coordinates held at rest
WRIST_DEFAULTS = dict(pro_sup=0.0, deviation=0.0, flexion=0.0)

COORD_ORDER = ["elv_angle", "shoulder_elv", "shoulder_rot",
               "elbow_flexion", "pro_sup", "deviation", "flexion"]

def write_mot(filepath, traj_deg):
    lines = [
        "inDegrees=no\n", "DataType=double\n", "version=3\n",
        "OpenSimVersion=4.4-2022-10-11-798caa840\n", "endheader\n",
        "\t".join(["time"] + COORD_ORDER) + "\n",
    ]
    for i in range(N_TOTAL):
        row = [times[i]] + [np.radians(traj_deg[c][i]) for c in COORD_ORDER]
        lines.append("\t".join(f"{v:.10f}" for v in row) + "\n")
    with open(filepath, "w") as f:
        f.writelines(lines)

ik_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "ik_*.npz")))
if not ik_files:
    raise FileNotFoundError(
        f"No ik_*.npz files in {CENTEROUT_DIR} -- run ik_centerout.py first"
    )

print(f"Generating {len(ik_files)} .mot files...")
print()

for ik_path in ik_files:
    name = os.path.basename(ik_path).replace("ik_","").replace(".npz","")
    out_mot = os.path.join(CENTEROUT_DIR, f"center_out_{name}.mot")

    d = np.load(ik_path, allow_pickle=True)
    ja = d['joint_angles']   # (1152, 4) degrees: [elv, sh_elv, sh_rot, elbow]

    # EF3D training bound check
    bounds = {
        'elv_angle':    (19, 79),
        'shoulder_elv': (39, 99),
        'shoulder_rot': (-6, 54),
        'elbow_flexion':(45, 130),
    }
    jnames = ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion']

    traj = {
        'elv_angle':     ja[:, 0],
        'shoulder_elv':  ja[:, 1],
        'shoulder_rot':  ja[:, 2],
        'elbow_flexion': ja[:, 3],
        'pro_sup':       np.full(N_TOTAL, WRIST_DEFAULTS['pro_sup']),
        'deviation':     np.full(N_TOTAL, WRIST_DEFAULTS['deviation']),
        'flexion':       np.full(N_TOTAL, WRIST_DEFAULTS['flexion']),
    }

    violations = []
    for coord, (lo, hi) in bounds.items():
        vals = traj[coord]
        if vals.min() < lo or vals.max() > hi:
            violations.append(
                f"  {coord}: {vals.min():.1f}->{vals.max():.1f} outside [{lo},{hi}]"
            )

    write_mot(out_mot, traj)

    status = "yup" if not violations else f"OOB"
    print(f"  {name:<20}: elbow=[{ja[:,3].min():.1f}->{ja[:,3].max():.1f}] "
          f"sh_elv=[{ja[:,1].min():.1f}->{ja[:,1].max():.1f}] {status}")
    if violations:
        for v in violations:
            print(v)

print()
print("Done. Run extractcenterout.py")