"""
Script 3 of 3: Generate .mot files from OpenSim IK joint angle solutions.

Reads ik_<direction>.npz (now contains all 7 coordinates from OpenSim IK),
writes center_out_<direction>.mot

Output: dataexp/centerout/center_out_<direction>.mot
"""

import numpy as np
import os
import glob

CENTEROUT_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout"

SAMPLE_RATE = 240
N_TOTAL     = 1152
times = np.linspace(0, N_TOTAL/SAMPLE_RATE, N_TOTAL)

# All 7 coordinates in MoBL-ARMS order
COORD_ORDER = ["elv_angle", "shoulder_elv", "shoulder_rot",
               "elbow_flexion", "pro_sup", "deviation", "flexion"]

def write_mot(filepath, joint_angles_deg):
    """
    joint_angles_deg: (N, 7) degrees, column order matches COORD_ORDER
    """
    lines = [
        "inDegrees=no\n", "DataType=double\n", "version=3\n",
        "OpenSimVersion=4.4-2022-10-11-798caa840\n", "endheader\n",
        "\t".join(["time"] + COORD_ORDER) + "\n",
    ]
    for i in range(N_TOTAL):
        row = [times[i]] + [np.radians(joint_angles_deg[i, j])
                            for j in range(7)]
        lines.append("\t".join(f"{v:.10f}" for v in row) + "\n")
    with open(filepath, "w") as f:
        f.writelines(lines)

ik_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "ik_*.npz")))
if not ik_files:
    raise FileNotFoundError(f"No ik_*.npz in {CENTEROUT_DIR}")

print(f"Generating {len(ik_files)} .mot files...")
print()

for ik_path in ik_files:
    name    = os.path.basename(ik_path).replace("ik_","").replace(".npz","")
    out_mot = os.path.join(CENTEROUT_DIR, f"center_out_{name}.mot")

    d  = np.load(ik_path, allow_pickle=True)
    ja = d['joint_angles']   # (N, 7) degrees

    # EF3D training bound check on the 4 driven coordinates
    bounds = {
        'elv_angle':    (0, 19, 79),
        'shoulder_elv': (1, 39, 99),
        'shoulder_rot': (2, -6, 54),
        'elbow_flexion':(3, 45, 130),
    }
    violations = []
    for coord, (j, lo, hi) in bounds.items():
        mn, mx = ja[:, j].min(), ja[:, j].max()
        if mn < lo or mx > hi:
            violations.append(f"  {coord}: {mn:.1f}->{mx:.1f} outside [{lo},{hi}]")

    write_mot(out_mot, ja)

    status = "✓" if not violations else f"⚠ {len(violations)} bound violation(s)"
    print(f"  {name:<20}: elbow=[{ja[:,3].min():.1f}->{ja[:,3].max():.1f}] "
          f"sh_elv=[{ja[:,1].min():.1f}->{ja[:,1].max():.1f}] {status}")
    for v in violations:
        print(v)

print()
print("Done. Run extractdata_centerout.py -> computefrcenterout.py -> centeroutinference.py")