"""Extract muscle fiber lengths and marker positions from experiment motions.

Independent motion files are processed in separate processes. Within each
motion, exact duplicate driven-coordinate poses reuse a cached equilibrium
result; this is especially effective for the long hold periods.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import opensim as osm

from experiment import load_manifest, resolve_artifact, set_artifact
from paths import EXPERIMENT_CONFIG, MODEL_PATH, MUSCLES_DIR

MUSCLE_NAMES = [
    'CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP',
    'LAT1', 'LAT2', 'LAT3', 'PECM1', 'PECM2',
    'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN',
    'ANC', 'BIClong', 'BICshort', 'BRA', 'BRD',
    'ECRL', 'PT', 'TRIlat', 'TRIlong', 'TRImed',
]
COORD_LABEL_ORDER = [
    'elv_angle', 'shoulder_elv', 'shoulder_rot',
    'elbow_flexion', 'pro_sup', 'deviation', 'flexion',
]
MUSCLE_CONFIG = EXPERIMENT_CONFIG.get("muscle_lengths", {})
COORD_ORDER = (
    COORD_LABEL_ORDER
    if MUSCLE_CONFIG.get("drive_distal_coordinates", True)
    else COORD_LABEL_ORDER[:4]
)
SHOULDER_MARKER = 'R.Shoulder'
ELBOW_MARKER = 'R.Elbow.Lateral'
WRIST_MARKER = 'Handle'
S2W = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


def _vec3_to_numpy(value):
    return np.array([value.get(0), value.get(1), value.get(2)])


def process_motion(trajectory_id, mot_path):
    """Process one motion in an isolated OpenSim model/state."""
    direction = trajectory_id
    out_npz = os.path.join(MUSCLES_DIR, f'{direction}.npz')

    model = osm.Model(MODEL_PATH)
    state = model.initSystem()
    model.equilibrateMuscles(state)
    coord_set = model.getCoordinateSet()
    muscle_set = model.getMuscles()
    marker_set = model.getMarkerSet()

    coordinates = [coord_set.get(name) for name in COORD_ORDER]
    muscles = [muscle_set.get(name) for name in MUSCLE_NAMES]
    markers = [marker_set.get(name) for name in
               (SHOULDER_MARKER, ELBOW_MARKER, WRIST_MARKER)]

    motion = osm.TimeSeriesTable(mot_path)
    column_names = list(motion.getColumnLabels())
    column_indices = {name: column_names.index(name) for name in COORD_LABEL_ORDER}
    times = np.array(motion.getIndependentColumn())
    n_frames = motion.getNumRows()

    joint_angles = np.empty((n_frames, len(COORD_LABEL_ORDER)), dtype=np.float32)
    driven_values = np.empty((n_frames, len(COORD_ORDER)), dtype=np.float64)
    for i in range(n_frames):
        row = motion.getRowAtIndex(i)
        for j, name in enumerate(COORD_LABEL_ORDER):
            joint_angles[i, j] = np.degrees(row[column_indices[name]])
        for j, name in enumerate(COORD_ORDER):
            driven_values[i, j] = row[column_indices[name]]

    fiber_lengths = np.empty((n_frames, len(MUSCLE_NAMES)), dtype=np.float32)
    shoulder_xyz = np.empty((n_frames, 3), dtype=np.float32)
    elbow_xyz = np.empty((n_frames, 3), dtype=np.float32)
    wrist_xyz = np.empty((n_frames, 3), dtype=np.float32)

    # key -> (fiber lengths in mm, shoulder, elbow, wrist in OpenSim meters)
    pose_cache = {}
    for i, values in enumerate(driven_values):
        key = values.tobytes()
        cached = pose_cache.get(key)
        if cached is None:
            for coordinate, value in zip(coordinates, values):
                coordinate.setValue(state, value)
            model.equilibrateMuscles(state)
            cached = (
                np.array([muscle.getFiberLength(state) * 1000
                          for muscle in muscles], dtype=np.float32),
                _vec3_to_numpy(markers[0].getLocationInGround(state)),
                _vec3_to_numpy(markers[1].getLocationInGround(state)),
                _vec3_to_numpy(markers[2].getLocationInGround(state)),
            )
            pose_cache[key] = cached

        fiber_lengths[i], shoulder_xyz[i], elbow_xyz[i], wrist_xyz[i] = cached

    shoulder_world = (S2W @ shoulder_xyz.T).T * 100
    elbow_world = (S2W @ elbow_xyz.T).T * 100
    wrist_world = (S2W @ wrist_xyz.T).T * 100
    wrist_centered = wrist_world - shoulder_world
    elbow_centered = elbow_world - shoulder_world

    np.savez(
        out_npz,
        times=times,
        fiber_lengths=fiber_lengths,
        joint_angles=joint_angles,
        wrist_xyz_world=wrist_centered,
        elbow_xyz_world=elbow_centered,
        coord_names=np.array(COORD_LABEL_ORDER),
        muscle_names=np.array(MUSCLE_NAMES),
    )

    bic_std = fiber_lengths[:, MUSCLE_NAMES.index('BIClong')].std()
    reach_dist = np.linalg.norm(wrist_centered[-n_frames // 2] - wrist_centered[0])
    return {
        'direction': direction,
        'frames': n_frames,
        'unique_poses': len(pose_cache),
        'bic_std_mm': float(bic_std),
        'reach_cm': float(reach_dist),
        'output': out_npz,
    }


def main():
    trajectories = load_manifest()["trajectories"]
    jobs = [(item["id"], resolve_artifact(item["motion"]))
            for item in trajectories]
    if not jobs:
        raise ValueError('The experiment manifest contains no trajectories.')

    requested_workers = int(os.environ.get('CENTEROUT_WORKERS', '4'))
    workers = max(1, min(requested_workers, len(jobs)))
    print(f'Extracting {len(jobs)} motions with {workers} worker(s)...')

    if workers == 1:
        results = [process_motion(*job) for job in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_motion, *job): job for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())

    for result in sorted(results, key=lambda item: item['direction']):
        set_artifact(result['direction'], 'muscle_data', result['output'])
        reused = result['frames'] - result['unique_poses']
        print(
            f"  {result['direction']:<20} {result['frames']} frames, "
            f"{result['unique_poses']} solved, {reused} cached | "
            f"BIClong std={result['bic_std_mm']:.3f}mm | "
            f"reach={result['reach_cm']:.1f}cm"
        )
    print(f'{len(results)} directions extracted.')


if __name__ == '__main__':
    main()
