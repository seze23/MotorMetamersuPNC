"""Regression tests for consistent OpenSim coordinate replay."""

import unittest
from pathlib import Path
import sys

import numpy as np
import opensim as osm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "process"))

from extractcenterout import COORD_LABEL_ORDER, COORD_ORDER


MODEL = (
    ROOT
    / "MoBL-ARMSDynamicUpperLimb-latest"
    / "MoBL-ARMS Upper Extremity Model"
    / "Model"
    / "4.1"
    / "DefaultMOBL_ARMS_fixed_41.osim"
)


def marker_xyz(marker, state):
    value = marker.getLocationInGround(state)
    return np.array([value.get(i) for i in range(3)])


class FullCoordinateReplayTest(unittest.TestCase):
    def test_extraction_replays_every_motion_coordinate(self):
        self.assertEqual(COORD_ORDER, COORD_LABEL_ORDER)

    def test_distal_state_materially_changes_handle_and_is_preserved(self):
        model = osm.Model(str(MODEL))
        state = model.initSystem()
        coordinates = model.getCoordinateSet()
        handle = model.getMarkerSet().get("Handle")

        pose_degrees = {
            "elv_angle": 20.0,
            "shoulder_elv": 40.0,
            "shoulder_rot": 25.0,
            "elbow_flexion": 85.0,
            "pro_sup": -30.0,
            "deviation": 0.0,
            "flexion": 0.0,
        }
        for name, degrees in pose_degrees.items():
            coordinates.get(name).setValue(state, np.radians(degrees))
        model.realizePosition(state)
        complete_state_handle = marker_xyz(handle, state)

        coordinates.get("pro_sup").setValue(state, 0.0)
        model.realizePosition(state)
        truncated_state_handle = marker_xyz(handle, state)

        shift_cm = np.linalg.norm(complete_state_handle - truncated_state_handle) * 100
        self.assertGreater(shift_cm, 3.0)

        replayed = model.initSystem()
        replay_coordinates = model.getCoordinateSet()
        for name in COORD_ORDER:
            replay_coordinates.get(name).setValue(
                replayed, np.radians(pose_degrees[name])
            )
        model.realizePosition(replayed)
        replayed_handle = marker_xyz(model.getMarkerSet().get("Handle"), replayed)
        np.testing.assert_allclose(replayed_handle, complete_state_handle, atol=1e-10)

    def test_reference_origin_is_fixed_not_framewise(self):
        shoulder_world = np.array([[1.0, 2.0, 3.0], [1.4, 2.2, 3.1]])
        handle_world = np.array([[11.0, 7.0, 3.0], [12.0, 8.0, 3.0]])
        fixed = handle_world - shoulder_world[0]
        moving = handle_world - shoulder_world
        np.testing.assert_allclose(fixed[1], [11.0, 6.0, 0.0])
        self.assertFalse(np.allclose(fixed[1], moving[1]))


if __name__ == "__main__":
    unittest.main()
