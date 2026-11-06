"""Regression tests for the Wang experiment's anatomical coordinate frame."""

import sys
import unittest
from pathlib import Path

import numpy as np
import opensim as osm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "process"))

from body_coordinates import body_to_legacy_shoulder_cm, opensim_to_body_cm


MODEL = (
    ROOT / "MoBL-ARMSDynamicUpperLimb-latest"
    / "MoBL-ARMS Upper Extremity Model" / "Model" / "4.1"
    / "DefaultMOBL_ARMS_fixed_41.osim"
)


def marker_xyz(markers, name, state):
    value = markers.get(name).getLocationInGround(state)
    return np.array([value.get(i) for i in range(3)])


class BodyCoordinatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = osm.Model(str(MODEL))
        cls.state = cls.model.initSystem()
        coordinates = cls.model.getCoordinateSet()
        pose = {
            "elv_angle": 20.0, "shoulder_elv": 40.0,
            "shoulder_rot": 25.0, "elbow_flexion": 85.0,
            "pro_sup": 0.0, "deviation": 0.0, "flexion": 0.0,
        }
        for name, degrees in pose.items():
            coordinates.get(name).setValue(cls.state, np.radians(degrees))
        cls.model.realizePosition(cls.state)

    def test_known_right_arm_markers_have_anatomical_signs(self):
        markers = self.model.getMarkerSet()
        c7 = marker_xyz(markers, "C7", self.state)
        shoulder = opensim_to_body_cm(
            marker_xyz(markers, "R.Shoulder", self.state), c7
        )
        handle = opensim_to_body_cm(marker_xyz(markers, "Handle", self.state), c7)
        self.assertGreater(shoulder[0], 0.0)  # right of body midline
        self.assertGreater(handle[0], 0.0)    # right-hand workspace
        self.assertGreater(handle[1], 0.0)    # in front of torso

    def test_body_to_legacy_flips_only_horizontal_axes(self):
        shoulder = np.array([17.0, -3.0, -2.0])
        target = np.array([5.0, 33.0, -23.0])
        converted = body_to_legacy_shoulder_cm(target, shoulder)
        np.testing.assert_allclose(converted, [12.0, -36.0, -21.0])


if __name__ == "__main__":
    unittest.main()
