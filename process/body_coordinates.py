"""Anatomical body coordinates for the right-arm OpenSim model.

The MoBL-ARMS ground axes are X forward, Y up, and Z to the model's right.
Body coordinates therefore use (right, forward, up) = (Z, X, Y).
"""

import numpy as np


BODY_FROM_OPENSIM = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

# Kept only for converting paths consumed by the legacy IK interface.
# Its first two axes point left and backward, opposite the anatomical frame.
LEGACY_WORLD_FROM_OPENSIM = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def opensim_to_body_cm(points_ground, origin_ground):
    """Convert OpenSim ground positions in metres to C7-relative body cm."""
    points = np.asarray(points_ground, dtype=float)
    origin = np.asarray(origin_ground, dtype=float)
    return (points - origin) @ BODY_FROM_OPENSIM.T * 100.0


def body_to_legacy_shoulder_cm(points_body_cm, shoulder_body_cm):
    """Convert C7-relative body targets to the IK interface's legacy frame."""
    relative_body = np.asarray(points_body_cm, dtype=float) - np.asarray(
        shoulder_body_cm, dtype=float
    )
    # Body and legacy frames differ only by the signs of the horizontal axes.
    return relative_body * np.array([-1.0, -1.0, 1.0])
