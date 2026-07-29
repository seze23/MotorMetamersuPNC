"""Minimal analytic forward-kinematics helper.

Extracted verbatim (numerically) from utils/visualize_sample.py in
seze23/MotorMetamersuPNC so the center-out reach-path + IK loop can run
locally without the full 55 KB module (which pulls in h5py/matplotlib).

Only the pieces used by generatereachpath.py and ikcenterout.py are kept:
rotx/roty/rotz, the arm-length constants, and get_shoulder_elbow_wrist_loc.

Convention (matches upstream): joint-angle arrays are (N, 7) in degrees with
columns [_, _, _, elv_angle, shoulder_elv, shoulder_rot, elbow_flexion].
Returned xyz locations are in the lab world frame, centimeters.
"""

import numpy as np

UPPER_ARM_LENGTH = 33  # cm
FOREARM_LENGTH = 26    # cm


def rotx(angle):
    angle = angle * np.pi / 180
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)],
    ])


def roty(angle):
    angle = angle * np.pi / 180
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])


def rotz(angle):
    angle = angle * np.pi / 180
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])


def get_shoulder_elbow_wrist_loc(
    outputs, upper_arm_length=UPPER_ARM_LENGTH, forearm_length=FOREARM_LENGTH
):
    shoulder_to_world = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    shoulder_loc = np.zeros((outputs.shape[0], 3))  # shoulder at origin

    elbow_angle_idx = 6
    elevation_idx = 3
    shoulder_elevation_idx = 4
    shoulder_rotation_idx = 5

    def shoulder_rotation(elv_angle, shoulder_elv, shoulder_rot):
        return (
            roty(elv_angle)
            .dot(rotz(shoulder_elv))
            .dot(roty(-elv_angle))
            .dot(roty(shoulder_rot))
        )

    def elbow_rotation(elbow_flexion):
        return rotx(elbow_flexion)

    elbow = np.array([0, -upper_arm_length, 0])
    hand = np.array([0, -forearm_length, 0])

    elbow_loc = np.array([
        shoulder_to_world.dot(
            shoulder_rotation(
                outputs[i, elevation_idx],
                outputs[i, shoulder_elevation_idx],
                outputs[i, shoulder_rotation_idx],
            ).dot(elbow)
        )
        for i in range(outputs.shape[0])
    ])

    wrist_loc = np.array([
        shoulder_to_world.dot(
            shoulder_rotation(
                outputs[i, elevation_idx],
                outputs[i, shoulder_elevation_idx],
                outputs[i, shoulder_rotation_idx],
            ).dot(elbow_rotation(outputs[i, elbow_angle_idx]).dot(hand) + elbow)
        )
        for i in range(outputs.shape[0])
    ])

    return shoulder_loc, elbow_loc, wrist_loc
