"""Load the MyoSuite arm and the MS-Human-700 manipulation arm."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from mujoco_pipeline.fetch_models import MANIPULATION_XML, fetch_ms_human_700
from utils.muscle_names import MUSCLE_NAMES

REPO_DIR = Path(__file__).resolve().parents[1]

# MS-Human-700 splits latissimus into more bundles than MoBL's LAT1-3.
# Those three entries are the closest bundles, not the same muscle geometry.
MS_HUMAN_MUSCLE_MAP = {
    "CORB": ("CORB_r", "exact"),
    "DELT1": ("DELT1_r", "exact"),
    "DELT2": ("DELT2_r", "exact"),
    "DELT3": ("DELT3_r", "exact"),
    "INFSP": ("INFSP_r", "exact"),
    "LAT1": ("LD_L1_r", "approximate"),
    "LAT2": ("LD_L2_r", "approximate"),
    "LAT3": ("LD_L3_r", "approximate"),
    "PECM1": ("PECM1_r", "exact"),
    "PECM2": ("PECM2_r", "exact"),
    "PECM3": ("PECM3_r", "exact"),
    "SUBSC": ("SUBSC_r", "exact"),
    "SUPSP": ("SUPSP_r", "exact"),
    "TMAJ": ("TMAJ_r", "exact"),
    "TMIN": ("TMIN_r", "exact"),
    "ANC": ("ANC_r", "exact"),
    "BIClong": ("BIClong_r", "exact"),
    "BICshort": ("BICshort_r", "exact"),
    "BRA": ("BRA_r", "exact"),
    "BRD": ("BRD_r", "exact"),
    "ECRL": ("ECRL", "exact"),
    "PT": ("PT_r", "exact"),
    "TRIlat": ("TRIlat_r", "exact"),
    "TRIlong": ("TRIlong_r", "exact"),
    "TRImed": ("TRImed_r", "exact"),
}


@dataclass
class ArmModel:
    """One posed right arm and the 25 muscles read out for spindles."""

    name: str
    description: str
    source: str
    model: mujoco.MjModel
    data: mujoco.MjData
    end_effector_id: int
    shoulder_body_id: int
    torso_body_id: int
    elbow_body_id: int
    free_joint_names: list
    free_joint_ids: list
    locked_joint_names: list
    locked_joint_ids: list
    actuator_ids: list
    actuator_names: list
    muscle_match: list
    notes: list = field(default_factory=list)


def _id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise KeyError(f"{name} is not a {kind} in this MuJoCo model")
    return index


def _resolve_joint(model, base_name):
    for candidate in (base_name, f"{base_name}_r"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
        if index >= 0:
            return candidate, index
    raise KeyError(f"No joint named {base_name} or {base_name}_r")


def _hinge_address(model, joint_id, joint_name):
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
        raise TypeError(f"{joint_name} is not a hinge joint")
    return int(model.jnt_qposadr[joint_id])


def _load_myosuite_arm(config):
    warnings.filterwarnings("ignore", message=".*Overriding environment.*")
    import myosuite

    xml = (
        Path(myosuite.__file__).resolve().parent
        / "simhive" / "myo_sim" / "arm" / "myoarm.xml"
    )
    if not xml.is_file():
        raise FileNotFoundError(
            "MyoSuite is installed but simhive/myo_sim/arm/myoarm.xml is missing. "
            "Reinstall with: pip install 'myosuite==2.11.6' --no-deps"
        )
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    actuator_ids = []
    for name in MUSCLE_NAMES:
        actuator_ids.append(_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
    return _assemble(
        name="myosuite",
        description=(
            "MyoSuite myoarm.xml, the arm used by myoArmReachFixed-v0. "
            "It was converted from the MoBL OpenSim upper extremity."
        ),
        source=str(xml),
        model=model,
        data=data,
        end_effector=_id(model, mujoco.mjtObj.mjOBJ_SITE, "IFtip"),
        shoulder=_id(model, mujoco.mjtObj.mjOBJ_BODY, "humerus"),
        torso=_id(model, mujoco.mjtObj.mjOBJ_BODY, "thorax"),
        elbow=_id(model, mujoco.mjtObj.mjOBJ_BODY, "ulna"),
        config=config,
        actuator_ids=actuator_ids,
        actuator_names=list(MUSCLE_NAMES),
        muscle_match=["exact"] * len(MUSCLE_NAMES),
        notes=[
            "End effector is the index fingertip site IFtip, which is the "
            "site MyoSuite's arm-reach task tracks. OpenSim tracks a Handle "
            "marker in the palm, so the Cartesian point is not identical.",
            "Muscle names match the OpenSim 25-muscle list exactly.",
            "Reported length is MuJoCo actuator length (musculotendon), "
            "not OpenSim equilibrated fiber length.",
        ],
    )


def _load_corrected_myosuite_arm(config):
    from mujoco_pipeline.corrected_myoarm import build_corrected_model

    model = build_corrected_model()
    data = mujoco.MjData(model)
    actuator_ids = [
        _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in MUSCLE_NAMES
    ]
    return _assemble(
        name="myosuite_corrected",
        description=(
            "Corrected myo-sim 0.2.3 right MyoArm with MoBL-aligned routes "
            "for DELT2, PECM1, SUPSP, TRIlong, BIClong, and BICshort."
        ),
        source="myo-sim==0.2.3 plus mujoco_pipeline.corrected_myoarm patches",
        model=model,
        data=data,
        end_effector=_id(model, mujoco.mjtObj.mjOBJ_SITE, "IFtip_r"),
        shoulder=_id(model, mujoco.mjtObj.mjOBJ_BODY, "humerus_r"),
        torso=_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso"),
        elbow=_id(model, mujoco.mjtObj.mjOBJ_BODY, "ulna_r"),
        config=config,
        actuator_ids=actuator_ids,
        actuator_names=list(MUSCLE_NAMES),
        muscle_match=["exact"] * len(MUSCLE_NAMES),
        notes=[
            "Uses the MoBL fiber adapter before the unchanged Mathis spindle layer.",
            "Coordinate-dependent path points are represented by polynomially "
            "coupled MuJoCo slide joints.",
            "Validated continuous-branch spindle RMSE was 1.633 Hz overall "
            "with correlation 0.951 on the synthetic paper-envelope audit.",
        ],
    )
def _load_ms_human(config):
    if not MANIPULATION_XML.is_file():
        fetch_ms_human_700()
    xml = MANIPULATION_XML.resolve()
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    actuator_ids = []
    actuator_names = []
    muscle_match = []
    for name in MUSCLE_NAMES:
        actuator_name, match = MS_HUMAN_MUSCLE_MAP[name]
        actuator_ids.append(_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))
        actuator_names.append(actuator_name)
        muscle_match.append(match)
    return _assemble(
        name="ms_human_700",
        description=(
            "MS-Human-700 manipulation variant from MuJoCo Menagerie. "
            "The right arm uses MoBL-style joint names on a full-body skeleton."
        ),
        source=str(xml),
        model=model,
        data=data,
        end_effector=_id(model, mujoco.mjtObj.mjOBJ_SITE, "palm_site"),
        shoulder=_id(model, mujoco.mjtObj.mjOBJ_BODY, "humerus_r"),
        torso=_id(model, mujoco.mjtObj.mjOBJ_BODY, "sternum"),
        elbow=_id(model, mujoco.mjtObj.mjOBJ_BODY, "ulna_r"),
        config=config,
        actuator_ids=actuator_ids,
        actuator_names=actuator_names,
        muscle_match=muscle_match,
        notes=[
            "End effector is palm_site. OpenSim tracks a Handle marker, "
            "so the Cartesian point is not identical.",
            "LAT1, LAT2, and LAT3 are read from LD_L1_r, LD_L2_r, and "
            "LD_L3_r. Those are latissimus bundles, not the MoBL LAT muscles.",
            "Reported length is MuJoCo actuator length (musculotendon), "
            "not OpenSim equilibrated fiber length.",
        ],
    )


def _assemble(
    name, description, source, model, data, end_effector, shoulder, torso, elbow,
    config, actuator_ids, actuator_names, muscle_match, notes,
):
    free_names, free_ids = [], []
    for base_name in config["free_joints"]:
        joint_name, joint_id = _resolve_joint(model, base_name)
        _hinge_address(model, joint_id, joint_name)
        free_names.append(joint_name)
        free_ids.append(joint_id)
    locked_names, locked_ids = [], []
    for base_name in config["locked_wrist_degrees"]:
        joint_name, joint_id = _resolve_joint(model, base_name)
        _hinge_address(model, joint_id, joint_name)
        locked_names.append(joint_name)
        locked_ids.append(joint_id)
    return ArmModel(
        name=name,
        description=description,
        source=source,
        model=model,
        data=data,
        end_effector_id=end_effector,
        shoulder_body_id=shoulder,
        torso_body_id=torso,
        elbow_body_id=elbow,
        free_joint_names=free_names,
        free_joint_ids=free_ids,
        locked_joint_names=locked_names,
        locked_joint_ids=locked_ids,
        actuator_ids=actuator_ids,
        actuator_names=actuator_names,
        muscle_match=muscle_match,
        notes=notes,
    )


def load_arm(name, config):
    if name == "myosuite":
        return _load_myosuite_arm(config)
    if name == "myosuite_corrected":
        return _load_corrected_myosuite_arm(config)
    if name == "ms_human_700":
        return _load_ms_human(config)
    raise KeyError(
        f"Unknown backend {name!r}. Choose myosuite, myosuite_corrected, "
        "or ms_human_700."
    )


def joint_value(arm, joint_id):
    return float(arm.data.qpos[arm.model.jnt_qposadr[joint_id]])


def set_joint_value(arm, joint_id, value):
    arm.data.qpos[arm.model.jnt_qposadr[joint_id]] = value


def apply_joint_equalities(arm):
    """Set dependent joints from each model's polynomial shoulder rhythm.

    MuJoCo enforces these equalities in the dynamics solver. This pipeline
    poses the arm kinematically, so the polynomials are applied directly.
    """
    model, data = arm.model, arm.data
    for index in range(model.neq):
        if model.eq_type[index] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        if model.eq_active0[index] == 0:
            continue
        independent = int(model.eq_obj2id[index])
        dependent = int(model.eq_obj1id[index])
        if model.jnt_type[independent] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if model.jnt_type[dependent] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        coefficients = model.eq_data[index]
        q_value = data.qpos[model.jnt_qposadr[independent]]
        predicted = (
            coefficients[0]
            + coefficients[1] * q_value
            + coefficients[2] * q_value**2
            + coefficients[3] * q_value**3
            + coefficients[4] * q_value**4
        )
        low, high = model.jnt_range[dependent]
        data.qpos[model.jnt_qposadr[dependent]] = np.clip(predicted, low, high)


def end_effector_position(arm):
    mujoco.mj_forward(arm.model, arm.data)
    return arm.data.site_xpos[arm.end_effector_id].copy()


def body_position(arm, body_id):
    return arm.data.xpos[body_id].copy()
