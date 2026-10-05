"""Build the validated corrected MyoArm from pinned ``myo-sim`` assets."""

import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import mujoco

MYO_SIM_VERSION = "0.2.3"


def _replace_once(text, old, new, label):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one {label} anchor in myo-sim {MYO_SIM_VERSION}")
    return text.replace(old, new)


def _patched_fragments():
    import myo_sim

    try:
        installed_version = version("myo-sim")
    except PackageNotFoundError:
        installed_version = None
    if installed_version is not None and installed_version != MYO_SIM_VERSION:
        raise RuntimeError(
            f"corrected MyoArm requires myo-sim=={MYO_SIM_VERSION}; "
            f"found {installed_version}"
        )
    root = Path(myo_sim.__file__).resolve().parent
    source = root / "models" / "arm" / "assets"
    cache = Path(tempfile.gettempdir()) / f"motor_meta_corrected_myoarm_{MYO_SIM_VERSION.replace('.', '_')}"
    cache.mkdir(parents=True, exist_ok=True)

    assets = (source / "myoarm_r_assets.xml").read_text()
    equalities = """        <joint joint1="DELT2_default_x_r" joint2="shoulder_rot_r" name="DELT2_default_x_con_r" polycoef="0.006916 -0.01981 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="DELT2_default_y_r" joint2="shoulder_rot_r" name="DELT2_default_y_con_r" polycoef="-0.06044 0.0009271 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="DELT2_default_z_r" joint2="shoulder_rot_r" name="DELT2_default_z_con_r" polycoef="0.02713 -0.001351 -0.01101 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="PECM1_unrot_r2_r" joint2="shoulder_elv_r" name="PECM1_unrot_r2_con_r" polycoef="0 0.242 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="PECM1_unrot_r3_r" joint2="shoulder_elv_r" name="PECM1_unrot_r3_con_r" polycoef="0 -0.1025 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="PECM1_default_x_r" joint2="shoulder_elv_r" name="PECM1_default_x_con_r" polycoef="0.01615 0.001878 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="PECM1_default_y_r" joint2="shoulder_elv_r" name="PECM1_default_y_con_r" polycoef="-0.04045 0.006589 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="PECM1_default_z_r" joint2="shoulder_elv_r" name="PECM1_default_z_con_r" polycoef="-0.00577 0.004584 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="PECM1_default_0_x_r" joint2="shoulder_elv_r" name="PECM1_default_0_x_con_r" polycoef="0.00958 -0.006143 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="PECM1_default_0_y_r" joint2="shoulder_elv_r" name="PECM1_default_0_y_con_r" polycoef="-0.01509 0.01483 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="PECM1_default_0_z_r" joint2="shoulder_elv_r" name="PECM1_default_0_z_con_r" polycoef="0.1266 -0.01407 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="BICshort_default_x_r" joint2="pro_sup_r" name="BICshort_default_x_con_r" polycoef="0.002723 0.005219 0 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="BICshort_default_y_r" joint2="pro_sup_r" name="BICshort_default_y_con_r" polycoef="-0.035 -0.003491 0.0005237 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
        <joint joint1="BICshort_default_z_r" joint2="pro_sup_r" name="BICshort_default_z_con_r" polycoef="-0.01357 -0.0002634 0.00127 0 0" solimp="0.9999 0.9999 0.001 0.5 2"/>
"""
    assets = _replace_once(assets, "    </equality>", equalities + "    </equality>", "equality")

    chain = (source / "myoarm_r_chain.xml").read_text()
    ground = """        <body name="PECM1_groundframe_r" pos="0 0 0">
            <inertial pos="0 0 0" mass="0.001" diaginertia="1e-8 1e-8 1e-8"/>
            <joint axis="-0.994473 0 -0.104997" name="PECM1_unrot_r3_r" pos="0 0 0" range="-0.318 0"/>
            <joint axis="0.0153 0.989299 -0.1451" name="PECM1_unrot_r2_r" pos="0 0 0" range="0 0.75"/>
            <joint name="PECM1_default_0_x_r" type="slide" axis="1 0 0" range="-0.00972 0.00958"/>
            <joint name="PECM1_default_0_y_r" type="slide" axis="0 1 0" range="-0.01509 0.03151"/>
            <joint name="PECM1_default_0_z_r" type="slide" axis="0 0 1" range="0.08244 0.1266"/>
            <site name="PECM1_default_0_r"/>
        </body>

"""
    chain = _replace_once(chain, '        <body name="clavphant_r"', ground + '        <body name="clavphant_r"', "PECM1 ground frame")
    chain = _replace_once(chain, 'pos="-0.0071293297535122493 0.0061517049319610526 0.011581990794449191"', 'pos="-0.01918 0.00127 -0.01271"', "SUPSP attachment")
    chain = _replace_once(chain, 'name="SUPSP_ellipsoid_SUPSP" pos="0.00110546 0.00820496 0.00197985"', 'name="SUPSP_ellipsoid_SUPSP" pos="0.0002 0.0077 0.0043"', "SUPSP wrap position")
    chain = _replace_once(chain, 'size="0.0202151"/>', 'size="0.02"/>', "SUPSP wrap size")
    chain = _replace_once(chain, '                                <site name="TMAJ_LAThum_sidesite_r"', '                                <site name="TMINhum_cylinder_TMIN_1_sidesite_r" pos="-0.01238 -0.01508 0.01487"/>\n                                <site name="TMAJ_LAThum_sidesite_r"', "TMIN side site")
    delt2 = """                                <body name="DELT2_default_r" pos="0 0 0">
                                    <inertial pos="0 0 0" mass="0.001" diaginertia="1e-8 1e-8 1e-8"/>
                                    <joint name="DELT2_default_x_r" type="slide" axis="1 0 0" range="0 0.03804"/>
                                    <joint name="DELT2_default_y_r" type="slide" axis="0 1 0" range="-0.0619 -0.06012"/>
                                    <joint name="DELT2_default_z_r" type="slide" axis="0 0 1" range="0.00208 0.02532"/>
                                    <site name="DELT2_default_r"/>
                                </body>
"""
    chain = _replace_once(chain, '                                <site name="DELT2_DELT2-P2_r"', delt2 + '                                <site name="DELT2_DELT2-P2_r"', "DELT2 moving point")
    chain = _replace_once(chain, '                                <site name="DELT3_DELT3-P3_r"', '                                <site name="DELT_TMAJ_LAT_PEC_CORBhh_sphere_DELT2_3_sidesite_r" pos="-2.667e-05 0.03402 -0.001008"/>\n                                <site name="DELT3_DELT3-P3_r"', "DELT2 side site")
    pecm1 = """                                <body name="PECM1_default_r" pos="0 0 0">
                                    <inertial pos="0 0 0" mass="0.001" diaginertia="1e-8 1e-8 1e-8"/>
                                    <joint name="PECM1_default_x_r" type="slide" axis="1 0 0" range="0.01615 0.02205"/>
                                    <joint name="PECM1_default_y_r" type="slide" axis="0 1 0" range="-0.04045 -0.01975"/>
                                    <joint name="PECM1_default_z_r" type="slide" axis="0 0 1" range="-0.00577 0.00863"/>
                                    <site name="PECM1_default_r"/>
                                </body>
"""
    chain = _replace_once(chain, '                                <site name="PECM1_PECM1-P2_r"', pecm1 + '                                <site name="PECM1_PECM1-P2_r"', "PECM1 moving point")
    chain = _replace_once(chain, '                                <site name="BIClong_BIClong-P4_r"', '                                <site name="BIClong_ellipsoid_BIClong_3_sidesite_r" pos="0.02241 0.00769 0.01105"/>\n                                <site name="BIClong_ellipsoid_BIClong_4_sidesite_r" pos="0.02263 -0.007683 0.01083"/>\n                                <site name="BIClong_BIClong-P4_r"', "BIClong side sites")
    bicshort = """                                        <body name="BICshort_default_r" pos="0 0 0">
                                            <inertial pos="0 0 0" mass="0.001" diaginertia="1e-8 1e-8 1e-8"/>
                                            <joint name="BICshort_default_x_r" type="slide" axis="1 0 0" range="-0.0055 0.01096"/>
                                            <joint name="BICshort_default_y_r" type="slide" axis="0 1 0" range="-0.0392 -0.0282"/>
                                            <joint name="BICshort_default_z_r" type="slide" axis="0 0 1" range="-0.01358 -0.01"/>
                                            <site name="BICshort_default_r"/>
                                        </body>
"""
    chain = _replace_once(chain, '                                        <site name="BICshort_BICshort-P7_r"', bicshort + '                                        <site name="BICshort_BICshort-P7_r"', "BICshort moving point")

    tendon = (source / "myoarm_r_tendon.xml").read_text()
    replacements = [
        ("""            <geom geom="delt2hum_cylinder" sidesite="delt2hum_cylinder_DELT2_1_sidesite_r"/>
            <site site="DELT2_DELT2-P2_r"/>
            <geom geom="Deltoid2_ellipsoid_DELT2" sidesite="Deltoid2_ellipsoid_DELT2_2_sidesite_r"/>
            <site site="DELT2_DELT2-P3_r"/>""", """            <site site="DELT2_default_r"/>
            <site site="DELT2_DELT2-P3_r"/>
            <geom geom="DELT_TMAJ_LAT_PEC_CORBhh_sphere" sidesite="DELT_TMAJ_LAT_PEC_CORBhh_sphere_DELT2_3_sidesite_r"/>""", "DELT2 tendon"),
        ('            <!--<geom geom="INFSP_and_TMIN_hum_head_ellipsoid_TMIN" sidesite="INFSP_and_TMIN_hum_head_ellipsoid_TMIN_1_sidesite_r"/>-->', '            <geom geom="TMINhum_cylinder" sidesite="TMINhum_cylinder_TMIN_1_sidesite_r"/>', "TMIN tendon"),
        ("""            <site site="PECM1_PECM1-P2_r"/>
            <geom geom="PEC1hh_ellipsoid_PECM1" sidesite="PEC1hh_ellipsoid_PECM1_2_sidesite_r"/>
            <site site="PECM1_PECM1-P3_r"/>""", """            <site site="PECM1_default_r"/>
            <geom geom="PEC12hum_cylinder" sidesite="PEC12hum_sidesite_r"/>
            <site site="PECM1_default_0_r"/>""", "PECM1 tendon"),
        ("""            <geom geom="TRIlonghh_ellipsoid_TRIlong" />
            <site site="TRIlong_TRIlong-P1b_r" />""", '            <geom geom="TRIlongglen_cylinder" sidesite="TRIlongglen_sidesite_r"/>', "TRIlong tendon"),
        ('            <site site="BIClong_BIClong-P4_r"/>', '            <geom geom="BIClong_ellipsoid_BIClong" sidesite="BIClong_ellipsoid_BIClong_3_sidesite_r"/>\n            <site site="BIClong_BIClong-P4_r"/>\n            <geom geom="BIClong_ellipsoid_BIClong" sidesite="BIClong_ellipsoid_BIClong_4_sidesite_r"/>', "BIClong tendon"),
        ("""            <geom geom="Elbow_BIC_BRD_ellipsoid_BICshort" sidesite="Elbow_BIC_BRD_ellipsoid_BICshort_5_sidesite_r"/>
            <site site="BICshort_BICshort-P6_r"/>""", '            <site site="BICshort_default_r"/>', "BICshort tendon"),
    ]
    for old, new, label in replacements:
        tendon = _replace_once(tendon, old, new, label)

    paths = {}
    for name, text in (("assets", assets), ("chain", chain), ("tendon", tendon)):
        path = cache / f"myoarm_r_{name}.xml"
        path.write_text(text)
        paths[name] = path
    return paths


def build_corrected_model():
    """Compile the corrected right MyoArm plus passive torso scaffold."""
    from myo_sim.build import compose

    paths = _patched_fragments()
    original = (
        compose.RIGHT_ARM_ASSETS_XML,
        compose.RIGHT_ARM_CHAIN_XML,
        compose.RIGHT_ARM_TENDONS_XML,
    )
    try:
        compose.RIGHT_ARM_ASSETS_XML = paths["assets"]
        compose.RIGHT_ARM_CHAIN_XML = paths["chain"]
        compose.RIGHT_ARM_TENDONS_XML = paths["tendon"]
        return compose.build_spec("myoarm_r").compile()
    finally:
        (
            compose.RIGHT_ARM_ASSETS_XML,
            compose.RIGHT_ARM_CHAIN_XML,
            compose.RIGHT_ARM_TENDONS_XML,
        ) = original
