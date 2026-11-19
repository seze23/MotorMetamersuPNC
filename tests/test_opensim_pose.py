"""Validate the explicit pose setter against OpenSim's constraint assembler."""
import unittest
from pathlib import Path
import numpy as np
import opensim as osm
from utils.opensim_pose import CoupledPose

MODEL=Path(__file__).resolve().parents[1]/'MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim'

class PoseTests(unittest.TestCase):
    def test_exact_constraints_and_marker(self):
        model=osm.Model(str(MODEL));state=model.initSystem();pose=CoupledPose(model,MODEL)
        for values in [(20,40,25,85,-30,0,0),(-60,100,60,60,-30,10,5),(60,80,-40,90,0,-5,-10)]:
            names=['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion','pro_sup','deviation','flexion']
            requested=dict(zip(names,np.deg2rad(values)))
            pose.set_radians(state,requested);model.realizePosition(state)
            def marker():
                v=model.getMarkerSet().get('Handle').getLocationInGround(state)
                return np.array([v.get(i) for i in range(3)])
            before=marker();model.assemble(state);model.realizePosition(state)
            np.testing.assert_allclose(marker(),before,atol=1e-7,rtol=0)
            for name,value in requested.items():
                self.assertAlmostEqual(model.getCoordinateSet().get(name).getValue(state),value,places=6)

if __name__=='__main__':unittest.main()
