import unittest
import numpy as np
from process.generate_paths.circle_positions import make_trial
from utils.trajectory_shape import fit_ellipse,extent_shape

class CirclePositionTests(unittest.TestCase):
    def test_common_start_and_circle_window(self):
        common=np.array([2.,-40.,0.]);centers=[np.array([0.,-35.,0.]),np.array([5.,-30.,0.])]
        for center in centers:
            t=make_trial(common,center,3.)
            np.testing.assert_array_equal(t['xyz'][0],common)
            lo,hi=t['analysis_start_index'],t['analysis_end_index']
            self.assertEqual(hi-lo,600)
            xy=(t['xyz'][lo:hi]-center)[:,[0,2]]*[-1,1]
            np.testing.assert_allclose(np.linalg.norm(xy,axis=1),3.,atol=1e-12)
            self.assertLess(xy[0,0],0);self.assertLess(xy[1,1],0)
            self.assertGreater(len(t['xyz'])-hi,25)
            np.testing.assert_allclose(t['xyz'][:,1][lo:hi],center[1])
    def test_rotated_ellipse_is_not_hidden_by_equal_axis_extents(self):
        phase=np.linspace(0,2*np.pi,600,endpoint=False)**1.05
        xy=np.column_stack([4*np.cos(phase),2*np.sin(phase)])
        theta=np.pi/4;rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        xy=xy@rotation.T
        label,_=extent_shape(xy)
        self.assertEqual(label,'circle')  # An extent-only rule can miss tilt.
        fitted=fit_ellipse(xy)
        self.assertAlmostEqual(fitted['axis_ratio'],2.,places=4)
        self.assertAlmostEqual(abs(fitted['angle_deg']),45.,places=3)

if __name__=='__main__':unittest.main()
