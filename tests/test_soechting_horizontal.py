import unittest
import numpy as np
from process.generate_paths.soechting_horizontal import trajectory,ar_from_alpha,alpha_from_ar
class HorizontalProtocol(unittest.TestCase):
 def test_area_and_signed_shape_parameter(self):
  for alpha in [-1,-.5,0,.5,1]:
   p=trajectory(alpha,'N',1)
   self.assertAlmostEqual(p['A']*p['B'],225)
   self.assertAlmostEqual(alpha_from_ar(ar_from_alpha(alpha)),alpha)
 def test_start_plane_direction_and_repeated_cycle(self):
  p=trajectory(0,'N',1)
  np.testing.assert_allclose(p['paper_xy'][0],[0,15],atol=1e-12)
  self.assertGreater(p['paper_xy'][1,0],0)
  np.testing.assert_array_equal(p['paper_xy'][:960],p['paper_xy'][960:1920])
 def test_published_speed_modulation(self):
  for exp,ratio in [(1,.68),(2,.54)]:
   p=trajectory(0,'N',exp);self.assertAlmostEqual(np.ptp(p['speed_cm_s']),0,places=10)
   for condition in ['FS','FT']:
    p=trajectory(0,condition,exp);v=p['speed_cm_s'];self.assertAlmostEqual(v.min()/v.max(),ratio,delta=.003)
 def test_analysis_window_and_no_false_normal_constant_speed(self):
  for exp,n,cycles in [(1,960,3),(2,600,5)]:
   p=trajectory(.5,'N',exp)
   self.assertEqual(p['analysis_end_index']-p['analysis_start_index'],n)
   self.assertEqual(p['analysis_end_index'],cycles*n)
   self.assertGreater(np.ptp(p['speed_cm_s']),1)
if __name__=='__main__':unittest.main()
