import unittest
import numpy as np
from process.generate_paths.ellipses import ellipse_path

class EllipseTests(unittest.TestCase):
    def test_constant_perimeter_and_cycle(self):
        for e in [-.95,0.,.95]:
            times,xy=ellipse_path(e,'A',cycles=1)
            perimeter=np.linalg.norm(np.diff(np.vstack([xy,xy[0]]),axis=0),axis=1).sum()
            self.assertAlmostEqual(perimeter,160.,places=2)
            self.assertEqual(len(times),600)
            speed=np.linalg.norm(np.gradient(xy,1/240,axis=0),axis=1)[2:-2]
            self.assertLess(speed.std()/speed.mean(),.001)
    def test_scaling_preserves_phase_and_relative_velocity(self):
        for condition in 'ABC':
            t,large=ellipse_path(.7,condition,perimeter_cm=160.)
            small_t,small=ellipse_path(.7,condition,perimeter_cm=32.)
            np.testing.assert_array_equal(t,small_t)
            np.testing.assert_allclose(small,.2*large,atol=1e-9,rtol=0)

    def test_participant_counterclockwise_starts_left_and_moves_down(self):
        for condition in 'ABC':
            t,xy=ellipse_path(.7,condition,perimeter_cm=32.,direction='counterclockwise')
            self.assertLess(xy[0,0],0)
            self.assertAlmostEqual(xy[0,1],0)
            self.assertLess(xy[1,1],0)
            cycle=xy[:600]
            signed_area=np.sum(cycle[:,0]*np.roll(cycle[:,1],-1)-cycle[:,1]*np.roll(cycle[:,0],-1))/2
            self.assertGreater(signed_area,0)

    def test_velocity_minimum_directions(self):
        for condition in 'BC':
            t,xy=ellipse_path(0.,condition,cycles=1)
            speed=np.linalg.norm(np.gradient(xy,1/240,axis=0),axis=1)
            side=np.argmax(xy[:,0]);top=np.argmax(xy[:,1])
            if condition=='B':self.assertLess(speed[side],speed[top])
            else:self.assertGreater(speed[side],speed[top])

if __name__=='__main__':unittest.main()
