"""Check that numerical PSE ranges do not fabricate a unique crossing."""
import unittest
import numpy as np
from process.ellipse_error_analysis import crossing_bracket

class CrossingTests(unittest.TestCase):
    def test_known_crossing_is_inside_tested_bracket(self):
        estimate,low,high=crossing_bracket([.4,.5,.55,.6],[.4-.52,.5-.52,.55-.52,.6-.52])
        self.assertAlmostEqual(estimate,.52)
        self.assertEqual((low,high),(.5,.55))
    def test_multiple_or_missing_crossings_are_not_reported_as_a_pse(self):
        self.assertTrue(np.isnan(crossing_bracket([-1,0,1],[1,-1,1])[0]))
        self.assertTrue(np.isnan(crossing_bracket([-1,0,1],[1,2,3])[0]))

if __name__=='__main__':unittest.main()
