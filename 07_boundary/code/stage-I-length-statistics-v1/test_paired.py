import unittest
import numpy as np
from paired import transitions,paired_interval
class Boundary(unittest.TestCase):
 def test_four_cells(self):self.assertEqual(transitions([0,0,1,1],[0,1,0,1]),{'0_to_0':1,'0_to_1':1,'1_to_0':1,'1_to_1':1})
 def test_same(self):self.assertEqual(paired_interval([0,1],[0,1],np.array([[0,1],[1,0]])),{'difference_pp':0.,'nominal95CI_pp':[0.,0.]})
 def test_full_gain(self):self.assertEqual(paired_interval([0,0],[1,1],np.array([[0,1],[1,0]]))['nominal95CI_pp'],[100.,100.])
 def test_mismatch(self):
  with self.assertRaises(ValueError):transitions([0],[1,0])
if __name__=='__main__':unittest.main()
