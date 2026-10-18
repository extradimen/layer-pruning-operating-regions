import unittest
import numpy as np
from contrasts import contrasts,holm,cvar90
class Boundaries(unittest.TestCase):
 def test_directions(self):
  v={k:np.full((3,5),x) for k,x in [('k3',0),('k6',.2),('k9',.5),('branch',.1)]}
  for a,b in zip(contrasts(v),[.2,.3,-.1]):np.testing.assert_allclose(a,b)
 def test_shared_zero(self):
  self.assertTrue(all(np.all(x==0) for x in contrasts({k:np.ones((3,5)) for k in ['k3','k6','k9','branch']})))
 def test_shape(self):
  with self.assertRaises(ValueError):contrasts({k:np.ones((2,5)) for k in ['k3','k6','k9','branch']})
 def test_fractional_tail(self):
  self.assertAlmostEqual(cvar90(range(16)),(15+.6*14)/1.6)
 def test_holm(self):
  np.testing.assert_allclose(holm([.001]+[1]*8),[.009]+[1]*8)
if __name__=='__main__':unittest.main()
