import unittest
from selection import lowest,greedy,random_mask
class Tests(unittest.TestCase):
 def test_tie(self):self.assertEqual(lowest({4:1.,2:1.}),2)
 def test_nonfinite(self):
  with self.assertRaises(ValueError):lowest({1:float('nan')})
 def test_iterative(self):
  seen=[]
  def evaluate(mask):
   seen.append(tuple(mask));return { (1,):0,(2,):1,(3,):2,(1,2):3,(1,3):0 }[tuple(mask)]
  self.assertEqual(greedy(5,2,evaluate),[1,3]);self.assertEqual(len(seen),5)
 def test_random_legal(self):
  for n in [28,32]:
   for s in [11,23,37]:
    a=random_mask('revision',s,n,3);self.assertEqual(len(set(a)),3);self.assertTrue(all(0<i<n-1 for i in a));self.assertEqual(a,random_mask('revision',s,n,3))
if __name__=='__main__':unittest.main()
