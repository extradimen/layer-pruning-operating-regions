import unittest
from starts import block_starts,branch_starts,BINS
from recipe_solver import solve_tabu
class Boundary(unittest.TestCase):
 def test_balanced(self):
  for k in [3,6,9]:
   for m in block_starts(k,11):self.assertEqual([sum(i in g for i in m) for g in BINS],[k//3]*3);self.assertFalse(set(m)&{0,31})
 def test_matched(self):
  for b,r in zip(block_starts(6,23),branch_starts(23)):self.assertEqual(tuple(i for i in r if i<32),b);self.assertEqual(tuple(i-32 for i in r if i>=32),b)
 def solve(self):return solve_tabu([float(i%32) for i in range(64)],[[0.]*64 for _ in range(64)],12,branch_starts(11),protected=(0,31,32,63),max_evaluations=80,quota_bins=[list(range(1,31)),list(range(33,63))])
 def test_budget_quota(self):
  r=self.solve();self.assertEqual(r['evaluations'],80)
  for x in r['top_two']:self.assertEqual(sum(i<32 for i in x['mask']),6);self.assertEqual(sum(i>=32 for i in x['mask']),6)
 def test_deterministic(self):self.assertEqual(self.solve()['top_two'],self.solve()['top_two'])
if __name__=='__main__':unittest.main()
