import unittest
from selection_depth import depth_strata, tabu_starts


QWEN = "a09a35458c702b33eeacc393d103063234e8bc28"
MISTRAL = "c170c708c41dac9275d15a8fff4eca08d52bab71"


class StartsTest(unittest.TestCase):
    def test_strata_cover_interior_without_overlap(self):
        for layers in (list(range(1, 27)), list(range(1, 31))):
            for count in (2, 4, 8):
                strata = depth_strata(layers, count)
                self.assertEqual([layer for bucket in strata for layer in bucket], layers)
                self.assertLessEqual(max(map(len, strata)) - min(map(len, strata)), 1)

    def test_all_fixed_endpoints_legal_and_reproducible(self):
        for revision, depth in ((QWEN, 28), (MISTRAL, 32)):
            for seed in (11, 23, 37):
                for k in (1, 2, 4):
                    masks = tabu_starts(revision, seed, depth, k)
                    self.assertEqual(masks, tabu_starts(revision, seed, depth, k))
                    self.assertEqual(len(masks), 8)
                    self.assertTrue(all(len(mask) == k and all(0 < layer < depth - 1 for layer in mask) for mask in masks))
                    if k == 1:
                        self.assertEqual(len({tuple(mask) for mask in masks}), 8)

    def test_reject_out_of_scope(self):
        for args in ((QWEN, 11, 28, 3), (QWEN, 11, 28, 5), (QWEN, 2, 28, 1), (QWEN, 11, 30, 1)):
            with self.assertRaises(ValueError):
                tabu_starts(*args)


if __name__ == "__main__":
    unittest.main()
