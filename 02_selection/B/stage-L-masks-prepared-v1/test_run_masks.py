import unittest

from run_masks import qwen_prior, mistral_prior
from selection_depth import tabu_starts


class FrozenInputTests(unittest.TestCase):
    def test_qwen_source(self):
        ids = list(range(32))
        source = {'status': 'PASS_STAGE_B6_CALIBRATION_PRIORS_FOR12_OBJECTIVE_CONFIGS_NO_FINAL',
                  'priors': {'uniform-s11': {'a_by_objective': {'NLL': [0, 1]},
                                              'H': [[0, 0], [0, 0]], 'doc_ids': ids}}}
        self.assertEqual(qwen_prior(source, 11), ([0, 1], [[0, 0], [0, 0]], ids))
        source['status'] = 'FAIL'
        with self.assertRaises(AssertionError):
            qwen_prior(source, 11)

    def test_mistral_source(self):
        source = {'status': 'PASS_96_DOCUMENTS_3_SEEDS_INDEPENDENT_BLOCK_BRANCH_PRIORS',
                  'documents_completed': 96, 'actual_calls': 18816,
                  'priors': {'11': {'block': {'documents': 32, 'a': [0, 1],
                                              'H': [[0, 0], [0, 0]]}}}}
        self.assertEqual(mistral_prior(source, 11)[:2], ([0, 1], [[0, 0], [0, 0]]))
        source['actual_calls'] = 0
        with self.assertRaises(AssertionError):
            mistral_prior(source, 11)

    def test_all_starts_legal_and_fixed(self):
        for layers in (28, 32):
            for k in (1, 2, 4):
                starts = tabu_starts('immutable-revision', 23, layers, k)
                self.assertEqual(len(starts), 8)
                self.assertTrue(all(len(mask) == k and all(0 < i < layers - 1 for i in mask)
                                    for mask in starts))


if __name__ == '__main__':
    unittest.main()
