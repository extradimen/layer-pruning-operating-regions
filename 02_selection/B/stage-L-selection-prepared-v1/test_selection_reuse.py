import hashlib
import unittest

from selection_reuse import validated_prefix, bi_masks, random_masks


class SelectionReuseTests(unittest.TestCase):
    def test_validated_prefix(self):
        evaluations = []
        for step, chosen, deleted in [(0, 1, []), (1, 2, [1]), (2, 3, [1, 2])]:
            for candidate in set(range(1, 5)) - set(deleted):
                evaluations.append({'step': step, 'candidate': candidate,
                                    'deleted': sorted(deleted + [candidate]), 'score': float(candidate)})
            evaluations.append({'step': step, 'chosen': chosen, 'deleted': sorted(deleted + [chosen])})
        self.assertEqual(validated_prefix(evaluations, 6, [1, 2, 3]), [1, 2, 3])
        with self.assertRaises(ValueError):
            validated_prefix(evaluations[:-2], 6, [1, 2, 3])

    def test_bi_validation(self):
        self.assertEqual(bi_masks({'1': 0.2, '2': 0.1, '3': 0.3, '4': 0.4}, 6, [1, 2, 3])[2], [1, 2])
        with self.assertRaises(ValueError):
            bi_masks({'1': 0.2, '2': float('nan'), '3': 0.3, '4': 0.4}, 6, [1, 2, 3])

    def test_random_validation(self):
        ranking = sorted(range(1, 5), key=lambda layer: hashlib.sha256(
            f'selection-supplement-v1|rev|11|{layer}'.encode()).digest())
        masks = random_masks('rev', 11, 6, sorted(ranking[:3]))
        self.assertEqual(masks[4], [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            random_masks('rev', 11, 6, [])


if __name__ == '__main__':
    unittest.main()
