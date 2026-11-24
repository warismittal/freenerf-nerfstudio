import math
import unittest

import torch

from sparse_nerf.evaluate import image_scores


class EvaluationTests(unittest.TestCase):
    def test_metric_scale_and_tensor_layout(self):
        reference = torch.ones(20, 23, 3)
        predicted = torch.full_like(reference, .5)

        def check_inputs(output, target):
            self.assertEqual(tuple(output.shape), (1, 3, 20, 23))
            self.assertTrue((output == 0.).all())
            self.assertTrue((target == 1.).all())
            return torch.tensor(.125)

        psnr, distance = image_scores(predicted, reference, check_inputs, "cpu")
        self.assertAlmostEqual(psnr, -10 * math.log10(.25))
        self.assertEqual(distance, .125)

    def test_exact_and_nonfinite_predictions(self):
        target = torch.ones(5, 5, 3)

        def constant_distance(output, reference):
            return torch.tensor(0.)

        self.assertEqual(image_scores(target, target, constant_distance, "cpu")[0], 120.)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            image_scores(torch.full_like(target, float("nan")), target, constant_distance, "cpu")
