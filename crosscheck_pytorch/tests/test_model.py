import unittest

import torch

from sparse_nerf.model import FourierEncoding, ModelConfig, NeRFRenderer, _composite, _importance_samples


class FrequencyMaskTests(unittest.TestCase):
    def test_schedule_uses_raw_xyz_then_one_sine_or_cosine_triplet_at_a_time(self):
        xyz = torch.tensor([[0.2, 0.4, 0.6]])
        encoding = FourierEncoding(2)
        full = torch.cat((xyz, xyz.sin(), xyz.cos(), (2 * xyz).sin(), (2 * xyz).cos()), dim=-1)
        torch.testing.assert_close(encoding(xyz, 1.0), full)
        # Five xyz triplets. At 10% of the annealing period 1.5 are visible.
        mask = torch.tensor([1.0, 0.5, 1e-8, 1e-8, 1e-8]).repeat_interleave(3)
        torch.testing.assert_close(encoding(xyz, 0.1), full * mask)
        torch.testing.assert_close(encoding(xyz, 0.0)[:, :3], xyz)
        torch.testing.assert_close(FourierEncoding(0)(xyz, 0.0), xyz)


class RenderingTests(unittest.TestCase):
    def test_compositing_accounts_for_world_ray_length_and_background(self):
        z = torch.tensor([[1.0, 2.0, 3.0]])
        rgb = torch.eye(3).unsqueeze(0)
        # One camera-axis unit at density log(2) along a ray of length 2
        # absorbs 3/4 of the light, leaving the remaining 1/4 for the background.
        density = torch.tensor([[2.0, 1.0, 1.0]]).log()
        result, weights, background = _composite(rgb, density, z, torch.tensor([[2.0]]), True)
        torch.testing.assert_close(weights, torch.tensor([[0.75, 0, 0]]))
        torch.testing.assert_close(background, torch.tensor([[0.25]]))
        torch.testing.assert_close(result, torch.tensor([[1.0, 0.25, 0.25]]))

    def test_empty_pdf_and_concentrated_pdf(self):
        edges = torch.tensor([[0.0, 1.0, 2.0]])
        flat = _importance_samples(edges, torch.zeros(1, 2), 4, randomized=False)
        torch.testing.assert_close(flat, torch.tensor([[0.25, 0.75, 1.25, 1.75]]))
        concentrated = _importance_samples(edges, torch.tensor([[0.0, 1.0]]), 4, randomized=False)
        self.assertTrue(torch.all((concentrated >= 1.0) & (concentrated < 2.0)))

    def test_renderer_backprop_and_equality_once_frequencies_are_open(self):
        torch.manual_seed(22)
        model = NeRFRenderer(ModelConfig(width=12, depth=6, pos_bands=3, dir_bands=2,
                                         coarse_samples=5, fine_samples=3, anneal_fraction=0.5))
        origins = torch.zeros(3, 3)
        directions = torch.tensor([[0.1, 0.2, -1.0], [0.0, 0.0, -1.0], [0.3, -0.5, -1.0]])
        baseline = model(origins, directions, 1.0, 3.0)
        for progress in (0.5, 1.0):
            regularized = model(origins, directions, 1.0, 3.0, progress=progress, regularized=True)
            for key in baseline:
                torch.testing.assert_close(regularized[key], baseline[key], atol=0.0, rtol=0.0)
        train = model(origins, directions, 1.0, 3.0, progress=0.15, regularized=True, randomized=True)
        loss = train["rgb"].mean() + train["rgb_coarse"].mean()
        loss.backward()
        self.assertEqual(train["rgb"].shape, (3, 3))
        self.assertTrue(torch.all((train["rgb"] >= 0.0) & (train["rgb"] <= 1.0)))
        self.assertTrue(torch.all((train["depth"] >= 1.0) & (train["depth"] <= 3.0 + 1e-6)))
        for field in (model.coarse, model.fine):
            gradient = field.color.weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
