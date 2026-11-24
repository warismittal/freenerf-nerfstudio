import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from sparse_nerf.data import SceneDataset
from sparse_nerf.prepare import prepare


def orbit(i, n, shift=(0, 0, 0), phase=0.0):
    theta = 2*math.pi*(i + phase)/n
    back = np.array([math.cos(theta), math.sin(theta), 0])
    right = np.array([-math.sin(theta), math.cos(theta), 0])
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, [0, 0, 1], back))
    pose[:3, 3] = np.array(shift) + 2*back
    return pose.tolist()


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "scene"
        self.source.mkdir()
        self.output = self.root / "splits" / "scene.json"

    def blender(self):
        for split, count in [("train", 100), ("test", 10)]:
            entries = []
            folder = self.source / split
            folder.mkdir()
            for i in range(count):
                Image.new("RGBA", (11, 7), (90, 40, 10, 0)).save(folder / f"r_{i}.png")
                entries.append(dict(file_path=f"./{split}/r_{i}",
                                    transform_matrix=orbit(i, count, phase=0 if split == "train" else 0.22)))
            meta = dict(camera_angle_x=0.75, frames=entries)
            (self.source / f"transforms_{split}.json").write_text(json.dumps(meta))

    def capture(self):
        entries = []
        for i in range(16):
            filename = f"frame_{i}.png"
            Image.new("RGB", (11, 7), (80, 160, 240)).save(self.source / filename)
            entries.append(dict(file_path=filename, transform_matrix=orbit(i, 16, shift=(9, 8, 7))))
        meta = dict(w=11, h=7, fl_x=6, fl_y=7, cx=5.5, cy=3.5, camera_model="OPENCV",
                    k1=0.1, k2=-0.01, p1=0.003, p2=-0.005, frames=entries[::-1])
        (self.source / "transforms.json").write_text(json.dumps(meta))
        return meta

    def test_nested_blender_and_color(self):
        self.blender()
        first = prepare(self.source, self.output, "blender")
        self.assertEqual(first, prepare(self.source, self.output, "blender"))
        splits = first["splits"]
        self.assertEqual([len(splits[k]) for k in ("sparse", "dense", "test")], [8, 100, 10])
        self.assertTrue(set(splits["sparse"]).issubset(splits["dense"]))
        self.assertTrue(set(splits["dense"]).isdisjoint(splits["test"]))
        ds = SceneDataset(self.output, "test", downscale=2)
        origins, directions, rgb = ds.frame(0)
        self.assertEqual((ds.height, ds.width), (3, 5))
        self.assertEqual(rgb.shape, (3, 5, 3))
        self.assertTrue(torch.all(rgb == 1))
        np.testing.assert_allclose(directions[1, 2], -ds.poses[0, :3, 2], atol=1e-6)
        for i in (0, 1):
            torch.testing.assert_close(origins[i, i], ds.poses[0, :3, 3])
        rows, cols, cameras = torch.tensor([0, 1]), torch.tensor([2, 1]), torch.tensor([1, 2])
        org, dirs = ds.rays(cameras, rows, cols)
        for j in range(2):
            o, d = ds.rays(int(cameras[j]), rows[j], cols[j])
            torch.testing.assert_close(o, org[j])
            torch.testing.assert_close(d, dirs[j])
        orbit_org, orbit_dirs = ds.video_rays(0, 40)
        torch.testing.assert_close(orbit_org[1, 2] + 4*orbit_dirs[1, 2], torch.zeros(3), atol=1e-6, rtol=0)

    def test_capture_geometry_distortion_and_path(self):
        metadata = self.capture()
        m = prepare(self.source, self.output, "nerfstudio", dense_count=10, test_count=4)
        ds = SceneDataset(self.output, "test", downscale=2)
        self.assertEqual([ds.frame_id(i) for i in range(4)], ["frame_2.png", "frame_6.png", "frame_10.png", "frame_14.png"])
        np.testing.assert_allclose(m["normalization"]["center"], [9, 8, 7], atol=1e-6)
        self.assertAlmostEqual(m["normalization"]["scale"], 2)
        # Reproject an off-center, inverse-distorted ray to its resized pixel center.
        org, direction = ds.rays(0, 0, 0)
        x, neg_y, _ = (ds.poses[0, :3, :3].T @ direction).numpy()
        y = -neg_y
        r2 = x*x + y*y
        k1, k2, p1, p2 = (metadata[k] for k in ("k1", "k2", "p1", "p2"))
        radial = 1 + k1*r2 + k2*r2*r2
        xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
        np.testing.assert_allclose([(6*xd + 5.5)*5/11, (7*yd + 3.5)*3/7], [0.5, 0.5], atol=1e-6)
        o0, d0 = ds.video_rays(0, 13)
        torch.testing.assert_close(o0[0, 0], org)
        torch.testing.assert_close(d0[0, 0], direction)
        o1, d1 = ds.video_rays(12, 13)
        o_last, d_last, _ = ds.frame(3)
        torch.testing.assert_close(o1, o_last)
        torch.testing.assert_close(d1, d_last)
        img = ds.image(0)
        torch.testing.assert_close(img[0, 0], torch.tensor([80, 160, 240]).float()/255)

    def test_rejects_unusable_capture(self):
        m = self.capture()
        with self.assertRaisesRegex(ValueError, "registered images"):
            prepare(self.source, self.output, "nerfstudio")
        m["camera_model"] = "OPENCV_FISHEYE"
        (self.source / "transforms.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "camera model"):
            prepare(self.source, self.output, "nerfstudio", dense_count=9, test_count=2)
        m["camera_model"] = "OPENCV"
        m["frames"][1]["file_path"] = m["frames"][0]["file_path"]
        (self.source / "transforms.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "Repeated image path"):
            prepare(self.source, self.output, "nerfstudio", dense_count=9, test_count=2)


if __name__ == "__main__":
    unittest.main()
