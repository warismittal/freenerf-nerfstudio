import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from evalkit import controls
from freenerf_ns import datasets
from tests.helpers import make_fake_blender, make_fake_nerfstudio_scene


def _setup(tmp):
    raw, small, man, sparse = (Path(tmp) / x for x in ("raw", "small", "manifests", "sparse"))
    make_fake_blender(raw, ("lego", "chair"), size=32)
    for s in ("lego", "chair"):
        datasets.downscale_scene(raw, small, s, size=16)
    datasets.synthetic_manifests(small, man, scenes=("lego", "chair"))
    return raw, small, man, sparse


def test_downscale_changes_size_and_keeps_poses():
    with tempfile.TemporaryDirectory() as t:
        raw, small, *_ = _setup(t)
        assert Image.open(small / "lego" / "train" / "r_0.png").size == (16, 16)
        a = json.loads((raw / "lego" / "transforms_train.json").read_text())
        b = json.loads((small / "lego" / "transforms_train.json").read_text())
        assert a["frames"][5]["transform_matrix"] == b["frames"][5]["transform_matrix"]


def test_manifests_structure_nested_and_hashed():
    with tempfile.TemporaryDirectory() as t:
        _, small, man, _ = _setup(t)
        idx = json.loads((man / "index.json").read_text())
        assert idx["lego"]["dense"] == 100 and idx["lego"]["test"] == 25
        m = {n: json.loads((man / f"lego_n{n}.json").read_text()) for n in (4, 8, 16, 32, 100)}
        assert m[8]["train"] == datasets.DIETNERF_8
        assert set(m[4]["train"]) <= set(m[8]["train"]) <= set(m[16]["train"]) <= set(m[32]["train"])
        assert [len(m[n]["train"]) for n in (4, 8, 16, 32, 100)] == [4, 8, 16, 32, 100]
        assert m[8]["test"] == list(range(0, 200, 8)) and len(m[8]["test"]) == 25
        assert controls.verify_hashes(man) == []
        for n in m:
            assert controls.check_manifest(m[n]) == []


def test_hash_check_detects_tampering():
    with tempfile.TemporaryDirectory() as t:
        _, _, man, _ = _setup(t)
        p = man / "lego_n8.json"
        d = json.loads(p.read_text()); d["train"][0] = 99; p.write_text(json.dumps(d))
        assert controls.verify_hashes(man) == ["lego_n8.json changed"]


def test_sparse_dataset_contains_only_training_images_C5():
    with tempfile.TemporaryDirectory() as t:
        _, small, man, sparse = _setup(t)
        d = datasets.make_sparse(small, man, "lego", 8, sparse)
        files = sorted(p.name for p in (d / "train").glob("*.png"))
        assert files == sorted(f"r_{i}.png" for i in datasets.DIETNERF_8)
        assert not (d / "test").exists()
        for split in ("train", "val", "test"):
            fr = json.loads((d / f"transforms_{split}.json").read_text())["frames"]
            assert len(fr) == 8 and all(f["file_path"].startswith("./train/") for f in fr)
        man8 = json.loads((man / "lego_n8.json").read_text())
        assert controls.check_sparse_dir(d, man8) == []
        # the check really bites
        (d / "test").mkdir(); 
        assert controls.check_sparse_dir(d, man8) != []


def test_sparse_refuses_when_manifests_were_changed():
    with tempfile.TemporaryDirectory() as t:
        _, small, man, sparse = _setup(t)
        p = man / "lego_n8.json"; d = json.loads(p.read_text()); d["train"][0] = 99; p.write_text(json.dumps(d))
        try:
            datasets.make_sparse(small, man, "lego", 8, sparse)
        except RuntimeError as e:
            assert "integrity" in str(e); return
        raise AssertionError("make_sparse accepted a tampered manifest")


def test_permuted_poses_control_C1_shuffles_pairing_but_not_images():
    with tempfile.TemporaryDirectory() as t:
        _, small, man, sparse = _setup(t)
        a = datasets.make_sparse(small, man, "lego", 8, sparse)
        b = datasets.make_sparse(small, man, "lego", 8, sparse, permute_seed=0)
        assert b.name.endswith("_perm")
        fa = json.loads((a / "transforms_train.json").read_text())["frames"]
        fb = json.loads((b / "transforms_train.json").read_text())["frames"]
        assert [f["file_path"] for f in fa] == [f["file_path"] for f in fb]            # same images in the same order
        assert all(x["transform_matrix"] != y["transform_matrix"] for x, y in zip(fa, fb))  # no pose stayed with its image
        assert sorted(map(str, (f["transform_matrix"] for f in fa))) == sorted(map(str, (f["transform_matrix"] for f in fb)))


def test_dense_dataset_uses_all_training_views():
    with tempfile.TemporaryDirectory() as t:
        _, small, man, sparse = _setup(t)
        d = datasets.make_sparse(small, man, "chair", "dense", sparse)
        assert len(list((d / "train").glob("*.png"))) == 100


def test_real_scene_conversion():
    with tempfile.TemporaryDirectory() as t:
        make_fake_nerfstudio_scene(Path(t) / "ns", n=30)
        man, out = Path(t) / "manifests", Path(t) / "b400"
        info = datasets.convert_real_scene(Path(t) / "ns", out, man, n_test=6, n_train=8, width=32)
        assert info["n_test"] == 6 and info["dense"] == 24
        meta = json.loads((out / "real" / "transforms_train.json").read_text())
        pos = np.array([np.asarray(f["transform_matrix"])[:3, 3] for f in meta["frames"]])
        assert abs(np.linalg.norm(pos, axis=1).mean() - 4.0) < 1.0               # rescaled to Blender-like radius 4
        assert Image.open(out / "real" / "train" / "r_0.png").size == (32, 24)   # aspect ratio preserved
        m8 = json.loads((man / "real_n8.json").read_text())
        assert len(m8["train"]) == 8 and set(m8["train"]) <= set(range(24)) and len(m8["test"]) == 6
        assert controls.verify_hashes(man) == []
        sparse = datasets.make_sparse(out, man, "real", 8, Path(t) / "sparse")
        assert len(list((sparse / "train").glob("*.png"))) == 8
