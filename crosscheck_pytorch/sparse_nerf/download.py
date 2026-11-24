"""Fetch only the Lego and Chair experiment data from the NerfBaselines mirror."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

from PIL import Image


# Git LFS sizes and SHA-256 digests in the mirror's "Add blender mirror" commit.
REVISION = "761c60a179c5681fda6684a3bd62a56f7591e886"
BASE = f"https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/resolve/{REVISION}/blender"
ARCHIVES = {
    "lego": (179918070, "5c905f34d2b8909f1df0caaf87334995e6a02e1b1c6ca58f0bfcd162a037499f"),
    "chair": (130467387, "5cb2112f5e6fe1becf82256953575bb2e117619dcb1439a0a951f8a598b8d54a"),
}
SPLITS = {"train": 100, "test": 200}


def _verify_scene(path):
    """Check poses reference exactly the full original splits and the expected PNGs."""
    for split, count in SPLITS.items():
        meta = json.loads((path / f"transforms_{split}.json").read_text(encoding="utf-8"))
        frames = meta["frames"]
        if len(frames) != count or not 0 < float(meta["camera_angle_x"]) < 3.142:
            raise ValueError(f"{path}: expected the original {count}-view {split} split and camera FOV")
        expected = {f"{split}/r_{i}" for i in range(count)}
        references = {f["file_path"].removeprefix("./").removesuffix(".png") for f in frames}
        if references != expected:
            raise ValueError(f"{path}: unexpected {split} image paths in transforms metadata")
        for stem in expected:
            with Image.open(path / (stem + ".png")) as image:
                if image.format != "PNG" or image.size != (800, 800):
                    raise ValueError(f"{path / stem}: expected an 800 x 800 PNG")
                image.verify()


def _download_archive(scene, target):
    expected_size, expected_sha256 = ARCHIVES[scene]
    digest = hashlib.sha256()
    received = 0
    print(f"Downloading {scene} ({expected_size / 1e6:.0f} MB) from NerfBaselines...", flush=True)
    with urlopen(f"{BASE}/{scene}.zip?download=true", timeout=90) as response, target.open("xb") as dest:
        while block := response.read(1024 * 1024):
            received += len(block)
            if received > expected_size:
                raise ValueError(f"{scene}: archive exceeds its published size")
            digest.update(block)
            dest.write(block)
    if received != expected_size or digest.hexdigest() != expected_sha256:
        raise ValueError(f"{scene}: archive size/SHA-256 differs from the published mirror")


def install_scene(scene, output):
    target = output / scene
    if target.exists():
        _verify_scene(target)
        print(f"Found the 100 training and 200 test views in {target}")
        return
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{scene}-", dir=output) as folder:
        stage = Path(folder)
        archive = stage / "scene.zip"
        _download_archive(scene, archive)
        candidate = stage / scene
        candidate.mkdir()
        # Select only known image/metadata names; extra archive members are ignored.
        with zipfile.ZipFile(archive) as src:
            for split, count in SPLITS.items():
                (candidate / split).mkdir()
                names = [f"transforms_{split}.json"] + [f"{split}/r_{i}.png" for i in range(count)]
                for name in names:
                    with src.open(f"{scene}/{name}") as inp, (candidate / name).open("xb") as out:
                        shutil.copyfileobj(inp, out)
        _verify_scene(candidate)
        candidate.rename(target)
    print(f"Installed {target} (100 train, 200 test)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/blender"),
                        help="Data root, with lego and chair subdirectories (default: data/blender)")
    parser.add_argument("--scenes", nargs="+", choices=ARCHIVES, default=list(ARCHIVES),
                        help="Default: lego chair, approximately 310 MB together")
    args = parser.parse_args()
    try:
        for scene in dict.fromkeys(args.scenes):
            install_scene(scene, args.output)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Dataset download failed: {exc}\n")


if __name__ == "__main__":
    main()
