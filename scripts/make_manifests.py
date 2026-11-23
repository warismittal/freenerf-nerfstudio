#!/usr/bin/env python
"""Write the hashed split manifests (single source of truth).  python scripts/make_manifests.py --data-root data/blender400
   --check  verifies manifests/HASHES.txt and the structure of every manifest without writing anything."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalkit import controls  # noqa: E402
from freenerf_ns import datasets  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data/blender400")
ap.add_argument("--manifests", default="manifests")
ap.add_argument("--scenes", nargs="+", default=datasets.SYNTHETIC_SCENES)
ap.add_argument("--dietnerf-indices", default=",".join(map(str, datasets.DIETNERF_8)),
                help="the 8 DietNeRF training indices (comma separated)")
ap.add_argument("--check", action="store_true")
a = ap.parse_args()

if a.check:
    bad = controls.verify_hashes(a.manifests)
    for p in sorted(Path(a.manifests).glob("*_n*.json")):
        bad += [f"{p.name}: {x}" for x in controls.check_manifest(json.loads(p.read_text()))]
    print("manifests OK" if not bad else "PROBLEMS:\n  " + "\n  ".join(bad))
    sys.exit(1 if bad else 0)

eight = [int(x) for x in a.dietnerf_indices.split(",")]
print("8-view training indices:", eight, "(DietNeRF: np.random.seed(0); np.random.choice(100, 8, replace=False))")
for scene, n in datasets.synthetic_manifests(a.data_root, a.manifests, a.scenes, eight):
    print(f"  wrote manifests/{scene}_n{n}.json")
print("hashes written to", Path(a.manifests) / "HASHES.txt", "- commit them, then tag `prereg-v1` at the end of Gate 1")
