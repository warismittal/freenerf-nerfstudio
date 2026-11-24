import json
from pathlib import Path
import tempfile
import unittest

from sparse_nerf.video import movie_is_current, movie_metadata


class MovieCacheTests(unittest.TestCase):
    def test_replaced_run_and_damaged_file_require_new_video(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder, "novel.mp4")
            output.write_bytes(b"first completed movie")
            self.assertFalse(movie_is_current(output, "run-a", 36, 12))
            sidecar = output.with_suffix(".mp4.json")
            sidecar.write_text(json.dumps(movie_metadata(output, "run-a", 36, 12)))
            self.assertTrue(movie_is_current(output, "run-a", 36, 12))
            self.assertFalse(movie_is_current(output, "run-b", 36, 12))
            self.assertFalse(movie_is_current(output, "run-a", 36, 24))
            output.write_bytes(b"truncated")
            self.assertFalse(movie_is_current(output, "run-a", 36, 12))
