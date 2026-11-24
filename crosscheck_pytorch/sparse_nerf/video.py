"""Stream a checkpoint's matched novel-camera trajectory to an H.264 MP4."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile

import torch

from .data import SceneDataset
from .run import load_run, render_image, select_device


def encoder_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def movie_metadata(output, run_id, frames, fps):
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(run_id=run_id, frames=frames, fps=fps, sha256=digest)


def movie_is_current(output, run_id, frames, fps):
    try:
        expected = json.loads(output.with_suffix(".mp4.json").read_text(encoding="utf-8"))
        return expected == movie_metadata(output, run_id, frames, fps)
    except (OSError, ValueError):
        return False


@torch.inference_mode()
def make_video(run, output, frames, fps, chunk, device):
    if min(frames, fps, chunk) <= 0:
        raise ValueError("frames, fps and chunk must be positive")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Choose an output filename ending in .mp4")
    _, renderer, meta = load_run(run, device)
    test = SceneDataset(meta["manifest_path"], "test", downscale=meta["downscale"])
    executable = encoder_executable()
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        # yuv420p needs even dimensions. At odd sizes pad just the final row/column.
        command = [executable, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{test.width}x{test.height}",
                   "-framerate", str(fps), "-i", "pipe:0", "-an", "-vcodec", "libx264",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                   "-crf", "18", "-movflags", "+faststart", str(temp_path)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for i in range(frames):
                origins, directions = test.video_rays(i, frames)
                predicted = render_image(
                    renderer, origins, directions, test.near, test.far, device=device, chunk=chunk,
                    white_background=test.white_background, regularized=meta["protocol"] == "sparse-regularized",
                )
                if not torch.isfinite(predicted).all():
                    raise ValueError(f"Nonfinite render at video frame {i}")
                pixels = (predicted.clamp(0, 1) * 255).round().byte().contiguous().numpy()
                encoder.stdin.write(pixels.tobytes())
                print(f"Frame {i+1}/{frames}", flush=True)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError(f"FFmpeg failed with exit code {encoder.returncode}")
        metadata = movie_metadata(temp_path, meta["run_id"], frames, fps)
        temp_path.replace(output)
        with NamedTemporaryFile("w", dir=output.parent, delete=False, encoding="utf-8") as sidecar:
            json.dump(metadata, sidecar, indent=2)
            sidecar.write("\n")
            sidecar_path = Path(sidecar.name)
        try:
            sidecar_path.replace(output.with_suffix(".mp4.json"))
        finally:
            sidecar_path.unlink(missing_ok=True)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="destination MP4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--chunk", type=int, default=2048, help="rays per render batch")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    make_video(args.run, args.output, args.frames, args.fps, args.chunk, select_device(args.device))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
