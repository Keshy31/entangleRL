"""
Concatenate the four rendered act clips into the final teaser.

Run with the manim venv python after rendering all acts at the same quality:

    .manim-env\\Scripts\\python.exe video/stitch.py 1080p60
    .manim-env\\Scripts\\python.exe video/stitch.py 480p15   (dev preview)

Uses the ffmpeg binary bundled with imageio-ffmpeg (stream copy, no re-encode).
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "video" / "media" / "videos"

ACTS = [
    ("act1_problem", "Act1Problem"),
    ("act2_trap", "Act2Trap"),
    ("act3_fix", "Act3Fix"),
    ("act4_payoff", "Act4Payoff"),
]


def main(quality: str = "1080p60") -> None:
    files = []
    for module, scene in ACTS:
        f = MEDIA / module / quality / f"{scene}.mp4"
        if not f.exists():
            sys.exit(f"Missing {f} -- render it first (manim render -q* ...)")
        files.append(f)

    concat_list = REPO / "video" / "media" / f"concat_{quality}.txt"
    concat_list.write_text("\n".join(f"file '{f.as_posix()}'" for f in files))

    out = REPO / "video" / f"teaser_{quality}.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c", "copy", str(out)],
        check=True,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1080p60")
