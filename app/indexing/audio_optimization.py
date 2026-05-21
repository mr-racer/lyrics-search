"""ffmpeg-based audio optimisation (M4A faststart for HTTP Range streaming).

Extracted from legacy file_processor/utils.py during Refactor 3.
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def optimize_m4a_for_streaming(file_path: str) -> bool:
    """Re-encode M4A file to move moov atom to the beginning (enables HTTP 206 Range requests).

    Uses ffmpeg with -movflags +faststart for streamable output.
    Returns True if optimization was applied, False if skipped or failed.
    """
    tmp_path = file_path + ".tmp.m4a"
    try:
        subprocess.run([
            "ffmpeg", "-i", file_path,
            "-c", "copy",
            "-movflags", "+faststart",
            tmp_path, "-y"
        ], check=True, capture_output=True, timeout=60)

        os.replace(tmp_path, file_path)
        return True
    except FileNotFoundError:
        # ffmpeg not available — skip gracefully
        return False
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
