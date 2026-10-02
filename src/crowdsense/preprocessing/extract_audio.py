"""Compatibility wrapper for the legacy preprocessing.extract_audio module."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.extract_audio import *  # noqa: F401,F403
