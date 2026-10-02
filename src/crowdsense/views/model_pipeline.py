"""Compatibility wrapper for the legacy model_pipeline page."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from views.model_pipeline import *  # noqa: F401,F403
