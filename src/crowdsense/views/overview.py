"""Compatibility wrapper for the legacy overview page."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from views.overview import *  # noqa: F401,F403
