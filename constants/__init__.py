"""Application constants package.

Use the package root for imports, e.g. `from constants import TEST_MATCH_IDS`.
"""

from .constants import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
