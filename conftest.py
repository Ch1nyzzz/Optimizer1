"""Inject the local src/ tree before any system-installed optimizer1 package."""

from __future__ import annotations

import sys
from pathlib import Path

_LOCAL_SRC = Path(__file__).resolve().parent / "src"
_LOCAL_SRC_TEXT = str(_LOCAL_SRC)
if _LOCAL_SRC_TEXT in sys.path:
    sys.path.remove(_LOCAL_SRC_TEXT)
sys.path.insert(0, _LOCAL_SRC_TEXT)
