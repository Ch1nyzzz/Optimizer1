"""Inject the local src/ tree before any system-installed optimizer1 package."""

from __future__ import annotations

import sys
from pathlib import Path

_LOCAL_SRC = Path(__file__).resolve().parent / "src"
if str(_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SRC))
