"""OptiHarness command line entry point."""

from __future__ import annotations

import sys
from pathlib import Path

_LOCAL_SRC = Path(__file__).resolve().parents[1]
if str(_LOCAL_SRC) not in sys.path[:1]:
    sys.path.insert(0, str(_LOCAL_SRC))

from optimizer1.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
