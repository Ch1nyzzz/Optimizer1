"""Unique console entry point for the Optimizer1 checkout."""

from __future__ import annotations

import sys
from pathlib import Path

_LOCAL_SRC = Path(__file__).resolve().parent
if str(_LOCAL_SRC) not in sys.path[:1]:
    sys.path.insert(0, str(_LOCAL_SRC))

from memomemo.cli import main  # noqa: E402

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
