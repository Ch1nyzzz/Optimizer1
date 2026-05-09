"""Helpers for working with vendored upstream memory systems."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / "references" / "vendor"


def vendor_path(name: str) -> Path:
    """Return the checked-out reference repository path."""

    return VENDOR_ROOT / name


@contextmanager
def prepend_sys_path(path: Path) -> Iterator[None]:
    """Temporarily prepend a path for upstream imports."""

    text = str(path)
    inserted = False
    if text not in sys.path:
        sys.path.insert(0, text)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(text)
            except ValueError:
                pass
