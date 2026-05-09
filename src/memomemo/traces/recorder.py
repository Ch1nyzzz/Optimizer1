"""Recorder: writes one JSONL file per (iteration, candidate)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .schema import Trace, write_jsonl


class Recorder:
    def __init__(self, root: Path) -> None:
        # `root` is `runs/<run>/traces`
        self.root = root

    def spans_dir(self, *, iteration: int) -> Path:
        return self.root / "spans" / f"iter_{iteration:03d}"

    def spans_path(self, *, iteration: int, candidate_id: str) -> Path:
        return self.spans_dir(iteration=iteration) / f"{candidate_id}.jsonl"

    def write(
        self,
        *,
        iteration: int,
        candidate_id: str,
        traces: Iterable[Trace],
    ) -> Path:
        path = self.spans_path(iteration=iteration, candidate_id=candidate_id)
        write_jsonl(path, traces)
        return path
