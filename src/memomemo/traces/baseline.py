"""Baseline trace lookup.

A `Baseline` represents the per-task reference outcome of a separately
run "unoptimized" run. Each task_id maps to exactly one
`BaselineEntry`. When the baseline run has multiple traces for the same
task_id (different candidates / iterations), we pick the **best** one
by `(passed desc, score desc)` — this captures the strongest baseline
showing for each task.

Loading order:
  1. If `index.db` exists under the baseline traces dir, query SQLite
     directly (fast, single round-trip).
  2. Otherwise, scan all `spans/iter_*/*.jsonl` files (slower but
     works before M2 has run on the baseline).

A missing baseline directory or empty index returns an empty Baseline;
callers see `lookup(task_id) is None` and classify those traces as
``no_baseline``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .diff import BaselineEntry
from .schema import read_jsonl


class Baseline:
    """task_id → best `BaselineEntry` from a baseline run."""

    def __init__(self, entries: dict[str, BaselineEntry]) -> None:
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, task_id: str) -> BaselineEntry | None:
        return self._entries.get(task_id)

    @classmethod
    def empty(cls) -> "Baseline":
        return cls({})

    @classmethod
    def from_jsonl_dir(cls, jsonl_dir: Path) -> "Baseline":
        """Load baseline entries from a single iteration's jsonl
        directory (`<traces>/spans/iter_NNN/`)."""

        if not jsonl_dir.exists():
            return cls.empty()
        entries: dict[str, BaselineEntry] = {}
        for path in sorted(jsonl_dir.glob("*.jsonl")):
            for trace in read_jsonl(path):
                summary = trace.summary or {}
                cls._maybe_replace(
                    entries,
                    BaselineEntry(
                        trace_id=trace.trace_id,
                        task_id=trace.task_id,
                        passed=bool(summary.get("passed", False)),
                        score=float(summary.get("score") or 0.0),
                    ),
                )
        return cls(entries)

    @classmethod
    def load(cls, traces_dir: Path) -> "Baseline":
        """Load from `<traces_dir>/index.db` if present, else scan jsonl."""

        traces_dir = Path(traces_dir)
        if not traces_dir.exists():
            return cls.empty()

        db_path = traces_dir / "index.db"
        if db_path.exists():
            return cls._load_from_sqlite(db_path)

        spans_root = traces_dir / "spans"
        if spans_root.exists():
            return cls._load_from_jsonl(spans_root)

        return cls.empty()

    # ------------------------------------------------------------------

    @classmethod
    def _load_from_sqlite(cls, db_path: Path) -> "Baseline":
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT trace_id, task_id, passed, score FROM traces"
            ).fetchall()
        entries: dict[str, BaselineEntry] = {}
        for row in rows:
            cls._maybe_replace(
                entries,
                BaselineEntry(
                    trace_id=str(row["trace_id"]),
                    task_id=str(row["task_id"]),
                    passed=bool(row["passed"]),
                    score=float(row["score"] if row["score"] is not None else 0.0),
                ),
            )
        return cls(entries)

    @classmethod
    def _load_from_jsonl(cls, spans_root: Path) -> "Baseline":
        entries: dict[str, BaselineEntry] = {}
        for jsonl_path in sorted(spans_root.glob("iter_*/*.jsonl")):
            for trace in read_jsonl(jsonl_path):
                summary = trace.summary or {}
                cls._maybe_replace(
                    entries,
                    BaselineEntry(
                        trace_id=trace.trace_id,
                        task_id=trace.task_id,
                        passed=bool(summary.get("passed", False)),
                        score=float(summary.get("score") or 0.0),
                    ),
                )
        return cls(entries)

    @staticmethod
    def _maybe_replace(
        entries: dict[str, BaselineEntry],
        candidate: BaselineEntry,
    ) -> None:
        existing = entries.get(candidate.task_id)
        if existing is None:
            entries[candidate.task_id] = candidate
            return
        # Prefer passed=True, then higher score.
        if candidate.passed and not existing.passed:
            entries[candidate.task_id] = candidate
            return
        if candidate.passed == existing.passed and candidate.score > existing.score:
            entries[candidate.task_id] = candidate
