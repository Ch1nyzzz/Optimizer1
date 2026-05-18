#!/usr/bin/env python3
"""Build or refresh unified evidence stores for existing optimizer runs."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from optimizer1.evidence_store import EvidenceStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Only scan raw artifacts for one iteration. Structured DB imports remain full.",
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        store = EvidenceStore(run_dir)
        store.refresh(iteration=args.iteration)
        print(_summary(store.db_path))
    return 0


def _summary(db_path: Path) -> str:
    with sqlite3.connect(db_path) as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "artifacts",
                "candidates",
                "eval_results",
                "traces",
                "trace_spans",
                "modifications",
                "tool_accesses",
                "evidence_links",
            )
        }
    parts = " ".join(f"{key}={value}" for key, value in counts.items())
    return f"{db_path}: {parts}"


if __name__ == "__main__":
    raise SystemExit(main())
