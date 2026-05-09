"""CLI entry point: ``python -m optimizer1.traces.query <subcommand>``.

Exposes :class:`TraceQuery` as six subcommands. Default output is JSON
on stdout (machine-friendly); ``--format md`` renders a quick Markdown
table for human inspection.

The DB path is resolved by:
  1. ``--db <path>`` if given;
  2. otherwise ``$PWD/traces/index.db`` (so a proposer running inside
     its workspace can omit the flag).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .query import TraceQuery


def _resolve_db(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    default = Path.cwd() / "traces" / "index.db"
    return default


def _emit(rows: Any, fmt: str) -> None:
    if fmt == "md":
        sys.stdout.write(_render_md(rows))
        sys.stdout.write("\n")
        return
    sys.stdout.write(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
    sys.stdout.write("\n")


def _render_md(rows: Any) -> str:
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        return "_(no rows)_"
    if not all(isinstance(r, dict) for r in rows):
        return json.dumps(rows, indent=2, ensure_ascii=False, default=str)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    head = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body = []
    for row in rows:
        cells = []
        for k in keys:
            value = row.get(k)
            if isinstance(value, (dict, list)):
                cells.append(json.dumps(value, ensure_ascii=False, default=str))
            elif value is None:
                cells.append("")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="optimizer1.traces.query")
    parser.add_argument(
        "--db",
        default=None,
        help="path to traces/index.db (default: ./traces/index.db)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "md"),
        default="json",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("task-history")
    p.add_argument("task_id")

    p = sub.add_parser("persistent-failures")
    p.add_argument("--min-streak", type=int, default=3)

    p = sub.add_parser("breakthroughs")
    p.add_argument("--since-iter", type=int, default=0)

    p = sub.add_parser("regressions")
    p.add_argument("--window", type=int, default=3)

    p = sub.add_parser("file-history")
    p.add_argument("path")

    p = sub.add_parser("candidate-outcome")
    p.add_argument("iteration", type=int)
    p.add_argument("candidate_id")

    args = parser.parse_args(argv)
    db = _resolve_db(args.db)
    query = TraceQuery(db)

    if args.cmd == "task-history":
        result = query.task_history(args.task_id)
    elif args.cmd == "persistent-failures":
        result = query.persistent_failures(min_streak=args.min_streak)
    elif args.cmd == "breakthroughs":
        result = query.breakthroughs(since_iter=args.since_iter)
    elif args.cmd == "regressions":
        result = query.regressions(window=args.window)
    elif args.cmd == "file-history":
        result = query.file_history(args.path)
    elif args.cmd == "candidate-outcome":
        result = query.candidate_outcome(args.iteration, args.candidate_id)
    else:  # argparse should have rejected this
        parser.error(f"unknown subcommand: {args.cmd}")
        return 2

    _emit(result, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
