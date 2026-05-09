"""Trace harness data model.

A `Trace` is a per-task execution record produced under a `(iteration,
candidate_id)` pair. It carries:

- `summary`: a benchmark-defined dict with the headline fields the renderer
  and the indexer rely on (`question`, `gold`, `prediction`, `score`,
  `passed`, ...). What goes in `summary` is the adapter's choice; only
  `score` and `passed` are required by downstream tooling.
- `diff`: filled by the indexer in M2 (status vs baseline). `None` until
  then.
- `spans`: nested execution steps (retrieval / generation / tool / ...).

Persisted as JSONL — one Trace per line under
`runs/<run>/traces/spans/iter_NNN/<candidate_id>.jsonl`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1.0"


@dataclass
class Span:
    id: str
    kind: str
    input: Any | None = None
    output: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    trace_id: str
    iteration: int
    candidate_id: str
    task_id: str
    benchmark: str
    summary: dict[str, Any]
    diff: dict[str, Any] | None = None
    spans: list[Span] = field(default_factory=list)


def trace_to_dict(trace: Trace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "iteration": trace.iteration,
        "candidate_id": trace.candidate_id,
        "task_id": trace.task_id,
        "benchmark": trace.benchmark,
        "summary": trace.summary,
        "diff": trace.diff,
        "spans": [asdict(span) for span in trace.spans],
    }


def trace_from_dict(payload: dict[str, Any]) -> Trace:
    spans_payload = payload.get("spans") or []
    spans = [
        Span(
            id=str(item.get("id") or ""),
            kind=str(item.get("kind") or ""),
            input=item.get("input"),
            output=item.get("output"),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in spans_payload
        if isinstance(item, dict)
    ]
    return Trace(
        trace_id=str(payload["trace_id"]),
        iteration=int(payload["iteration"]),
        candidate_id=str(payload["candidate_id"]),
        task_id=str(payload["task_id"]),
        benchmark=str(payload["benchmark"]),
        summary=dict(payload.get("summary") or {}),
        diff=payload.get("diff"),
        spans=spans,
    )


def write_jsonl(path: Path, traces: Iterable[Trace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for trace in traces:
            fh.write(json.dumps(trace_to_dict(trace), ensure_ascii=False))
            fh.write("\n")


def read_jsonl(path: Path) -> list[Trace]:
    if not path.exists():
        return []
    out: list[Trace] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(trace_from_dict(json.loads(line)))
    return out
