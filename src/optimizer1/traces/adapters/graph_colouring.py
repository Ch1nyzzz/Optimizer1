"""Graph-colouring adapter.

Each per-instance task carries (colors_used, runtime_ms, known_optimal,
vertices, edges) on `metadata`, plus a `reason` field that flags compile
failures, runner timeouts, or non-zero exits. The unified Trace summary
keeps the colour count primary and runtime as a tiebreaker so renderers
and the lex comparator stay aligned with the agreed evaluation rule.
"""

from __future__ import annotations

from typing import Any

from ..schema import Span, Trace


def _summary(task: dict[str, Any]) -> dict[str, Any]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    graph_name = (
        metadata.get("graph_name")
        or task.get("task_id")
        or task.get("question")
        or ""
    )
    vertices = metadata.get("vertices")
    edges = metadata.get("edges")
    if vertices is not None and edges is not None:
        question = f"{graph_name}@{vertices}v/{edges}e"
    else:
        question = str(graph_name)

    known_optimal = metadata.get("known_optimal")
    gold = (
        f"chromatic={known_optimal}" if known_optimal is not None else "chromatic=?"
    )
    colors_used = metadata.get("colors_used")
    runtime_ms = metadata.get("runtime_ms")
    if colors_used is not None and runtime_ms is not None:
        prediction = f"colors={colors_used}, runtime_ms={float(runtime_ms):.1f}"
    else:
        prediction = str(task.get("prediction") or "<no result>")

    return {
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "score": task.get("score"),
        "passed": bool(task.get("passed")),
        "graph_name": graph_name,
        "vertices": vertices,
        "edges": edges,
        "known_optimal": known_optimal,
        "colors_used": colors_used,
        "runtime_ms": runtime_ms,
        "duration_s": metadata.get("duration_s"),
        "returncode": metadata.get("returncode"),
        "reason": metadata.get("reason"),
        "algorithm": metadata.get("algorithm"),
    }


class GraphColouringAdapter:
    name = "graph_colouring"

    def build_trace(
        self,
        *,
        iteration: int,
        candidate_id: str,
        task: dict[str, Any],
    ) -> Trace:
        task_id = str(task.get("task_id") or "")
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        agent_span = Span(
            id="s1",
            kind="agent",
            input={
                "graph_name": metadata.get("graph_name") or "",
                "vertices": metadata.get("vertices"),
                "edges": metadata.get("edges"),
                "known_optimal": metadata.get("known_optimal"),
                "algorithm": metadata.get("algorithm"),
            },
            output={
                "colors_used": metadata.get("colors_used"),
                "runtime_ms": metadata.get("runtime_ms"),
                "passed": bool(task.get("passed")),
            },
            metadata={
                "duration_s": metadata.get("duration_s"),
                "returncode": metadata.get("returncode"),
                "reason": metadata.get("reason"),
                "detail": metadata.get("detail"),
            },
        )
        return Trace(
            trace_id=f"iter{iteration:03d}_{candidate_id}_{task_id}",
            iteration=iteration,
            candidate_id=candidate_id,
            task_id=task_id,
            benchmark=self.name,
            summary=_summary(task),
            diff=None,
            spans=[agent_span],
        )
