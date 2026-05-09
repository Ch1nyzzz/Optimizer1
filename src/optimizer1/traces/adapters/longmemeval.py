"""LongMemEval adapter: case dict → Trace.

The case dict shape comes from `candidate_results/<candidate>.json` and
matches what `post_eval._case_preview` consumes:

    {
      "task_id", "question", "gold_answer", "prediction",
      "score", "passed",
      "prompt_tokens", "completion_tokens",
      "retrieved": [{"text", "score", "source", "metadata"}, ...]
    }

Locomo cases share this shape, so the same adapter logic powers both
benchmarks at construction time (registered separately under different
`name` values from `__init__.py`).
"""

from __future__ import annotations

from typing import Any

from ..schema import Span, Trace


def _build_documents(retrieved: Any) -> list[dict[str, Any]]:
    if not isinstance(retrieved, list):
        return []
    kept = [hit for hit in retrieved if isinstance(hit, dict)]
    return [
        {
            "rank": rank,
            "score": hit.get("score"),
            "source": hit.get("source"),
            "content": hit.get("text") or "",
            "metadata": dict(hit.get("metadata") or {}),
        }
        for rank, hit in enumerate(kept, start=1)
    ]


def _build_trace_for_qa(
    *,
    benchmark: str,
    iteration: int,
    candidate_id: str,
    task: dict[str, Any],
) -> Trace:
    task_id = str(task.get("task_id") or "")
    documents = _build_documents(task.get("retrieved"))

    retrieval_span = Span(
        id="s1",
        kind="retrieval",
        input={"query": task.get("question")},
        output={"documents": documents, "total_returned": len(documents)},
    )
    generation_span = Span(
        id="s2",
        kind="generation",
        input=None,
        output={"content": task.get("prediction")},
        metadata={
            "prompt_tokens": task.get("prompt_tokens"),
            "completion_tokens": task.get("completion_tokens"),
        },
    )

    summary = {
        "question": task.get("question"),
        "gold": task.get("gold_answer"),
        "prediction": task.get("prediction"),
        "score": task.get("score"),
        "passed": bool(task.get("passed")),
        "prompt_tokens": task.get("prompt_tokens"),
        "completion_tokens": task.get("completion_tokens"),
    }

    return Trace(
        trace_id=f"iter{iteration:03d}_{candidate_id}_{task_id}",
        iteration=iteration,
        candidate_id=candidate_id,
        task_id=task_id,
        benchmark=benchmark,
        summary=summary,
        diff=None,
        spans=[retrieval_span, generation_span],
    )


class LongMemEvalAdapter:
    name = "longmemeval"

    def build_trace(
        self,
        *,
        iteration: int,
        candidate_id: str,
        task: dict[str, Any],
    ) -> Trace:
        return _build_trace_for_qa(
            benchmark=self.name,
            iteration=iteration,
            candidate_id=candidate_id,
            task=task,
        )
