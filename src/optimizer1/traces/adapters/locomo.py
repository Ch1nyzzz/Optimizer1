"""Locomo adapter — same shape as LongMemEval QA cases.

Locomo cases produced by the evaluator have identical fields to
LongMemEval (`task_id, question, gold_answer, prediction, score,
passed, prompt_tokens, completion_tokens, retrieved`). The only
difference is the benchmark label, so we delegate to the shared
`_build_trace_for_qa` helper.
"""

from __future__ import annotations

from typing import Any

from ..schema import Trace
from .longmemeval import _build_trace_for_qa


class LocomoAdapter:
    name = "locomo"

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
