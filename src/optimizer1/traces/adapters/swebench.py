"""Swebench adapter.

Swebench cases produced by the mini-SWE-agent evaluator carry most of
their useful information in `metadata` rather than in the QA-style
question/gold/prediction slots — those three are empty strings because
swebench scoring is via test execution, not text comparison.

Mapping into the unified Trace summary:

  - question   = "<repo>@<base_commit>"  (or task_id if metadata absent)
  - gold       = "tests pass"             (semantic placeholder)
  - prediction = patch_path                (or "<no patch produced>")
  - passed     = task['passed']
  - score      = task['score']            (0.0 / 1.0 in practice)
  - duration_s, returncode, evaluator_returncode, and task_dir are kept
    in the summary so trace-level tools can surface them on demand.

The full multi-step agent trace (tool calls, file reads, patch
generation steps) lives in `metadata.task_dir`. M6 keeps the adapter
trace-only for now rather than emitting a dummy one-step span. Parsing
the task_dir log into real nested tool spans is a follow-up.
"""

from __future__ import annotations

from typing import Any

from ..schema import Trace


def _summary(task: dict[str, Any]) -> dict[str, Any]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    repo = metadata.get("repo") or ""
    base_commit = metadata.get("base_commit") or ""

    if repo and base_commit:
        question = f"{repo}@{base_commit[:10]}"
    elif repo:
        question = str(repo)
    else:
        question = str(task.get("question") or task.get("task_id") or "")

    patch_path = metadata.get("patch_path") or ""
    prediction = (
        f"patch: {patch_path}" if patch_path else "<no patch produced>"
    )

    return {
        "question": question,
        "gold": "tests pass",
        "prediction": prediction,
        "score": task.get("score"),
        "passed": bool(task.get("passed")),
        "repo": repo,
        "base_commit": base_commit,
        "patch_path": patch_path,
        "task_dir": metadata.get("task_dir"),
        "duration_s": metadata.get("duration_s"),
        "agent_returncode": metadata.get("returncode"),
        "evaluator_returncode": metadata.get("evaluator_returncode"),
    }


class SwebenchAdapter:
    name = "swebench"

    def build_trace(
        self,
        *,
        iteration: int,
        candidate_id: str,
        task: dict[str, Any],
    ) -> Trace:
        task_id = str(task.get("task_id") or "")
        return Trace(
            trace_id=f"iter{iteration:03d}_{candidate_id}_{task_id}",
            iteration=iteration,
            candidate_id=candidate_id,
            task_id=task_id,
            benchmark=self.name,
            summary=_summary(task),
            diff=None,
            spans=[],
        )
