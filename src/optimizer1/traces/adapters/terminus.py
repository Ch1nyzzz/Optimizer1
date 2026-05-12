"""Terminal-Bench 2.0 (Terminus) trace adapter.

Each Terminus task result produced by ``TerminusHarborRunner`` carries its
useful information in ``metadata`` (reward, token counts, episode count,
cost, returncodes) rather than in the QA-style question/gold/prediction
slots. The full multi-step agent trajectory lives under
``metadata.trial_dir``. Like the SWE-bench adapter, this stays intentionally
shallow — one ``agent`` span per task — so renderer output is consistent
with the other benchmarks. Parsing ``trial_dir`` into nested tool spans is a
follow-up.
"""

from __future__ import annotations

from typing import Any

from ..schema import Span, Trace


def _summary(task: dict[str, Any]) -> dict[str, Any]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    terminus_task_id = metadata.get("terminus_task_id") or task.get("question") or task.get("task_id") or ""
    attempt = metadata.get("attempt")
    score = task.get("score")
    return {
        "question": str(terminus_task_id),
        "gold": "verifier reward > 0",
        "prediction": f"reward={score}" if score is not None else "<no reward>",
        "score": score,
        "passed": bool(task.get("passed")),
        "terminus_task_id": str(terminus_task_id),
        "attempt": attempt,
        "n_episodes": metadata.get("n_episodes"),
        "n_api_calls": metadata.get("n_api_calls"),
        "cost_usd": metadata.get("cost_usd"),
        "cache_tokens": metadata.get("cache_tokens"),
        "agent_returncode": metadata.get("agent_returncode"),
        "verifier_returncode": metadata.get("verifier_returncode"),
        "agent_import_path": metadata.get("agent_import_path"),
    }


class TerminusAdapter:
    name = "terminus"

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
            input={"task": metadata.get("terminus_task_id") or task.get("question") or ""},
            output={
                "passed": bool(task.get("passed")),
                "reward": task.get("score"),
            },
            metadata={
                "terminus_task_id": metadata.get("terminus_task_id"),
                "attempt": metadata.get("attempt"),
                "trial_dir": metadata.get("trial_dir"),
                "agent_import_path": metadata.get("agent_import_path"),
                "n_episodes": metadata.get("n_episodes"),
                "n_api_calls": metadata.get("n_api_calls"),
                "cost_usd": metadata.get("cost_usd"),
                "cache_tokens": metadata.get("cache_tokens"),
                "agent_returncode": metadata.get("agent_returncode"),
                "verifier_returncode": metadata.get("verifier_returncode"),
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
