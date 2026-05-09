"""Lock in the post_eval contract: skip_trace_slices=True must not write
the legacy trace_slices/ tree, but must still write all other artifacts."""

from __future__ import annotations

import json

from memomemo.post_eval import write_post_eval_artifacts
from memomemo.schemas import CandidateResult


def _seed(tmp_path):
    result_path = tmp_path / "candidate_results" / "c.json"
    result_path.parent.mkdir(parents=True)
    tasks = [
        {
            "task_id": f"t{i}",
            "question": "q",
            "gold_answer": "g",
            "prediction": "p",
            "score": 0.0,
            "passed": False,
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "retrieved": [{"text": "x", "score": 1.0, "source": "s", "metadata": {}}],
        }
        for i in range(4)
    ]
    candidate = CandidateResult(
        candidate_id="c",
        scaffold_name="bm25",
        passrate=0.0,
        average_score=0.0,
        token_consuming=40,
        avg_token_consuming=10,
        avg_prompt_tokens=10,
        avg_completion_tokens=1,
        count=4,
        config={},
        result_path=str(result_path),
    )
    result_path.write_text(
        json.dumps({"candidate": candidate.to_dict(), "tasks": tasks}),
        encoding="utf-8",
    )
    return candidate


def test_skip_trace_slices_does_not_create_trace_slices_dir(tmp_path):
    candidate = _seed(tmp_path)
    call_dir = tmp_path / "proposer_calls" / "iter_001"

    write_post_eval_artifacts(
        run_dir=tmp_path,
        call_dir=call_dir,
        iteration=1,
        candidates=[candidate],
        frontier_ids=set(),
        skip_trace_slices=True,
    )

    assert not (tmp_path / "trace_slices").exists()
    assert not (call_dir / "trace_slices").exists()
    # Other artifacts must still be there.
    assert (tmp_path / "candidate_score_table.json").exists()
    assert (tmp_path / "retrieval_diagnostics_summary.json").exists()
    assert (call_dir / "eval" / "eval_summary.json").exists()


def test_skip_trace_slices_default_false_preserves_legacy_writes(tmp_path):
    candidate = _seed(tmp_path)
    call_dir = tmp_path / "proposer_calls" / "iter_001"

    # Default invocation = legacy behavior.
    write_post_eval_artifacts(
        run_dir=tmp_path,
        call_dir=call_dir,
        iteration=1,
        candidates=[candidate],
        frontier_ids=set(),
    )

    assert (tmp_path / "trace_slices" / "low" / "c.json").exists()
    assert (tmp_path / "trace_slices" / "medium" / "c.json").exists()
    assert (tmp_path / "trace_slices" / "high" / "c.json").exists()
