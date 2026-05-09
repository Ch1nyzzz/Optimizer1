"""End-to-end optimizer wiring for the trace harness.

Verifies that an optimizer run produces the structured trace tree
under ``runs/<run>/traces/`` and that existing eval artefacts are
still emitted alongside it.

Heavy LLM and dataset calls are monkeypatched out.
"""

from __future__ import annotations

import json

from optimizer1.longmemeval_optimizer import (
    LongMemEvalOptimizer,
    LongMemEvalOptimizerConfig,
)
from optimizer1.schemas import CandidateResult
from optimizer1.traces import read_jsonl


def _seed_optimizer(tmp_path, *, monkeypatch):
    cfg = LongMemEvalOptimizerConfig(
        run_id="r",
        out_dir=tmp_path,
        iterations=0,
        proposer_docker_image="memo-proposer:test",
    )
    optimizer = LongMemEvalOptimizer(cfg)

    result_path = tmp_path / "candidate_results" / "seed.json"
    candidate = CandidateResult(
        candidate_id="seed",
        scaffold_name="memgpt_source",
        passrate=0.0,
        average_score=0.0,
        token_consuming=10,
        avg_token_consuming=5,
        avg_prompt_tokens=4,
        avg_completion_tokens=1,
        count=1,
        config={"top_k": 8, "extra": {"source_family": "memgpt"}},
        result_path=str(result_path),
    )

    def fake_run_seed_frontier(*args, **kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "candidate": candidate.to_dict(),
                    "tasks": [
                        {
                            "task_id": "hard-case",
                            "question": "question?",
                            "gold_answer": "gold",
                            "prediction": "pred",
                            "score": 0.0,
                            "passed": False,
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "retrieved": [
                                {
                                    "text": "doc-text",
                                    "score": 0.7,
                                    "source": "archival",
                                    "metadata": {"memory_tier": "core"},
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"candidates": [candidate.to_dict()]}

    monkeypatch.setattr(optimizer, "_load_examples", lambda: [object()])
    monkeypatch.setattr(optimizer, "_run_seed_frontier", fake_run_seed_frontier)
    return optimizer


def test_run_writes_traces_tree(tmp_path, monkeypatch):
    optimizer = _seed_optimizer(tmp_path, monkeypatch=monkeypatch)
    optimizer.run()

    # No legacy slices directory.
    assert not (tmp_path / "trace_slices").exists()

    # New traces/ tree created.
    manifest_path = tmp_path / "traces" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmark"] == "longmemeval"
    assert manifest["is_baseline_run"] is True

    # iteration 0 (scaffold) candidate jsonl exists with 1 trace.
    jsonl = tmp_path / "traces" / "spans" / "iter_000" / "seed.jsonl"
    assert jsonl.exists()
    traces = read_jsonl(jsonl)
    assert len(traces) == 1
    trace = traces[0]
    assert trace.task_id == "hard-case"
    assert trace.candidate_id == "seed"
    assert trace.iteration == 0
    assert trace.summary["passed"] is False
    assert trace.spans[0].kind == "retrieval"
    assert trace.spans[0].output["documents"][0]["score"] == 0.7

    # Existing artefacts still emitted.
    assert (tmp_path / "candidate_score_table.json").exists()
    assert (tmp_path / "retrieval_diagnostics_summary.json").exists()


def test_copy_workspace_traces_mirrors_run_traces_into_workspace(tmp_path):
    """Direct unit test for the copy helper that mirrors run-level
    ``traces/`` into the proposer workspace."""

    optimizer = LongMemEvalOptimizer(
        LongMemEvalOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            iterations=0,
            proposer_docker_image="memo-proposer:test",
        )
    )
    # Seed a manifest + diagnostic file in the run-level traces dir.
    traces_root = tmp_path / "traces"
    (traces_root / "diagnostic").mkdir(parents=True)
    (traces_root / "manifest.json").write_text('{"benchmark": "longmemeval"}', "utf-8")
    (traces_root / "diagnostic" / "iter_001.md").write_text("# header", "utf-8")

    workspace = tmp_path / "ws"
    optimizer._copy_workspace_traces(workspace / "traces")

    assert (workspace / "traces" / "manifest.json").exists()
    assert (workspace / "traces" / "diagnostic" / "iter_001.md").exists()
