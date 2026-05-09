"""Renderer — every status type, baseline mode, truncation edges."""

from __future__ import annotations

from pathlib import Path

from optimizer1.traces import (
    Baseline,
    BaselineEntry,
    Indexer,
    RenderConfig,
    Renderer,
    Span,
    Trace,
    write_jsonl,
)


def _trace(
    *,
    iteration: int,
    candidate_id: str,
    task_id: str,
    passed: bool,
    score: float,
    question: str = "How much cashback?",
    gold: str = "$0.75",
    prediction: str = "unknown",
    docs: list[dict] | None = None,
    benchmark: str = "longmemeval",
) -> Trace:
    if docs is None:
        docs = [
            {
                "rank": 1,
                "score": 0.82,
                "source": "archival",
                "content": "session 1 about saveMart purchase",
                "metadata": {},
            },
            {
                "rank": 2,
                "score": 0.78,
                "source": "archival",
                "content": "session 2 unrelated",
                "metadata": {},
            },
        ]
    return Trace(
        trace_id=f"iter{iteration:03d}_{candidate_id}_{task_id}",
        iteration=iteration,
        candidate_id=candidate_id,
        task_id=task_id,
        benchmark=benchmark,
        summary={
            "question": question,
            "gold": gold,
            "prediction": prediction,
            "score": score,
            "passed": passed,
            "prompt_tokens": 1669,
            "completion_tokens": 24,
        },
        diff=None,
        spans=[
            Span(
                id="s1",
                kind="retrieval",
                input={"query": question},
                output={"documents": docs, "total_returned": len(docs)},
            ),
            Span(
                id="s2",
                kind="generation",
                output={"content": prediction},
            ),
        ],
    )


def _seed_traces(traces_dir: Path, *, iteration: int, candidate_id: str, traces):
    path = traces_dir / "spans" / f"iter_{iteration:03d}" / f"{candidate_id}.jsonl"
    write_jsonl(path, traces)
    return path


def _baseline_with(*entries: BaselineEntry) -> Baseline:
    return Baseline({entry.task_id: entry for entry in entries})


# ---- diff-mode rendering ------------------------------------------


def test_render_diff_run_covers_all_statuses(tmp_path):
    traces_dir = tmp_path / "traces"
    baseline = _baseline_with(
        BaselineEntry(trace_id="bl_t1", task_id="t1", passed=True, score=0.9),
        BaselineEntry(trace_id="bl_t2", task_id="t2", passed=False, score=0.1),
        BaselineEntry(trace_id="bl_t3", task_id="t3", passed=True, score=0.85),
        BaselineEntry(trace_id="bl_t4", task_id="t4", passed=False, score=0.0),
    )

    jsonl = _seed_traces(
        traces_dir,
        iteration=5,
        candidate_id="c",
        traces=[
            _trace(iteration=5, candidate_id="c", task_id="t1", passed=False, score=0.1),  # regressed
            _trace(iteration=5, candidate_id="c", task_id="t2", passed=True, score=0.95),  # breakthrough
            _trace(iteration=5, candidate_id="c", task_id="t3", passed=True, score=0.88),  # stable_pass
            _trace(iteration=5, candidate_id="c", task_id="t4", passed=False, score=0.05),  # persistent_fail
            _trace(iteration=5, candidate_id="c", task_id="t5", passed=True, score=0.5),   # no_baseline
        ],
    )

    indexer = Indexer(traces_dir / "index.db", baseline=baseline)
    indexer.write_manifest({"benchmark": "longmemeval", "is_baseline_run": False, "baseline_path": "/somewhere"})
    indexer.materialize_iteration(iteration=5, jsonl_paths=[jsonl])

    renderer = Renderer(traces_dir)
    out_path = renderer.render_iteration(5)
    text = out_path.read_text(encoding="utf-8")

    assert out_path.name == "iter_005.md"
    # Header
    assert "Iteration 5 — Diagnostic vs baseline" in text
    assert "Benchmark: longmemeval" in text
    assert "/somewhere" in text

    # Summary mentions every populated status.
    assert "regressed: 1" in text
    assert "breakthrough: 1" in text
    assert "persistent_fail: 1" in text
    assert "stable_pass: 1" in text
    assert "no_baseline: 1" in text

    # Detail sections exist for the actionable statuses.
    assert "REGRESSED" in text
    assert "BREAKTHROUGH" in text
    assert "PERSISTENT FAIL" in text

    # Per-trace block visible.
    assert "passed→failed" in text
    assert "failed→passed" in text
    # Score arrow with delta is present.
    assert "Δ" in text

    # Counts-only sections do not include detail.
    assert "STABLE PASS" in text
    assert "NO BASELINE" in text


def test_render_top_k_limits_detail_section(tmp_path):
    traces_dir = tmp_path / "traces"
    baseline = _baseline_with(
        *[
            BaselineEntry(trace_id=f"bl_t{i}", task_id=f"t{i}", passed=True, score=0.9)
            for i in range(10)
        ]
    )
    jsonl = _seed_traces(
        traces_dir,
        iteration=1,
        candidate_id="c",
        traces=[
            _trace(
                iteration=1,
                candidate_id="c",
                task_id=f"t{i}",
                passed=False,
                score=float(i) * 0.05,
            )
            for i in range(10)
        ],
    )

    indexer = Indexer(traces_dir / "index.db", baseline=baseline)
    indexer.write_manifest({"benchmark": "longmemeval", "is_baseline_run": False})
    indexer.materialize_iteration(iteration=1, jsonl_paths=[jsonl])

    Renderer(traces_dir, config=RenderConfig(top_k=3)).render_iteration(1)
    text = (traces_dir / "diagnostic" / "iter_001.md").read_text("utf-8")

    # Section header reports "top 3 of 10".
    assert "REGRESSED  (top 3 of 10)" in text
    # Heading lines for traces — count exactly 3 in the regressed
    # section (we only have one detail section in this test).
    headings = [line for line in text.splitlines() if line.startswith("### t")]
    assert len(headings) == 3


def test_render_truncates_long_prediction_and_doc(tmp_path):
    """Force a 'breakthrough' trace so detail rendering kicks in,
    then verify long fields get truncated by RenderConfig."""

    traces_dir = tmp_path / "traces"
    long_pred = "x" * 500
    long_doc = "y" * 500
    baseline = _baseline_with(
        BaselineEntry(trace_id="bl_t1", task_id="t1", passed=False, score=0.0)
    )
    jsonl = _seed_traces(
        traces_dir,
        iteration=1,
        candidate_id="c",
        traces=[
            _trace(
                iteration=1,
                candidate_id="c",
                task_id="t1",
                passed=True,  # was failing in baseline → breakthrough
                score=1.0,
                prediction=long_pred,
                docs=[
                    {
                        "rank": 1,
                        "score": 0.5,
                        "source": "s",
                        "content": long_doc,
                        "metadata": {},
                    }
                ],
            )
        ],
    )

    indexer = Indexer(traces_dir / "index.db", baseline=baseline)
    indexer.write_manifest({"benchmark": "longmemeval", "is_baseline_run": False})
    indexer.materialize_iteration(iteration=1, jsonl_paths=[jsonl])

    cfg = RenderConfig(truncate_prediction=50, truncate_doc_content=30)
    Renderer(traces_dir, config=cfg).render_iteration(1)
    text = (traces_dir / "diagnostic" / "iter_001.md").read_text("utf-8")

    # Truncation marker is present, full untruncated string is not.
    assert "…" in text
    assert long_pred not in text
    assert long_doc not in text


def test_render_baseline_run_emits_simplified_section(tmp_path):
    traces_dir = tmp_path / "traces"
    jsonl = _seed_traces(
        traces_dir,
        iteration=0,
        candidate_id="bl",
        traces=[
            _trace(iteration=0, candidate_id="bl", task_id="t1", passed=True, score=1.0),
            _trace(iteration=0, candidate_id="bl", task_id="t2", passed=False, score=0.0),
        ],
    )
    indexer = Indexer(traces_dir / "index.db")
    indexer.write_manifest({"benchmark": "longmemeval", "is_baseline_run": True})
    indexer.materialize_iteration(
        iteration=0, jsonl_paths=[jsonl], treat_as_baseline=True
    )

    Renderer(traces_dir).render_iteration(0)
    text = (traces_dir / "diagnostic" / "iter_000.md").read_text("utf-8")

    assert "Baseline Run" in text
    # No diff section headers in baseline mode.
    assert "REGRESSED" not in text
    assert "BREAKTHROUGH" not in text
    assert "passed: 1" in text
    assert "failed: 1" in text


def test_render_handles_missing_jsonl_gracefully(tmp_path):
    """If jsonl is purged after indexing, renderer notes the absence."""

    traces_dir = tmp_path / "traces"
    jsonl = _seed_traces(
        traces_dir,
        iteration=1,
        candidate_id="c",
        traces=[
            _trace(iteration=1, candidate_id="c", task_id="t1", passed=False, score=0.0)
        ],
    )
    baseline = _baseline_with(
        BaselineEntry(trace_id="bl_t1", task_id="t1", passed=True, score=1.0)
    )
    indexer = Indexer(traces_dir / "index.db", baseline=baseline)
    indexer.write_manifest({"benchmark": "longmemeval", "is_baseline_run": False})
    indexer.materialize_iteration(iteration=1, jsonl_paths=[jsonl])

    # Now nuke the jsonl.
    jsonl.unlink()

    Renderer(traces_dir).render_iteration(1)
    text = (traces_dir / "diagnostic" / "iter_001.md").read_text("utf-8")
    assert "trace body unavailable" in text


def test_render_for_unknown_iteration_writes_empty_summary(tmp_path):
    traces_dir = tmp_path / "traces"
    indexer = Indexer(traces_dir / "index.db")
    indexer.write_manifest({"benchmark": "longmemeval"})
    # No iteration data → renderer falls through to baseline-style
    # output and reports zero traces.
    out = Renderer(traces_dir).render_iteration(99)
    text = out.read_text("utf-8")
    assert "Traces: 0" in text
