"""Indexer — schema, materialization, idempotency, diff write-through."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from memomemo.traces import (
    STATUS_BASELINE,
    STATUS_BREAKTHROUGH,
    STATUS_NO_BASELINE,
    STATUS_PERSISTENT_FAIL,
    STATUS_REGRESSED,
    STATUS_STABLE_PASS,
    Baseline,
    BaselineEntry,
    Indexer,
    Span,
    Trace,
    write_jsonl,
)


def _trace(*, iteration: int, candidate_id: str, task_id: str, passed: bool, score: float) -> Trace:
    return Trace(
        trace_id=f"iter{iteration:03d}_{candidate_id}_{task_id}",
        iteration=iteration,
        candidate_id=candidate_id,
        task_id=task_id,
        benchmark="longmemeval",
        summary={"passed": passed, "score": score},
        diff=None,
        spans=[Span(id="s1", kind="retrieval", input=None, output=None)],
    )


def _write(path: Path, traces) -> Path:
    write_jsonl(path, traces)
    return path


def _query_all(db_path: Path, sql: str, params=()) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


# ---- schema --------------------------------------------------------


def test_schema_creates_three_tables(tmp_path):
    indexer = Indexer(tmp_path / "index.db")
    indexer.write_manifest({"benchmark": "longmemeval"})

    with sqlite3.connect(tmp_path / "index.db") as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"traces", "diffs", "manifest"}.issubset(names)


def test_manifest_stores_kv(tmp_path):
    indexer = Indexer(tmp_path / "index.db")
    indexer.write_manifest({"benchmark": "longmemeval", "is_baseline_run": True})
    rows = _query_all(tmp_path / "index.db", "SELECT key, value FROM manifest")
    by_key = {row["key"]: row["value"] for row in rows}
    assert by_key["benchmark"] == "longmemeval"
    assert by_key["is_baseline_run"] == "True"


# ---- materialization (baseline run) -------------------------------


def test_baseline_run_writes_baseline_status_for_all_traces(tmp_path):
    jsonl = _write(
        tmp_path / "spans" / "iter_001" / "c.jsonl",
        [
            _trace(iteration=1, candidate_id="c", task_id="t1", passed=True, score=1.0),
            _trace(iteration=1, candidate_id="c", task_id="t2", passed=False, score=0.0),
        ],
    )
    indexer = Indexer(tmp_path / "index.db")
    indexer.materialize_iteration(
        iteration=1, jsonl_paths=[jsonl], treat_as_baseline=True
    )

    diffs = _query_all(
        tmp_path / "index.db",
        "SELECT status, baseline_trace, baseline_score, delta FROM diffs ORDER BY trace_id",
    )
    assert len(diffs) == 2
    assert all(d["status"] == STATUS_BASELINE for d in diffs)
    assert all(d["baseline_trace"] is None for d in diffs)
    assert all(d["delta"] is None for d in diffs)


# ---- materialization (vs baseline) --------------------------------


def _baseline_with(*entries: BaselineEntry) -> Baseline:
    return Baseline({entry.task_id: entry for entry in entries})


def test_materialize_assigns_each_status_correctly(tmp_path):
    """One iteration with 4 task_ids covering each diff status."""

    baseline = _baseline_with(
        BaselineEntry(trace_id="bl_t1", task_id="t1", passed=True, score=0.9),
        BaselineEntry(trace_id="bl_t2", task_id="t2", passed=False, score=0.1),
        BaselineEntry(trace_id="bl_t3", task_id="t3", passed=True, score=0.8),
        BaselineEntry(trace_id="bl_t4", task_id="t4", passed=False, score=0.0),
        # task_id 't5' has no baseline → no_baseline
    )

    jsonl = _write(
        tmp_path / "spans" / "iter_005" / "c.jsonl",
        [
            # t1: pass→fail   regressed
            _trace(iteration=5, candidate_id="c", task_id="t1", passed=False, score=0.2),
            # t2: fail→pass   breakthrough
            _trace(iteration=5, candidate_id="c", task_id="t2", passed=True, score=0.95),
            # t3: pass→pass   stable_pass
            _trace(iteration=5, candidate_id="c", task_id="t3", passed=True, score=0.85),
            # t4: fail→fail   persistent_fail
            _trace(iteration=5, candidate_id="c", task_id="t4", passed=False, score=0.05),
            # t5: no baseline
            _trace(iteration=5, candidate_id="c", task_id="t5", passed=True, score=0.5),
        ],
    )

    indexer = Indexer(tmp_path / "index.db", baseline=baseline)
    indexer.materialize_iteration(iteration=5, jsonl_paths=[jsonl])

    diffs = _query_all(
        tmp_path / "index.db",
        "SELECT t.task_id, d.status, d.baseline_trace, d.baseline_score, d.delta "
        "FROM traces t JOIN diffs d USING (trace_id) ORDER BY t.task_id",
    )
    by_task = {row["task_id"]: row for row in diffs}

    assert by_task["t1"]["status"] == STATUS_REGRESSED
    assert by_task["t1"]["baseline_trace"] == "bl_t1"
    assert by_task["t1"]["baseline_score"] == 0.9
    assert abs(by_task["t1"]["delta"] - (0.2 - 0.9)) < 1e-9

    assert by_task["t2"]["status"] == STATUS_BREAKTHROUGH
    assert by_task["t3"]["status"] == STATUS_STABLE_PASS
    assert by_task["t4"]["status"] == STATUS_PERSISTENT_FAIL

    assert by_task["t5"]["status"] == STATUS_NO_BASELINE
    assert by_task["t5"]["baseline_trace"] is None
    assert by_task["t5"]["delta"] is None


def test_index_supports_status_filter_query(tmp_path):
    """Verifies the M2 acceptance criterion: SELECT WHERE status='regressed' works."""

    baseline = _baseline_with(
        BaselineEntry(trace_id="bl_a", task_id="t1", passed=True, score=0.9),
        BaselineEntry(trace_id="bl_b", task_id="t2", passed=True, score=0.9),
    )

    jsonl = _write(
        tmp_path / "spans" / "iter_001" / "c.jsonl",
        [
            _trace(iteration=1, candidate_id="c", task_id="t1", passed=False, score=0.0),
            _trace(iteration=1, candidate_id="c", task_id="t2", passed=True, score=1.0),
        ],
    )
    Indexer(tmp_path / "index.db", baseline=baseline).materialize_iteration(
        iteration=1, jsonl_paths=[jsonl]
    )

    rows = _query_all(
        tmp_path / "index.db",
        "SELECT trace_id FROM diffs WHERE status = ?",
        (STATUS_REGRESSED,),
    )
    assert len(rows) == 1
    assert "t1" in rows[0]["trace_id"]


# ---- idempotency --------------------------------------------------


def test_materialize_iteration_is_idempotent(tmp_path):
    indexer = Indexer(tmp_path / "index.db")

    jsonl = _write(
        tmp_path / "spans" / "iter_002" / "c.jsonl",
        [_trace(iteration=2, candidate_id="c", task_id="t1", passed=True, score=1.0)],
    )
    indexer.materialize_iteration(
        iteration=2, jsonl_paths=[jsonl], treat_as_baseline=True
    )

    # Re-run with a different payload for same iteration.
    jsonl2 = _write(
        tmp_path / "spans" / "iter_002" / "c.jsonl",
        [
            _trace(iteration=2, candidate_id="c", task_id="t1", passed=False, score=0.5),
            _trace(iteration=2, candidate_id="c", task_id="t2", passed=True, score=0.7),
        ],
    )
    indexer.materialize_iteration(
        iteration=2, jsonl_paths=[jsonl2], treat_as_baseline=True
    )

    rows = _query_all(
        tmp_path / "index.db",
        "SELECT trace_id, passed, score FROM traces ORDER BY task_id",
    )
    assert len(rows) == 2
    assert rows[0]["passed"] == 0
    assert rows[0]["score"] == 0.5
    assert rows[1]["passed"] == 1


def test_materialize_does_not_disturb_other_iterations(tmp_path):
    indexer = Indexer(tmp_path / "index.db")

    iter1_jsonl = _write(
        tmp_path / "spans" / "iter_001" / "c.jsonl",
        [_trace(iteration=1, candidate_id="c", task_id="t1", passed=True, score=1.0)],
    )
    indexer.materialize_iteration(
        iteration=1, jsonl_paths=[iter1_jsonl], treat_as_baseline=True
    )

    iter2_jsonl = _write(
        tmp_path / "spans" / "iter_002" / "c.jsonl",
        [_trace(iteration=2, candidate_id="c", task_id="t2", passed=False, score=0.0)],
    )
    indexer.materialize_iteration(
        iteration=2, jsonl_paths=[iter2_jsonl], treat_as_baseline=True
    )

    # Re-running iter 2 must keep iter 1 row intact.
    indexer.materialize_iteration(
        iteration=2, jsonl_paths=[iter2_jsonl], treat_as_baseline=True
    )
    rows = _query_all(
        tmp_path / "index.db",
        "SELECT iteration, task_id FROM traces ORDER BY iteration, task_id",
    )
    assert [(row["iteration"], row["task_id"]) for row in rows] == [(1, "t1"), (2, "t2")]
