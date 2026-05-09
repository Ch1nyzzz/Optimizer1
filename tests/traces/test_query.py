"""Tests for TraceQuery — structured queries over index.db."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from optimizer1.traces.query import TraceQuery
from optimizer1.traces.diff import (
    STATUS_BASELINE,
    STATUS_BREAKTHROUGH,
    STATUS_PERSISTENT_FAIL,
    STATUS_REGRESSED,
    STATUS_STABLE_PASS,
)
from optimizer1.traces.indexer import Indexer


# ---- fixtures: build a tiny index.db by hand -----------------------


def _seed_db(tmp_path: Path) -> Path:
    """Build a minimal index.db with a known status mosaic.

    Layout (3 iters × 2 tasks × 1 candidate):
        iter 0  cand=baseline   task_a baseline    task_b baseline
        iter 1  cand=cand_x     task_a stable_pass  task_b regressed
        iter 2  cand=cand_x     task_a persistent_fail (was failing in iter1?)
                                task_b persistent_fail
        iter 3  cand=cand_x     task_a persistent_fail   task_b breakthrough
    """

    db_path = tmp_path / "traces" / "index.db"
    indexer = Indexer(db_path)
    # Touch schema by opening a connection.
    with indexer._connect() as conn:
        rows_traces = []
        rows_diffs = []

        def add(trace_id, iteration, candidate, task, passed, score):
            rows_traces.append(
                (
                    trace_id,
                    iteration,
                    candidate,
                    task,
                    "longmemeval",
                    1 if passed else 0,
                    score,
                    str(tmp_path / "fake.jsonl"),
                    1,
                )
            )

        def diff(trace_id, status, baseline_score=None, delta=None):
            rows_diffs.append((trace_id, None, status, baseline_score, delta))

        # iter 0: baseline traces (both pass).
        add("t-0-a", 0, "baseline", "task_a", True, 1.0)
        diff("t-0-a", STATUS_BASELINE)
        add("t-0-b", 0, "baseline", "task_b", True, 1.0)
        diff("t-0-b", STATUS_BASELINE)

        # iter 1: task_a stable, task_b regressed
        add("t-1-a", 1, "cand_x", "task_a", True, 1.0)
        diff("t-1-a", STATUS_STABLE_PASS, baseline_score=1.0, delta=0.0)
        add("t-1-b", 1, "cand_x", "task_b", False, 0.0)
        diff("t-1-b", STATUS_REGRESSED, baseline_score=1.0, delta=-1.0)

        # iter 2: task_a regressed (was passing in baseline), task_b still
        # regressed; both flagged as 'regressed' since baseline is iter_0.
        add("t-2-a", 2, "cand_x", "task_a", False, 0.0)
        diff("t-2-a", STATUS_REGRESSED, baseline_score=1.0, delta=-1.0)
        add("t-2-b", 2, "cand_x", "task_b", False, 0.0)
        diff("t-2-b", STATUS_REGRESSED, baseline_score=1.0, delta=-1.0)

        # iter 3: task_a still failing; task_b is fictional task that
        # was failing in baseline (override) — emulate breakthrough
        # by retroactive hand-crafted row.
        add("t-3-a", 3, "cand_x", "task_a", False, 0.0)
        diff("t-3-a", STATUS_PERSISTENT_FAIL)
        add("t-3-b", 3, "cand_x", "task_c", True, 1.0)
        diff("t-3-b", STATUS_BREAKTHROUGH, baseline_score=0.0, delta=1.0)

        conn.executemany(
            "INSERT INTO traces "
            "(trace_id, iteration, candidate_id, task_id, benchmark, passed, score, jsonl_path, jsonl_lineno) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows_traces,
        )
        conn.executemany(
            "INSERT INTO diffs (trace_id, baseline_trace, status, baseline_score, delta) "
            "VALUES (?, ?, ?, ?, ?)",
            rows_diffs,
        )

    # file_modifications: iter 1 changed memgpt.py; iter 2 changed both.
    indexer.record_file_modifications(
        iteration=1, paths=["src/optimizer1/scaffolds/memgpt_scaffold.py"]
    )
    indexer.record_file_modifications(
        iteration=2,
        paths=[
            "src/optimizer1/scaffolds/memgpt_scaffold.py",
            "src/optimizer1/locomo.py",
        ],
    )
    return db_path


# ---- task_history --------------------------------------------------


def test_task_history_returns_iter_ordered_rows(tmp_path):
    db_path = _seed_db(tmp_path)
    query = TraceQuery(db_path)

    rows = query.task_history("task_a")

    assert [r["iteration"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["status"] == STATUS_BASELINE
    assert rows[1]["status"] == STATUS_STABLE_PASS
    assert rows[2]["status"] == STATUS_REGRESSED
    assert rows[3]["status"] == STATUS_PERSISTENT_FAIL


def test_task_history_unknown_task_returns_empty(tmp_path):
    db_path = _seed_db(tmp_path)
    assert TraceQuery(db_path).task_history("never_seen") == []


# ---- persistent_failures ------------------------------------------


def test_persistent_failures_counts_trailing_streak(tmp_path):
    db_path = _seed_db(tmp_path)
    # Add an extra iter so task_a has a 2-long persistent_fail streak.
    indexer = Indexer(db_path)
    with indexer._connect() as conn:
        conn.execute(
            "INSERT INTO traces "
            "(trace_id, iteration, candidate_id, task_id, benchmark, passed, score, jsonl_path, jsonl_lineno) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "t-4-a",
                4,
                "cand_x",
                "task_a",
                "longmemeval",
                0,
                0.0,
                str(tmp_path / "fake.jsonl"),
                1,
            ),
        )
        conn.execute(
            "INSERT INTO diffs (trace_id, baseline_trace, status, baseline_score, delta) "
            "VALUES (?, ?, ?, ?, ?)",
            ("t-4-a", None, STATUS_PERSISTENT_FAIL, None, None),
        )

    failures = TraceQuery(db_path).persistent_failures(min_streak=2)

    assert any(f["task_id"] == "task_a" and f["current_streak"] == 2 for f in failures)
    # task_b never enters persistent_fail (its iter 2 row is regressed).
    assert not any(f["task_id"] == "task_b" for f in failures)


def test_persistent_failures_min_streak_filters_short_runs(tmp_path):
    db_path = _seed_db(tmp_path)
    failures = TraceQuery(db_path).persistent_failures(min_streak=2)
    # Only task_a's 1-long streak exists in seed; raise threshold and we
    # should get nothing.
    assert TraceQuery(db_path).persistent_failures(min_streak=5) == []


# ---- breakthroughs --------------------------------------------------


def test_breakthroughs_filters_by_iter_lower_bound(tmp_path):
    db_path = _seed_db(tmp_path)
    rows = TraceQuery(db_path).breakthroughs(since_iter=0)
    assert len(rows) == 1
    assert rows[0]["task_id"] == "task_c"
    assert rows[0]["iteration"] == 3
    assert TraceQuery(db_path).breakthroughs(since_iter=4) == []


# ---- regressions ----------------------------------------------------


def test_regressions_only_within_window(tmp_path):
    db_path = _seed_db(tmp_path)
    # max iter is 3, window=2 covers iter ∈ {2, 3}.
    rows = TraceQuery(db_path).regressions(window=2)
    iters = sorted({r["iteration"] for r in rows})
    assert iters == [2]  # iter 1 row excluded; iter 3 has no regressions
    # window=3 covers iter ∈ {1, 2, 3} so iter 1's regression is included
    rows = TraceQuery(db_path).regressions(window=3)
    iters = sorted({r["iteration"] for r in rows})
    assert iters == [1, 2]


# ---- file_history ---------------------------------------------------


def test_file_history_aggregates_iter_status_counts(tmp_path):
    db_path = _seed_db(tmp_path)
    rows = TraceQuery(db_path).file_history(
        "src/optimizer1/scaffolds/memgpt_scaffold.py"
    )
    assert [r["iteration"] for r in rows] == [1, 2]
    iter1 = rows[0]
    assert iter1["status_counts"]["stable_pass"] == 1
    assert iter1["status_counts"]["regressed"] == 1
    iter2 = rows[1]
    assert iter2["status_counts"]["regressed"] == 2


def test_file_history_unknown_path_returns_empty(tmp_path):
    db_path = _seed_db(tmp_path)
    assert TraceQuery(db_path).file_history("does/not/exist.py") == []


# ---- candidate_outcome ---------------------------------------------


def test_candidate_outcome_summarizes_iter_candidate(tmp_path):
    db_path = _seed_db(tmp_path)
    out = TraceQuery(db_path).candidate_outcome(2, "cand_x")
    assert out["n_traces"] == 2
    assert out["passrate"] == 0.0
    assert out["status_counts"]["regressed"] == 2
    # iter 2 modified two files
    assert sorted(out["modified_paths"]) == [
        "src/optimizer1/locomo.py",
        "src/optimizer1/scaffolds/memgpt_scaffold.py",
    ]


def test_candidate_outcome_missing_returns_zero(tmp_path):
    db_path = _seed_db(tmp_path)
    out = TraceQuery(db_path).candidate_outcome(99, "cand_x")
    assert out["n_traces"] == 0
    assert out["passrate"] is None
    assert out["modified_paths"] == []


# ---- TraceQuery.__init__ -------------------------------------------


def test_traces_query_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TraceQuery(tmp_path / "nope.db")



