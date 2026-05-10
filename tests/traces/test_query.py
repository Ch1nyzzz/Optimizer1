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
    iter2_dir = tmp_path / "proposer_calls" / "iter_002"
    iter2_dir.mkdir(parents=True)
    (iter2_dir / "diff.patch").write_text(
        "diff --git a/src/optimizer1/locomo.py b/src/optimizer1/locomo.py\n",
        encoding="utf-8",
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
    assert out["diff_path"].endswith("proposer_calls/iter_002/diff.patch")
    # iter 2 modified two files
    assert sorted(out["modified_paths"]) == [
        "src/optimizer1/locomo.py",
        "src/optimizer1/scaffolds/memgpt_scaffold.py",
    ]
    assert [item["task_id"] for item in out["regressed_tasks"]] == [
        "task_a",
        "task_b",
    ]
    assert [item["task_id"] for item in out["failed_tasks"]] == [
        "task_a",
        "task_b",
    ]
    assert out["regressed_tasks"][0]["jsonl_path"].endswith("fake.jsonl")
    assert out["regressed_tasks"][0]["delta"] == -1.0


def test_candidate_outcome_returns_breakthrough_examples(tmp_path):
    db_path = _seed_db(tmp_path)
    out = TraceQuery(db_path).candidate_outcome(3, "cand_x")
    assert [item["task_id"] for item in out["breakthrough_tasks"]] == ["task_c"]
    assert [item["task_id"] for item in out["failed_tasks"]] == ["task_a"]


def test_candidate_outcome_caps_examples(tmp_path):
    db_path = _seed_db(tmp_path)
    out = TraceQuery(db_path).candidate_outcome(2, "cand_x", max_examples=1)
    assert [item["task_id"] for item in out["regressed_tasks"]] == ["task_a"]
    assert [item["task_id"] for item in out["failed_tasks"]] == ["task_a"]


def test_candidate_outcome_missing_returns_zero(tmp_path):
    db_path = _seed_db(tmp_path)
    out = TraceQuery(db_path).candidate_outcome(99, "cand_x")
    assert out["n_traces"] == 0
    assert out["passrate"] is None
    assert out["diff_path"] is None
    assert out["modified_paths"] == []
    assert out["regressed_tasks"] == []
    assert out["breakthrough_tasks"] == []
    assert out["failed_tasks"] == []


# ---- list_tasks ----------------------------------------------------


def test_list_tasks_returns_distinct_ids_across_iters(tmp_path):
    db_path = _seed_db(tmp_path)
    tasks = TraceQuery(db_path).list_tasks()
    # task_a / task_b appear in iters 0-2; task_c only in iter 3.
    assert tasks == ["task_a", "task_b", "task_c"]


def test_list_tasks_scoped_to_one_iteration(tmp_path):
    db_path = _seed_db(tmp_path)
    assert TraceQuery(db_path).list_tasks(iteration=0) == ["task_a", "task_b"]
    assert TraceQuery(db_path).list_tasks(iteration=3) == ["task_a", "task_c"]


def test_list_tasks_unknown_iter_returns_empty(tmp_path):
    db_path = _seed_db(tmp_path)
    assert TraceQuery(db_path).list_tasks(iteration=99) == []


# ---- iteration_metadata --------------------------------------------


def test_iteration_metadata_returns_full_rows(tmp_path):
    db_path = _seed_db(tmp_path)
    Indexer(db_path).upsert_iteration_meta(
        iteration=2,
        patch_base=1,
        budget="high",
        selection_policy="pareto",
        advanced_frontier=False,
        on_pareto_frontier=True,
        passrate=0.5,
        mean_score=0.4,
        proposer_call_dir="/runs/r/proposer_calls/iter_002",
    )
    rows = TraceQuery(db_path).iteration_metadata()
    assert len(rows) == 1
    row = rows[0]
    assert row["iteration"] == 2
    assert row["patch_base"] == 1
    assert row["budget"] == "high"
    assert row["selection_policy"] == "pareto"
    assert row["advanced_frontier"] is False
    assert row["on_pareto_frontier"] is True
    assert row["passrate"] == 0.5
    assert row["mean_score"] == 0.4
    assert row["proposer_call_dir"].endswith("iter_002")


def test_iteration_metadata_filters_by_iters_list(tmp_path):
    db_path = _seed_db(tmp_path)
    indexer = Indexer(db_path)
    indexer.upsert_iteration_meta(iteration=1, passrate=0.3)
    indexer.upsert_iteration_meta(iteration=2, passrate=0.5)
    indexer.upsert_iteration_meta(iteration=3, passrate=0.7)

    rows = TraceQuery(db_path).iteration_metadata(iters=[1, 3])
    assert [r["iteration"] for r in rows] == [1, 3]


def test_iteration_metadata_empty_filter_returns_empty(tmp_path):
    db_path = _seed_db(tmp_path)
    assert TraceQuery(db_path).iteration_metadata(iters=[]) == []


# ---- compare_iterations --------------------------------------------


def test_compare_iterations_classifies_each_pair(tmp_path):
    db_path = _seed_db(tmp_path)
    rows = TraceQuery(db_path).compare_iterations(left=0, right=1)
    by_task = {row["task_id"]: row for row in rows}

    # task_a: passed in both iters → stable_pass
    assert by_task["task_a"]["classification"] == "stable_pass"
    # task_b: passed at baseline (iter 0), failed in iter 1 → regressed
    assert by_task["task_b"]["classification"] == "regressed_RvL"
    assert by_task["task_b"]["delta"] == -1.0
    assert by_task["task_b"]["left"]["passed"] is True
    assert by_task["task_b"]["right"]["passed"] is False


def test_compare_iterations_handles_only_in_left_or_right(tmp_path):
    db_path = _seed_db(tmp_path)
    # iter 0 has {task_a, task_b}; iter 3 has {task_a, task_c}
    rows = TraceQuery(db_path).compare_iterations(left=0, right=3)
    by_task = {row["task_id"]: row for row in rows}

    assert by_task["task_b"]["classification"] == "only_in_left"
    assert by_task["task_b"]["delta"] is None
    assert by_task["task_b"]["right"] is None

    assert by_task["task_c"]["classification"] == "only_in_right"
    assert by_task["task_c"]["delta"] is None
    assert by_task["task_c"]["left"] is None


def test_compare_iterations_orders_regressions_first(tmp_path):
    db_path = _seed_db(tmp_path)
    # iter 0 vs iter 2 — both task_a and task_b regressed.
    rows = TraceQuery(db_path).compare_iterations(left=0, right=2)
    classifications = [r["classification"] for r in rows]
    # All regressed_RvL come before any other class.
    seen_other = False
    for c in classifications:
        if c != "regressed_RvL":
            seen_other = True
        elif seen_other:
            raise AssertionError(
                f"regressed_RvL after non-regressed: {classifications}"
            )


def test_compare_iterations_picks_headline_candidate_by_default(tmp_path):
    """When no candidate_id passed, the highest-passrate candidate is
    used. With ties, candidate_id desc wins (matches frontier rule)."""

    db_path = tmp_path / "traces" / "index.db"
    indexer = Indexer(db_path)
    with indexer._connect() as conn:
        # Iter 5 has two candidates: cand_a all-fail, cand_b all-pass.
        # Headline should be cand_b.
        for cid, passed in [("cand_a", 0), ("cand_b", 1)]:
            for task, score in [("t1", 1.0), ("t2", 1.0)]:
                conn.execute(
                    "INSERT INTO traces (trace_id, iteration, candidate_id, task_id, "
                    "benchmark, passed, score, jsonl_path, jsonl_lineno) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"5-{cid}-{task}",
                        5,
                        cid,
                        task,
                        "longmemeval",
                        passed,
                        score if passed else 0.0,
                        str(tmp_path / "fake.jsonl"),
                        1,
                    ),
                )
        # Iter 6 has only cand_b, all-fail.
        for task in ("t1", "t2"):
            conn.execute(
                "INSERT INTO traces (trace_id, iteration, candidate_id, task_id, "
                "benchmark, passed, score, jsonl_path, jsonl_lineno) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"6-cand_b-{task}",
                    6,
                    "cand_b",
                    task,
                    "longmemeval",
                    0,
                    0.0,
                    str(tmp_path / "fake.jsonl"),
                    1,
                ),
            )

    rows = TraceQuery(db_path).compare_iterations(left=5, right=6)
    by_task = {row["task_id"]: row for row in rows}
    assert by_task["t1"]["left"]["candidate_id"] == "cand_b"
    assert by_task["t1"]["classification"] == "regressed_RvL"


def test_compare_iterations_explicit_candidate_ids(tmp_path):
    db_path = _seed_db(tmp_path)
    # Force the comparison to use 'baseline' on left and 'cand_x' on right.
    rows = TraceQuery(db_path).compare_iterations(
        left=0,
        right=1,
        left_candidate_id="baseline",
        right_candidate_id="cand_x",
    )
    by_task = {row["task_id"]: row for row in rows}
    assert by_task["task_a"]["left"]["candidate_id"] == "baseline"
    assert by_task["task_a"]["right"]["candidate_id"] == "cand_x"


def test_compare_iterations_missing_iter_returns_empty(tmp_path):
    db_path = _seed_db(tmp_path)
    assert TraceQuery(db_path).compare_iterations(left=99, right=100) == []


# ---- TraceQuery.__init__ -------------------------------------------


def test_traces_query_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TraceQuery(tmp_path / "nope.db")


