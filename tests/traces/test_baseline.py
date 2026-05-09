"""Baseline loader — both jsonl and sqlite paths."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from optimizer1.traces import Baseline, Span, Trace, write_jsonl


def _trace(*, task_id: str, passed: bool, score: float, candidate_id: str = "bl") -> Trace:
    return Trace(
        trace_id=f"iter000_{candidate_id}_{task_id}",
        iteration=0,
        candidate_id=candidate_id,
        task_id=task_id,
        benchmark="longmemeval",
        summary={"passed": passed, "score": score},
        diff=None,
        spans=[Span(id="s1", kind="retrieval", input=None, output={"documents": []})],
    )


def _write_jsonl_iteration(traces_dir: Path, iteration: int, candidate_id: str, traces):
    path = traces_dir / "spans" / f"iter_{iteration:03d}" / f"{candidate_id}.jsonl"
    write_jsonl(path, traces)


def test_load_returns_empty_when_dir_missing(tmp_path):
    assert len(Baseline.load(tmp_path / "no_such")) == 0


def test_load_from_jsonl_when_no_index_db(tmp_path):
    traces_dir = tmp_path / "traces"
    _write_jsonl_iteration(
        traces_dir,
        0,
        "cand_a",
        [
            _trace(task_id="t1", passed=True, score=1.0),
            _trace(task_id="t2", passed=False, score=0.2),
        ],
    )
    bl = Baseline.load(traces_dir)
    assert len(bl) == 2
    assert bl.lookup("t1").passed is True
    assert bl.lookup("t1").score == 1.0
    assert bl.lookup("t2").passed is False
    assert bl.lookup("nope") is None


def test_load_picks_passed_over_failed_for_same_task(tmp_path):
    traces_dir = tmp_path / "traces"
    _write_jsonl_iteration(
        traces_dir,
        0,
        "cand_a",
        [_trace(task_id="t1", passed=False, score=0.7, candidate_id="cand_a")],
    )
    _write_jsonl_iteration(
        traces_dir,
        0,
        "cand_b",
        [_trace(task_id="t1", passed=True, score=0.4, candidate_id="cand_b")],
    )
    bl = Baseline.load(traces_dir)
    entry = bl.lookup("t1")
    # Passed wins regardless of score.
    assert entry.passed is True
    assert "cand_b" in entry.trace_id


def test_load_picks_higher_score_when_passed_status_ties(tmp_path):
    traces_dir = tmp_path / "traces"
    _write_jsonl_iteration(
        traces_dir,
        0,
        "cand_a",
        [_trace(task_id="t1", passed=True, score=0.6, candidate_id="cand_a")],
    )
    _write_jsonl_iteration(
        traces_dir,
        0,
        "cand_b",
        [_trace(task_id="t1", passed=True, score=0.9, candidate_id="cand_b")],
    )
    bl = Baseline.load(traces_dir)
    entry = bl.lookup("t1")
    assert entry.score == 0.9
    assert "cand_b" in entry.trace_id


def test_load_prefers_sqlite_when_index_db_exists(tmp_path):
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    db_path = traces_dir / "index.db"

    # Write jsonl with one set of values...
    _write_jsonl_iteration(
        traces_dir,
        0,
        "cand_jsonl",
        [_trace(task_id="t1", passed=False, score=0.0, candidate_id="cand_jsonl")],
    )
    # ...but sqlite says something different. Sqlite must win.
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE traces (
                trace_id TEXT PRIMARY KEY,
                iteration INTEGER, candidate_id TEXT, task_id TEXT,
                benchmark TEXT, passed INTEGER, score REAL,
                jsonl_path TEXT, jsonl_lineno INTEGER
            );
            INSERT INTO traces VALUES
                ('bl_db_t1', 0, 'cand_db', 't1', 'longmemeval', 1, 0.95, '', 0);
            """
        )

    bl = Baseline.load(traces_dir)
    entry = bl.lookup("t1")
    assert entry.passed is True
    assert entry.score == 0.95
    assert entry.trace_id == "bl_db_t1"
