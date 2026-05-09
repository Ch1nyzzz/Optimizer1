"""Tests for the trace-tools MCP server.

We don't spin up the JSON-RPC stdio loop here; we import the tool
functions directly and exercise them against a hand-seeded index.db.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from optimizer1.traces.embeddings import DiffEmbedding, pack_vector
from optimizer1.traces.indexer import Indexer
from optimizer1.traces.diff import (
    STATUS_BASELINE,
    STATUS_BREAKTHROUGH,
    STATUS_PERSISTENT_FAIL,
    STATUS_REGRESSED,
    STATUS_STABLE_PASS,
)


def _seed_db(tmp_path: Path) -> Path:
    """Build a small index.db with traces + diffs + embeddings."""

    db_path = tmp_path / "traces" / "index.db"
    indexer = Indexer(db_path)
    with indexer._connect() as conn:
        traces = []
        diffs = []

        def add(trace_id, iteration, cand, task, passed, score):
            traces.append(
                (trace_id, iteration, cand, task, "longmemeval",
                 1 if passed else 0, score, str(tmp_path / "fake.jsonl"), 1)
            )

        def diff(trace_id, status, **kw):
            diffs.append(
                (trace_id, None, status,
                 kw.get("baseline_score"), kw.get("delta"))
            )

        add("t0a", 0, "seed", "task_a", True, 1.0)
        diff("t0a", STATUS_BASELINE)
        add("t1a", 1, "cand_x", "task_a", True, 1.0)
        diff("t1a", STATUS_STABLE_PASS, baseline_score=1.0, delta=0.0)
        add("t2a", 2, "cand_x", "task_a", False, 0.0)
        diff("t2a", STATUS_REGRESSED, baseline_score=1.0, delta=-1.0)
        add("t3a", 3, "cand_x", "task_a", False, 0.0)
        diff("t3a", STATUS_PERSISTENT_FAIL)
        add("t1b", 1, "cand_x", "task_b", True, 1.0)
        diff("t1b", STATUS_BREAKTHROUGH, baseline_score=0.0, delta=1.0)

        conn.executemany(
            "INSERT INTO traces "
            "(trace_id, iteration, candidate_id, task_id, benchmark, "
            "passed, score, jsonl_path, jsonl_lineno) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            traces,
        )
        conn.executemany(
            "INSERT INTO diffs "
            "(trace_id, baseline_trace, status, baseline_score, delta) "
            "VALUES (?, ?, ?, ?, ?)",
            diffs,
        )
    indexer.record_file_modifications(iteration=1, paths=["src/a.py"])
    indexer.record_file_modifications(iteration=2, paths=["src/a.py", "src/b.py"])
    indexer.record_diff_embedding(
        iteration=1,
        model="stub-model",
        dim=3,
        diff_text="diff --git a/src/a.py b/src/a.py\n+ change",
        embedding=pack_vector((1.0, 0.0, 0.0)),
    )
    indexer.record_diff_embedding(
        iteration=2,
        model="stub-model",
        dim=3,
        diff_text="diff --git a/src/a.py b/src/a.py\n+ tweak",
        embedding=pack_vector((0.0, 1.0, 0.0)),
    )
    return db_path


def _import_server(db_path: Path):
    """Reload the mcp_server module so its module-level _DB_PATH binds
    to our tmp DB instead of cwd's."""

    os.environ["TRACE_DB"] = str(db_path)
    import optimizer1.traces.mcp_server as srv

    importlib.reload(srv)
    return srv


# ---- per-tool ------------------------------------------------------


def test_tool_task_history(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    rows = srv.trace_task_history("task_a")
    assert [r["iteration"] for r in rows] == [0, 1, 2, 3]
    assert rows[1]["status"] == STATUS_STABLE_PASS
    # Rationale layer removed: keys should not appear at all.
    assert "rationale_hypothesis" not in rows[1]
    assert "rationale_diagnosis" not in rows[1]


def test_tool_persistent_failures(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    rows = srv.trace_persistent_failures(min_streak=1)
    assert any(r["task_id"] == "task_a" for r in rows)


def test_tool_breakthroughs(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    rows = srv.trace_breakthroughs(since_iter=0)
    assert len(rows) == 1 and rows[0]["task_id"] == "task_b"
    # Rationale layer removed.
    assert "rationale_hypothesis" not in rows[0]


def test_tool_regressions(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    rows = srv.trace_regressions(window=2)
    assert any(r["iteration"] == 2 for r in rows)


def test_tool_file_history(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    rows = srv.trace_file_history("src/a.py")
    assert [r["iteration"] for r in rows] == [1, 2]


def test_tool_candidate_outcome(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    out = srv.trace_candidate_outcome(1, "cand_x")
    assert out["n_traces"] == 2
    # Rationale layer removed.
    assert "rationale" not in out


def test_tool_similar_finds_top_k(tmp_path, monkeypatch):
    """Stub the embedder so trace_similar returns deterministic similarity."""

    db_path = _seed_db(tmp_path)
    srv = _import_server(db_path)

    class _StubEmbedder:
        def __init__(self, *args, **kwargs):
            self.model = kwargs.get("model") or "stub-model"

        def embed(self, text):
            # exact match for stored vector(1, 0, 0) at iter 1
            if "iter1" in text or "match-iter1" in text:
                vec = (1.0, 0.0, 0.0)
            else:
                vec = (0.0, 1.0, 0.0)
            return DiffEmbedding(
                model=self.model, dim=3, diff_text=text, vector=vec
            )

    monkeypatch.setattr(srv, "DiffEmbedder", _StubEmbedder)
    rows = srv.trace_similar("match-iter1 please", k=2)
    assert rows[0]["iteration"] == 1
    assert rows[0]["similarity"] > 0.99
    assert rows[0]["status_counts"]
    # Rationale layer removed.
    assert "rationale_hypothesis" not in rows[0]


def test_tool_similar_empty_returns_empty(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    assert srv.trace_similar("   ", k=5) == []
