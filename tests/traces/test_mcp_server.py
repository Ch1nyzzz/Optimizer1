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
    # Seed raw diff text for trace_similar to embed lazily. No
    # diff_embeddings rows: tests stub the DiffEmbedder so the lazy
    # path computes deterministic vectors on demand.
    indexer.record_diff_text(
        iteration=1,
        diff_text="diff --git a/src/a.py b/src/a.py\n+ change",
    )
    indexer.record_diff_text(
        iteration=2,
        diff_text="diff --git a/src/a.py b/src/a.py\n+ tweak",
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


def test_tool_file_history(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    rows = srv.trace_file_history("src/a.py")
    assert [r["iteration"] for r in rows] == [1, 2]


def test_tool_candidate_outcome(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    out = srv.trace_candidate_outcome(1, "cand_x", max_examples=1)
    assert out["n_traces"] == 2
    assert [item["task_id"] for item in out["breakthrough_tasks"]] == ["task_b"]
    assert out["failed_tasks"] == []
    # Rationale layer removed.
    assert "rationale" not in out


class _StubEmbedder:
    """Deterministic embedder for lazy-embed tests.

    Maps textual cues to fixed unit vectors so cosine sim is exact.
    Counts calls so tests can assert cache behavior.
    """

    DEFAULT_MODEL = "stub-model"
    calls: int = 0

    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model") or self.DEFAULT_MODEL

    def embed(self, text):
        type(self).calls += 1
        if "tweak" in text or "match-iter2" in text:
            vec = (0.0, 1.0, 0.0)
        elif "change" in text or "match-iter1" in text:
            vec = (1.0, 0.0, 0.0)
        else:
            vec = (0.0, 0.0, 1.0)
        return DiffEmbedding(
            model=self.model, dim=3, diff_text=text, vector=vec
        )


def test_tool_similar_finds_top_k_via_lazy_embed(tmp_path, monkeypatch):
    """trace_similar lazily embeds the seeded diff_text rows on first
    call, then ranks by cosine similarity to the query embedding."""

    db_path = _seed_db(tmp_path)
    srv = _import_server(db_path)

    _StubEmbedder.calls = 0
    monkeypatch.setattr(srv, "DiffEmbedder", _StubEmbedder)

    rows = srv.trace_similar("match-iter1 please", k=2)
    assert rows[0]["iteration"] == 1
    assert rows[0]["similarity"] > 0.99
    assert rows[0]["model"] == "stub-model"
    assert rows[0]["status_counts"]
    # 2 iters embedded lazily + 1 query embed = 3 calls on first run.
    assert _StubEmbedder.calls == 3


def test_tool_similar_caches_embeddings_after_first_call(tmp_path, monkeypatch):
    """Second call to trace_similar should reuse the cached embeddings;
    only the query is re-embedded."""

    db_path = _seed_db(tmp_path)
    srv = _import_server(db_path)
    monkeypatch.setattr(srv, "DiffEmbedder", _StubEmbedder)

    _StubEmbedder.calls = 0
    srv.trace_similar("match-iter1", k=2)
    first_calls = _StubEmbedder.calls

    _StubEmbedder.calls = 0
    srv.trace_similar("match-iter2", k=2)
    second_calls = _StubEmbedder.calls

    # First run: 2 iter embeds + 1 query embed = 3.
    assert first_calls == 3
    # Second run: cache hit on both iters, only the query is embedded.
    assert second_calls == 1


def test_tool_similar_re_embeds_when_model_switches(tmp_path, monkeypatch):
    """A different DIFF_EMBEDDING_MODEL forces a fresh cache row per
    iter, then ranks against the new model's vectors."""

    import sqlite3

    db_path = _seed_db(tmp_path)
    srv = _import_server(db_path)
    monkeypatch.setattr(srv, "DiffEmbedder", _StubEmbedder)

    # First run with default model populates one cache row per iter.
    srv.trace_similar("match-iter1", k=2)

    # Switch model — env override drives lazy-embed to write fresh rows.
    os.environ["DIFF_EMBEDDING_MODEL"] = "alt-model"
    try:
        _StubEmbedder.DEFAULT_MODEL = "alt-model"
        _StubEmbedder.calls = 0
        rows = srv.trace_similar("match-iter1", k=2)
    finally:
        del os.environ["DIFF_EMBEDDING_MODEL"]
        _StubEmbedder.DEFAULT_MODEL = "stub-model"

    assert rows[0]["model"] == "alt-model"
    # Both iters must be re-embedded under the new model + the query = 3.
    assert _StubEmbedder.calls == 3
    # Both models' rows coexist in the cache — switching back stays cheap.
    with sqlite3.connect(db_path) as conn:
        models = {
            row[0]
            for row in conn.execute("SELECT DISTINCT model FROM diff_embeddings")
        }
    assert models == {"stub-model", "alt-model"}


def test_tool_similar_empty_returns_empty(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    assert srv.trace_similar("   ", k=5) == []


def test_tool_list_tasks_full_set(tmp_path):
    # Seed has task_a in iters 0-3 and task_b only at iter 1.
    srv = _import_server(_seed_db(tmp_path))
    tasks = srv.trace_list_tasks()
    assert tasks == ["task_a", "task_b"]


def test_tool_list_tasks_scoped(tmp_path):
    srv = _import_server(_seed_db(tmp_path))
    assert srv.trace_list_tasks(iteration=0) == ["task_a"]
    assert srv.trace_list_tasks(iteration=1) == ["task_a", "task_b"]


def test_tool_iteration_metadata(tmp_path):
    db_path = _seed_db(tmp_path)
    Indexer(db_path).upsert_iteration_meta(
        iteration=2,
        patch_base=1,
        budget="high",
        selection_policy="pareto",
        passrate=0.5,
    )
    srv = _import_server(db_path)
    rows = srv.trace_iteration_metadata()
    assert len(rows) == 1
    assert rows[0]["iteration"] == 2
    assert rows[0]["selection_policy"] == "pareto"


def test_tool_compare_iterations(tmp_path):
    # iter 0 has task_a (pass); iter 1 has task_a (pass) + task_b (pass).
    srv = _import_server(_seed_db(tmp_path))
    rows = srv.trace_compare_iterations(left=0, right=1)
    by_task = {r["task_id"]: r["classification"] for r in rows}
    assert by_task["task_a"] == "stable_pass"
    assert by_task["task_b"] == "only_in_right"
