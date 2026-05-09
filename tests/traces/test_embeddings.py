"""Tests for the diff embedding helper and harness integration."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from optimizer1.schemas import CandidateResult
from optimizer1.traces.embeddings import (
    DiffEmbedder,
    DiffEmbedding,
    cosine_similarity,
    pack_vector,
    unpack_vector,
)
from optimizer1.traces.harness import TraceHarness


# ---- pure helpers ---------------------------------------------------


def test_pack_unpack_roundtrip():
    vec = (0.1, -0.2, 0.3, 0.4, -0.5)
    blob = pack_vector(vec)
    out = unpack_vector(blob, len(vec))
    for a, b in zip(vec, out):
        assert abs(a - b) < 1e-6


def test_cosine_similarity_orthogonal_zero():
    assert abs(cosine_similarity((1.0, 0.0), (0.0, 1.0))) < 1e-9


def test_cosine_similarity_identical_one():
    assert abs(cosine_similarity((0.5, 0.5), (1.0, 1.0)) - 1.0) < 1e-9


def test_cosine_similarity_zero_norm_returns_zero():
    assert cosine_similarity((0.0, 0.0), (1.0, 1.0)) == 0.0


# ---- DiffEmbedder with stub client ----------------------------------


@dataclass
class _StubResponse:
    data: list[object]


@dataclass
class _StubData:
    embedding: list[float]


class _StubEmbeddings:
    def __init__(self, vector):
        self._vector = vector
        self.calls: list[dict] = []

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": input})
        return _StubResponse(data=[_StubData(embedding=list(self._vector))])


class _StubOpenAI:
    def __init__(self, vector):
        self.embeddings = _StubEmbeddings(vector)


def _patched_embedder(monkeypatch, vector):
    embedder = DiffEmbedder(model="text-embedding-3-small")
    stub = _StubOpenAI(vector)
    embedder._client = stub
    return embedder, stub


def test_embedder_returns_vector(monkeypatch):
    embedder, stub = _patched_embedder(monkeypatch, [0.1, 0.2, 0.3])
    out = embedder.embed("diff text")
    assert out is not None
    assert out.dim == 3
    assert out.vector == (0.1, 0.2, 0.3)
    assert stub.embeddings.calls[0]["input"] == "diff text"


def test_embedder_skips_empty_input(monkeypatch):
    embedder, stub = _patched_embedder(monkeypatch, [0.1])
    assert embedder.embed("") is None
    assert embedder.embed("   ") is None
    assert stub.embeddings.calls == []


def test_embedder_truncates_long_input(monkeypatch):
    embedder, stub = _patched_embedder(monkeypatch, [0.1])
    long_text = "x" * 50_000
    out = embedder.embed(long_text)
    assert out is not None
    assert len(out.diff_text) == 32_000
    assert stub.embeddings.calls[0]["input"] == out.diff_text


def test_embedder_returns_none_on_error(monkeypatch):
    embedder = DiffEmbedder(model="text-embedding-3-small")

    class _Boom:
        def create(self, **kw):
            raise RuntimeError("upstream timeout")

    embedder._client = type("_C", (), {"embeddings": _Boom()})()
    assert embedder.embed("x") is None


# ---- harness integration -------------------------------------------


def _candidate(tmp_path: Path, *, candidate_id: str = "cand_x") -> CandidateResult:
    result_path = tmp_path / "candidate_results" / f"{candidate_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "candidate": {
                    "candidate_id": candidate_id,
                    "scaffold_name": "memgpt_source",
                },
                "tasks": [
                    {"task_id": "t1", "passed": True, "score": 1.0},
                    {"task_id": "t2", "passed": False, "score": 0.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return CandidateResult(
        candidate_id=candidate_id,
        scaffold_name="memgpt_source",
        passrate=0.5,
        average_score=0.5,
        token_consuming=10,
        avg_token_consuming=5,
        avg_prompt_tokens=4,
        avg_completion_tokens=1,
        count=2,
        config={"top_k": 8},
        result_path=str(result_path),
    )


class _FixedEmbedder:
    """Deterministic 'embedder' for harness integration tests."""

    def __init__(self):
        self.model = "stub-model"
        self.calls: list[str] = []

    def embed(self, text: str) -> DiffEmbedding | None:
        self.calls.append(text)
        return DiffEmbedding(
            model=self.model,
            dim=4,
            diff_text=text,
            vector=(0.1, 0.2, 0.3, 0.4),
        )


def test_harness_records_diff_embedding(tmp_path):
    candidate = _candidate(tmp_path)
    diff_path = tmp_path / "proposer_calls" / "iter_001" / "diff.patch"
    diff_path.parent.mkdir(parents=True)
    diff_path.write_text(
        "diff --git a/x.py b/x.py\n+ change\n", encoding="utf-8"
    )

    embedder = _FixedEmbedder()
    harness = TraceHarness(
        run_dir=tmp_path,
        benchmark="longmemeval",
        diff_embedder=embedder,  # type: ignore[arg-type]
    )

    # iter_0 baseline first (no embedding because diff file absent).
    iter0 = _candidate(tmp_path, candidate_id="seed")
    harness.record_iteration(iteration=0, candidates=[iter0])
    # iter_1 should embed.
    harness.record_iteration(iteration=1, candidates=[candidate])

    assert embedder.calls == ["diff --git a/x.py b/x.py\n+ change\n"]
    with sqlite3.connect(tmp_path / "traces" / "index.db") as conn:
        row = conn.execute(
            "SELECT iteration, model, dim, diff_text FROM diff_embeddings"
        ).fetchone()
    assert row[0] == 1
    assert row[1] == "stub-model"
    assert row[2] == 4
    assert row[3].startswith("diff --git")


def test_harness_skips_embedding_when_no_embedder(tmp_path):
    candidate = _candidate(tmp_path)
    diff_path = tmp_path / "proposer_calls" / "iter_001" / "diff.patch"
    diff_path.parent.mkdir(parents=True)
    diff_path.write_text("diff --git a/x.py b/x.py\n", encoding="utf-8")

    harness = TraceHarness(run_dir=tmp_path, benchmark="longmemeval")
    iter0 = _candidate(tmp_path, candidate_id="seed")
    harness.record_iteration(iteration=0, candidates=[iter0])
    harness.record_iteration(iteration=1, candidates=[candidate])

    with sqlite3.connect(tmp_path / "traces" / "index.db") as conn:
        rows = conn.execute("SELECT COUNT(*) FROM diff_embeddings").fetchone()
    assert rows[0] == 0
