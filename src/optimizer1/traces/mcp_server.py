"""MCP server exposing trace-harness queries as first-class tools.

Run as ``python -m optimizer1.traces.mcp_server``. The optimizer
registers this server in ``<workspace>/.claude/settings.local.json``
so the Claude Code proposer can call the tools the same way it calls
built-in tools (Read, Grep, Bash, ...).

The DB path comes from the ``TRACE_DB`` environment variable (set by
the launcher). All tools are read-only.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .embeddings import DiffEmbedder, cosine_similarity, unpack_vector
from .query import TraceQuery


def _resolve_db_path() -> Path:
    raw = os.environ.get("TRACE_DB")
    if not raw:
        # Reasonable fallback for local development.
        return Path.cwd() / "traces" / "index.db"
    return Path(raw)


_DB_PATH = _resolve_db_path()
_QUERY: TraceQuery | None = None


def _query() -> TraceQuery:
    global _QUERY
    if _QUERY is None:
        _QUERY = TraceQuery(_DB_PATH)
    return _QUERY


mcp = FastMCP("optimizer1-traces")


@mcp.tool()
def trace_task_history(task_id: str) -> list[dict[str, Any]]:
    """Iter-ordered status history for one task across all candidates.

    Returns rows of (iteration, candidate_id, passed, score, status,
    baseline_score, delta).
    """

    return _query().task_history(task_id)


@mcp.tool()
def trace_persistent_failures(min_streak: int = 3) -> list[dict[str, Any]]:
    """Tasks whose trailing run of `persistent_fail` statuses is at
    least `min_streak` long. Use to spot tasks that have repeatedly
    resisted recent attempts."""

    return _query().persistent_failures(min_streak=min_streak)


@mcp.tool()
def trace_breakthroughs(since_iter: int = 0) -> list[dict[str, Any]]:
    """Traces with status='breakthrough' (baseline-fail → currently pass)
    at iteration >= since_iter. Use to see which directions have been
    paying off recently."""

    return _query().breakthroughs(since_iter=since_iter)


@mcp.tool()
def trace_regressions(window: int = 3) -> list[dict[str, Any]]:
    """Traces with status='regressed' (baseline-pass → currently fail)
    in the last `window` iterations. Use to detect which recent change
    broke things."""

    return _query().regressions(window=window)


@mcp.tool()
def trace_file_history(path: str) -> list[dict[str, Any]]:
    """Iterations in which the given source path appears in diff.patch,
    with that iteration's aggregated outcome (passrate + status counts)."""

    return _query().file_history(path)


@mcp.tool()
def trace_candidate_outcome(iteration: int, candidate_id: str) -> dict[str, Any]:
    """Full per-(iter, candidate) summary: trace count, passrate, mean
    score, jsonl pointer for drill-in, and modified files."""

    return _query().candidate_outcome(iteration, candidate_id)


@mcp.tool()
def trace_similar(diff_or_query: str, k: int = 5) -> list[dict[str, Any]]:
    """Find historical iterations whose diff embedding is most similar
    to the given text by cosine similarity.

    `diff_or_query` can be either an actual diff snippet you're
    considering, or a natural-language description of what you plan to
    change. The text is embedded with the same model used during
    indexing (typically ``text-embedding-3-small``).

    Returns rows of (iteration, similarity, model, status_counts)
    sorted by similarity descending. Excludes iterations whose embedding
    model differs from the query's.
    """

    if not diff_or_query.strip():
        return []
    rows = _load_embeddings(_DB_PATH)
    if not rows:
        return []
    # Reuse the model from existing rows so we don't drift across
    # different embedding spaces.
    primary_model = rows[0]["model"]
    embedder = DiffEmbedder(model=primary_model)
    query_emb = embedder.embed(diff_or_query)
    if query_emb is None:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        if row["model"] != primary_model:
            continue
        sim = cosine_similarity(query_emb.vector, row["vector"])
        iteration = row["iteration"]
        scored.append(
            (
                sim,
                {
                    "iteration": iteration,
                    "similarity": sim,
                    "model": row["model"],
                    "status_counts": _status_counts_for(_DB_PATH, iteration),
                },
            )
        )
    scored.sort(key=lambda item: -item[0])
    return [row for _, row in scored[: max(0, int(k))]]


def _load_embeddings(db_path: Path) -> list[dict[str, Any]]:
    sql = "SELECT iteration, model, dim, embedding FROM diff_embeddings"
    out: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        for iteration, model, dim, blob in conn.execute(sql):
            out.append(
                {
                    "iteration": int(iteration),
                    "model": str(model),
                    "dim": int(dim),
                    "vector": unpack_vector(bytes(blob), int(dim)),
                }
            )
    return out


def _status_counts_for(db_path: Path, iteration: int) -> dict[str, int]:
    sql = (
        "SELECT d.status, COUNT(*) FROM traces t "
        "JOIN diffs d USING (trace_id) WHERE t.iteration = ? GROUP BY d.status"
    )
    out: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        for status, count in conn.execute(sql, (iteration,)):
            out[str(status)] = int(count)
    return out


def main(argv: list[str] | None = None) -> int:
    # FastMCP runs over stdio by default; Claude Code launches us as a
    # subprocess and speaks JSON-RPC on stdin/stdout.
    mcp.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
