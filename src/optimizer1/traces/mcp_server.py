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
def trace_list_tasks(iteration: int | None = None) -> list[str]:
    """List distinct task_ids, optionally scoped to one iteration.

    With ``iteration=None`` returns the union of task_ids across all
    iterations. Pair with ``trace_task_history`` to walk a task's
    timeline, or with ``trace_compare_iterations`` to look at a
    pair-wise diff.
    """

    return _query().list_tasks(iteration=iteration)


@mcp.tool()
def trace_iteration_metadata(
    iters: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Return iteration_meta rows for ``iters`` (or all iterations).

    Each row carries: iteration, patch_base, budget, selection_policy,
    advanced_frontier, on_pareto_frontier, passrate, mean_score,
    proposer_call_dir. Filter / sort yourself to find e.g. the current
    pareto frontier (``on_pareto_frontier=True``) or recent stalled
    iters (``advanced_frontier=False``).
    """

    return _query().iteration_metadata(iters=iters)


@mcp.tool()
def trace_compare_iterations(
    left: int,
    right: int,
    left_candidate_id: str | None = None,
    right_candidate_id: str | None = None,
) -> list[dict[str, Any]]:
    """Per-task pass/fail comparison between any two iterations.

    Each row carries the per-side (passed, score, candidate_id) plus a
    ``classification`` (``regressed_RvL`` / ``breakthrough_RvL`` /
    ``stable_pass`` / ``both_fail`` / ``only_in_left`` / ``only_in_right``)
    and a ``delta`` (``right.score - left.score``). Output order is
    regressions first, then absent-on-one-side, then both-fail, then
    stable, then breakthroughs — surface what broke before what held.

    When ``left_candidate_id`` / ``right_candidate_id`` are None, each
    iter's highest-passrate candidate is picked (tie-break by
    candidate_id desc — same rule as the frontier). Pass explicit ids
    to compare specific candidates.

    Use this for any "vs baseline / vs frontier-best / vs patch base"
    comparison — the calling agent decides which two iters are
    meaningful.
    """

    return _query().compare_iterations(
        left,
        right,
        left_candidate_id=left_candidate_id,
        right_candidate_id=right_candidate_id,
    )


@mcp.tool()
def trace_file_history(path: str) -> list[dict[str, Any]]:
    """Iterations in which the given source path appears in diff.patch,
    with that iteration's aggregated outcome (passrate + status counts)."""

    return _query().file_history(path)


@mcp.tool()
def trace_candidate_outcome(
    iteration: int,
    candidate_id: str,
    max_examples: int = 8,
) -> dict[str, Any]:
    """Full per-(iter, candidate) summary: trace count, passrate, mean
    score, artifact pointers, modified files, and representative task
    examples.

    ``regressed_tasks`` / ``breakthrough_tasks`` / ``failed_tasks`` are
    capped by ``max_examples`` so callers can inspect examples without
    opening every trace JSONL row for 80-100 task batches.
    """

    return _query().candidate_outcome(
        iteration,
        candidate_id,
        max_examples=max_examples,
    )


@mcp.tool()
def trace_similar(diff_or_query: str, k: int = 5) -> list[dict[str, Any]]:
    """Find historical iterations whose diff is most similar to the
    given text by cosine similarity.

    ``diff_or_query`` can be either an actual diff snippet you're
    considering, or a natural-language description of what you plan to
    change.

    The embedding model is read from the ``DIFF_EMBEDDING_MODEL`` env
    var (default ``text-embedding-3-small``). On first call the tool
    lazily embeds every iter that has a stored ``diff_text`` but no
    cached embedding for the active model, writing the cache back to
    ``diff_embeddings``. Subsequent calls reuse the cache; the only
    per-call work is embedding the query.

    Returns rows of ``{iteration, similarity, model, status_counts}``
    sorted by similarity descending.
    """

    if not diff_or_query.strip():
        return []
    model = os.environ.get("DIFF_EMBEDDING_MODEL") or DiffEmbedder.DEFAULT_MODEL
    embedder = DiffEmbedder(model=model)
    _ensure_embeddings_for_model(_DB_PATH, model=model, embedder=embedder)
    rows = _load_embeddings_for_model(_DB_PATH, model=model)
    if not rows:
        return []
    query_emb = embedder.embed(diff_or_query)
    if query_emb is None:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        sim = cosine_similarity(query_emb.vector, row["vector"])
        iteration = row["iteration"]
        scored.append(
            (
                sim,
                {
                    "iteration": iteration,
                    "similarity": sim,
                    "model": model,
                    "status_counts": _status_counts_for(_DB_PATH, iteration),
                },
            )
        )
    scored.sort(key=lambda item: -item[0])
    return [row for _, row in scored[: max(0, int(k))]]


def _ensure_embeddings_for_model(
    db_path: Path,
    *,
    model: str,
    embedder: DiffEmbedder,
) -> None:
    """Embed every iter with stored diff_text but no cached embedding
    for ``model``, persisting results to ``diff_embeddings``.

    No-op when every iter is already cached. Failures from the embedder
    are non-fatal: we log via DiffEmbedder's own warning and skip.
    """

    missing = _find_missing_embeddings(db_path, model=model)
    if not missing:
        return
    from .indexer import Indexer

    indexer = Indexer(db_path)
    for iteration, diff_text in missing:
        emb = embedder.embed(diff_text)
        if emb is None:
            continue
        indexer.record_diff_embedding(
            iteration=iteration,
            model=emb.model,
            dim=emb.dim,
            embedding=emb.to_bytes(),
        )


def _find_missing_embeddings(
    db_path: Path,
    *,
    model: str,
) -> list[tuple[int, str]]:
    sql = """
        SELECT d.iteration, d.diff_text
        FROM iteration_diffs d
        WHERE NOT EXISTS (
            SELECT 1 FROM diff_embeddings e
            WHERE e.iteration = d.iteration AND e.model = ?
        )
        ORDER BY d.iteration ASC
    """
    with sqlite3.connect(db_path) as conn:
        return [
            (int(row[0]), str(row[1]))
            for row in conn.execute(sql, (model,))
        ]


def _load_embeddings_for_model(
    db_path: Path,
    *,
    model: str,
) -> list[dict[str, Any]]:
    sql = (
        "SELECT iteration, dim, embedding FROM diff_embeddings WHERE model = ?"
    )
    out: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        for iteration, dim, blob in conn.execute(sql, (model,)):
            out.append(
                {
                    "iteration": int(iteration),
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
