"""MCP tools backed by ``runs/<run>/runstore.db``.

The tool names intentionally keep evaluation trace queries separate from
modification-history queries:

* ``runstore_trace_*`` tools read task/candidate/frontier evidence.
* ``runstore_mod_*`` tools read proposer/diff/file-access evidence.

Both groups read the same RunStore database, but they answer different
classes of questions for the proposer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from optimizer1.run_store import RunStore


def _resolve_db_path() -> Path:
    raw = os.environ.get("RUNSTORE_DB")
    if raw:
        return Path(raw)
    return Path.cwd() / "runstore.db"


_DB_PATH = _resolve_db_path()
_STORE: RunStore | None = None


def _store() -> RunStore:
    global _STORE
    if _STORE is None:
        if not _DB_PATH.exists():
            raise FileNotFoundError(f"RunStore DB not found: {_DB_PATH}")
        _STORE = RunStore(_DB_PATH.parent, initialize=False)
    return _STORE


mcp = FastMCP("optimizer1-runstore")


# ---- trace / evaluation evidence ---------------------------------


@mcp.tool()
def runstore_trace_iteration_meta(
    iteration: int | None = None,
    as_of_iteration: int | None = None,
) -> list[dict[str, Any]]:
    """Iteration lifecycle rows, optionally scoped by iteration/as_of."""

    return _store().iteration_meta(
        iteration=iteration,
        as_of_iteration=as_of_iteration,
    )


@mcp.tool()
def runstore_trace_candidate_outcome(
    iteration: int,
    candidate_id: str | None = None,
    as_of_iteration: int | None = None,
    max_examples: int = 8,
) -> dict[str, Any] | None:
    """Candidate metrics and representative failed task examples."""

    return _store().candidate_outcome(
        iteration,
        candidate_id,
        as_of_iteration=as_of_iteration,
        max_examples=max_examples,
    )


@mcp.tool()
def runstore_trace_compare_iterations(
    left: int,
    right: int,
    left_candidate_id: str | None = None,
    right_candidate_id: str | None = None,
    as_of_iteration: int | None = None,
) -> list[dict[str, Any]]:
    """Per-task pass/fail comparison between two iterations."""

    return _store().compare_iterations(
        left,
        right,
        left_candidate_id=left_candidate_id,
        right_candidate_id=right_candidate_id,
        as_of_iteration=as_of_iteration,
    )


@mcp.tool()
def runstore_trace_task_history(
    task_id: str,
    as_of_iteration: int | None = None,
) -> list[dict[str, Any]]:
    """Task result timeline across candidates and iterations."""

    return _store().task_history(task_id, as_of_iteration=as_of_iteration)


@mcp.tool()
def runstore_trace_frontier(
    as_of_iteration: int | None = None,
) -> list[dict[str, Any]]:
    """Latest frontier snapshot at or before ``as_of_iteration``."""

    return _store().frontier(as_of_iteration=as_of_iteration)


# ---- modification evidence ---------------------------------------


@mcp.tool()
def runstore_mod_iteration_diff(
    iteration: int,
    as_of_iteration: int | None = None,
) -> dict[str, Any] | None:
    """Raw patch and diff stats for one iteration."""

    return _store().iteration_diff(
        iteration,
        as_of_iteration=as_of_iteration,
    )


@mcp.tool()
def runstore_mod_proposer_call(
    iteration: int,
    as_of_iteration: int | None = None,
) -> dict[str, Any] | None:
    """Proposer return code, metrics, usage, and call/workspace metadata."""

    return _store().proposer_call(
        iteration,
        as_of_iteration=as_of_iteration,
    )


@mcp.tool()
def runstore_mod_file_history(
    path: str,
    as_of_iteration: int | None = None,
) -> list[dict[str, Any]]:
    """Read/write/grep/changed history for a source path."""

    return _store().modification_file_history(
        path,
        as_of_iteration=as_of_iteration,
    )


@mcp.tool()
def runstore_mod_similar_changes(
    text_or_diff: str,
    as_of_iteration: int | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Lexical similar-change search over historical diffs.

    This is deterministic and local. Embedding-based similarity can be
    layered in later without changing the tool contract.
    """

    return _store().similar_changes(
        text_or_diff,
        as_of_iteration=as_of_iteration,
        k=k,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
