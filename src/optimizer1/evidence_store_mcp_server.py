"""MCP tools backed by the unified ``evidence_store.db``.

The tools are grouped by the three evidence-store layers:

* ``evidence_artifact_*`` reads raw artifacts.
* ``evidence_fact_*`` reads structured trace/eval/modification facts.
* ``evidence_link_*`` reads explicit evidence graph links.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


def _resolve_db_path() -> Path:
    raw = os.environ.get("EVIDENCE_DB")
    if raw:
        return Path(raw)
    return Path.cwd() / "evidence_store.db"


_DB_PATH = _resolve_db_path()


def _connect() -> sqlite3.Connection:
    if not _DB_PATH.exists():
        raise FileNotFoundError(f"EvidenceStore DB not found: {_DB_PATH}")
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


mcp = FastMCP("optimizer1-evidence")


# ---- raw artifact layer ------------------------------------------


@mcp.tool()
def evidence_artifact_list(
    iteration: int | None = None,
    kind: str | None = None,
    path_contains: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List raw artifacts by iteration/kind/path substring."""

    clauses: list[str] = []
    params: list[Any] = []
    if iteration is not None:
        clauses.append("iteration = ?")
        params.append(int(iteration))
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if path_contains:
        clauses.append("path LIKE ?")
        params.append(f"%{path_contains}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT artifact_id, iteration, candidate_id, kind, path, sha256,
                   byte_count, imported_at
            FROM artifacts
            {where}
            ORDER BY COALESCE(iteration, -1), kind, path
            LIMIT ?
            """,
            (*params, max(1, int(limit))),
        ).fetchall()
    return [_row(row) for row in rows]


@mcp.tool()
def evidence_artifact_get(
    artifact_id: str | None = None,
    path: str | None = None,
    max_chars: int = 12000,
) -> dict[str, Any] | None:
    """Fetch one raw artifact by id or exact/suffix path."""

    if not artifact_id and not path:
        return None
    with _connect() as conn:
        if artifact_id:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM artifacts
                WHERE path = ? OR path LIKE ?
                ORDER BY byte_count DESC
                LIMIT 1
                """,
                (path, f"%/{path}"),
            ).fetchone()
    if row is None:
        return None
    text, truncated = _artifact_text(row, max_chars=max_chars)
    out = _row_without_blob(row)
    out["text"] = text
    out["truncated"] = truncated
    return out


@mcp.tool()
def evidence_artifact_search(
    query: str,
    kind: str | None = None,
    iteration: int | None = None,
    limit: int = 20,
    max_chars_per_match: int = 1200,
) -> list[dict[str, Any]]:
    """Search raw artifact text with simple case-insensitive substring matching."""

    needle = query.strip().lower()
    if not needle:
        return []
    clauses: list[str] = []
    params: list[Any] = []
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if iteration is not None:
        clauses.append("iteration = ?")
        params.append(int(iteration))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM artifacts {where} ORDER BY COALESCE(iteration, -1), path",
            tuple(params),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        text, _ = _artifact_text(row, max_chars=0)
        idx = text.lower().find(needle)
        if idx < 0:
            continue
        start = max(0, idx - max_chars_per_match // 2)
        end = min(len(text), start + max(1, int(max_chars_per_match)))
        out.append(
            {
                "artifact_id": row["artifact_id"],
                "iteration": row["iteration"],
                "kind": row["kind"],
                "path": row["path"],
                "match_start": idx,
                "snippet": text[start:end],
            }
        )
        if len(out) >= max(1, int(limit)):
            break
    return out


# ---- structured fact layer ---------------------------------------


@mcp.tool()
def evidence_fact_state(as_of_iteration: int | None = None) -> dict[str, Any]:
    """Current structured state: frontier, latest iteration, and counts."""

    with _connect() as conn:
        as_of = _as_of_iteration(conn, as_of_iteration)
        latest = conn.execute(
            "SELECT * FROM iterations WHERE iteration <= ? ORDER BY iteration DESC LIMIT 1",
            (as_of,),
        ).fetchone()
        frontier = conn.execute(
            """
            SELECT * FROM frontier_members
            WHERE as_of_iteration = (
                SELECT MAX(as_of_iteration) FROM frontier_members
                WHERE as_of_iteration <= ?
            )
            ORDER BY passrate DESC, token_consuming ASC
            """,
            (as_of,),
        ).fetchall()
        counts = {
            table: conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE iteration <= ?",
                (as_of,),
            ).fetchone()[0]
            for table in ("candidates", "eval_results", "traces", "modifications")
        }
    return {
        "as_of_iteration": as_of,
        "latest_iteration": _row(latest) if latest else None,
        "frontier": [_row(row) for row in frontier],
        "counts": counts,
    }


@mcp.tool()
def evidence_fact_candidate_outcome(
    iteration: int,
    candidate_id: str | None = None,
    max_examples: int = 8,
    include_retrieval: bool = True,
) -> dict[str, Any] | None:
    """Candidate metrics plus representative failed tasks and trace ids."""

    with _connect() as conn:
        cand = _candidate(conn, iteration, candidate_id)
        if cand is None:
            return None
        cid = str(cand["candidate_id"])
        stats = conn.execute(
            """
            SELECT COUNT(*) AS task_count, AVG(passed) AS passrate,
                   AVG(score) AS average_score,
                   SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) AS passed_count,
                   SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) AS failed_count,
                   SUM(COALESCE(token_consuming, 0)) AS token_consuming
            FROM eval_results
            WHERE iteration = ? AND candidate_id = ?
            """,
            (int(iteration), cid),
        ).fetchone()
        examples = conn.execute(
            """
            SELECT er.*, t.summary_json, t.trace_id
            FROM eval_results er
            LEFT JOIN traces t ON t.trace_id = er.trace_id
            WHERE er.iteration = ? AND er.candidate_id = ? AND er.passed = 0
            ORDER BY er.score ASC, er.task_id ASC
            LIMIT ?
            """,
            (int(iteration), cid, max(0, int(max_examples))),
        ).fetchall()
    return {
        "iteration": int(iteration),
        "candidate": _candidate_row(cand),
        "stats": _row(stats),
        "failed_examples": [
            _example_row(row, include_retrieval=include_retrieval) for row in examples
        ],
    }


@mcp.tool()
def evidence_fact_compare_iterations(
    left: int,
    right: int,
    left_candidate_id: str | None = None,
    right_candidate_id: str | None = None,
    max_examples: int = 20,
    include_retrieval: bool = False,
) -> list[dict[str, Any]]:
    """Compare per-task outcomes for two iterations."""

    with _connect() as conn:
        left_cand = _candidate(conn, left, left_candidate_id)
        right_cand = _candidate(conn, right, right_candidate_id)
        if left_cand is None or right_cand is None:
            return []
        rows = _comparison_rows(
            conn,
            left=int(left),
            right=int(right),
            left_candidate_id=str(left_cand["candidate_id"]),
            right_candidate_id=str(right_cand["candidate_id"]),
            include_retrieval=include_retrieval,
        )
    rank = {
        "regressed_RvL": 0,
        "only_in_left": 1,
        "both_fail": 2,
        "only_in_right": 3,
        "stable_pass": 4,
        "breakthrough_RvL": 5,
    }
    rows.sort(key=lambda item: (rank.get(item["classification"], 99), item["task_id"]))
    return rows[: max(1, int(max_examples))]


@mcp.tool()
def evidence_fact_task_history(
    task_id: str,
    as_of_iteration: int | None = None,
    include_retrieval: bool = False,
) -> list[dict[str, Any]]:
    """Task result timeline across iterations."""

    with _connect() as conn:
        as_of = _as_of_iteration(conn, as_of_iteration)
        rows = conn.execute(
            """
            SELECT er.*, t.summary_json, t.trace_id
            FROM eval_results er
            LEFT JOIN traces t ON t.trace_id = er.trace_id
            WHERE er.task_id = ? AND er.iteration <= ?
            ORDER BY er.iteration ASC, er.candidate_id ASC
            """,
            (task_id, as_of),
        ).fetchall()
    return [_example_row(row, include_retrieval=include_retrieval) for row in rows]


@mcp.tool()
def evidence_fact_trace(
    trace_id: str,
    include_spans: bool = True,
    max_spans: int = 20,
) -> dict[str, Any] | None:
    """Fetch one trace and optionally its structured spans."""

    with _connect() as conn:
        row = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        if row is None:
            return None
        spans = []
        if include_spans:
            spans = [
                _span_row(item)
                for item in conn.execute(
                    """
                    SELECT * FROM trace_spans
                    WHERE trace_id = ?
                    ORDER BY span_ord ASC
                    LIMIT ?
                    """,
                    (trace_id, max(0, int(max_spans))),
                ).fetchall()
            ]
    out = _trace_row(row)
    out["spans"] = spans
    return out


@mcp.tool()
def evidence_fact_modification(
    iteration: int,
    include_diff: bool = True,
    max_diff_chars: int = 12000,
) -> dict[str, Any] | None:
    """Modification, proposal, changed files, and diff stats for one iteration."""

    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM modifications WHERE iteration = ?",
            (int(iteration),),
        ).fetchone()
        if row is None:
            return None
        changed = conn.execute(
            "SELECT path FROM modified_files WHERE iteration = ? ORDER BY path",
            (int(iteration),),
        ).fetchall()
    out = _row(row)
    out["files_changed"] = [item["path"] for item in changed]
    out["proposal"] = _loads(out.pop("proposal_json", None), {})
    out["files_changed_json"] = _loads(out.get("files_changed_json"), [])
    if not include_diff:
        out.pop("diff_text", None)
    elif out.get("diff_text") and len(out["diff_text"]) > max_diff_chars:
        out["diff_text"] = out["diff_text"][:max_diff_chars]
        out["diff_truncated"] = True
    return out


@mcp.tool()
def evidence_fact_proposer_call(iteration: int) -> dict[str, Any] | None:
    """Proposer return code, usage, metrics, and workspace/call paths."""

    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM proposer_calls WHERE iteration = ?",
            (int(iteration),),
        ).fetchone()
    if row is None:
        return None
    out = _row(row)
    out["metrics"] = _loads(out.pop("metrics_json", None), {})
    out["usage"] = _loads(out.pop("usage_json", None), None)
    out["extra"] = _loads(out.pop("extra_json", None), {})
    return out


@mcp.tool()
def evidence_fact_file_history(path: str, limit: int = 50) -> list[dict[str, Any]]:
    """Read/write/grep/change history for a path or path suffix."""

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM tool_accesses
            WHERE path = ? OR path LIKE ?
            ORDER BY iteration ASC, access_type ASC
            LIMIT ?
            """,
            (path, f"%{path}", max(1, int(limit))),
        ).fetchall()
    return [_row(row) for row in rows]


# ---- evidence link layer -----------------------------------------


@mcp.tool()
def evidence_link_for(
    source_type: str | None = None,
    source_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    relation: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List evidence graph links by source/target/relation filters."""

    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("source_type", source_type),
        ("source_id", source_id),
        ("target_type", target_type),
        ("target_id", target_id),
        ("relation", relation),
    ):
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM evidence_links
            {where}
            ORDER BY created_at ASC, relation ASC
            LIMIT ?
            """,
            (*params, max(1, int(limit))),
        ).fetchall()
    return [_row(row) for row in rows]


@mcp.tool()
def evidence_link_explain_iteration(
    iteration: int,
    include_diff: bool = False,
    include_examples: bool = True,
) -> dict[str, Any]:
    """One iteration's proposer call, modification, outcome, and links."""

    mod_id = _mod_id(iteration)
    call_id = _call_id(iteration)
    with _connect() as conn:
        links = conn.execute(
            """
            SELECT * FROM evidence_links
            WHERE source_id IN (?, ?) OR target_id IN (?, ?)
            ORDER BY relation ASC
            """,
            (mod_id, call_id, mod_id, call_id),
        ).fetchall()
    return {
        "iteration": int(iteration),
        "proposer_call": evidence_fact_proposer_call(iteration),
        "modification": evidence_fact_modification(
            iteration,
            include_diff=include_diff,
        ),
        "candidate_outcome": (
            evidence_fact_candidate_outcome(iteration, max_examples=5)
            if include_examples
            else None
        ),
        "links": [_row(row) for row in links],
    }


@mcp.tool()
def evidence_link_chain_task(
    task_id: str,
    as_of_iteration: int | None = None,
    include_retrieval: bool = False,
) -> dict[str, Any]:
    """Task-centered chain: eval results -> traces -> producing modifications."""

    history = evidence_fact_task_history(
        task_id,
        as_of_iteration=as_of_iteration,
        include_retrieval=include_retrieval,
    )
    trace_ids = [item.get("trace_id") for item in history if item.get("trace_id")]
    with _connect() as conn:
        links = []
        for trace_id in trace_ids:
            links.extend(
                _row(row)
                for row in conn.execute(
                    """
                    SELECT * FROM evidence_links
                    WHERE source_id = ? OR target_id = ?
                    ORDER BY relation
                    """,
                    (trace_id, trace_id),
                ).fetchall()
            )
    return {"task_id": task_id, "history": history, "links": links}


def _as_of_iteration(conn: sqlite3.Connection, value: int | None) -> int:
    if value is not None:
        return int(value)
    row = conn.execute("SELECT MAX(iteration) FROM iterations").fetchone()
    return int(row[0] or 0)


def _candidate(
    conn: sqlite3.Connection,
    iteration: int,
    candidate_id: str | None,
) -> sqlite3.Row | None:
    if candidate_id:
        return conn.execute(
            "SELECT * FROM candidates WHERE iteration = ? AND candidate_id = ?",
            (int(iteration), candidate_id),
        ).fetchone()
    return conn.execute(
        """
        SELECT * FROM candidates
        WHERE iteration = ?
        ORDER BY passrate DESC, average_score DESC, candidate_id DESC
        LIMIT 1
        """,
        (int(iteration),),
    ).fetchone()


def _comparison_rows(
    conn: sqlite3.Connection,
    *,
    left: int,
    right: int,
    left_candidate_id: str,
    right_candidate_id: str,
    include_retrieval: bool,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for side, iteration, candidate_id in (
        ("left", left, left_candidate_id),
        ("right", right, right_candidate_id),
    ):
        for row in conn.execute(
            """
            SELECT er.*, t.summary_json, t.trace_id
            FROM eval_results er
            LEFT JOIN traces t ON t.trace_id = er.trace_id
            WHERE er.iteration = ? AND er.candidate_id = ?
            """,
            (iteration, candidate_id),
        ).fetchall():
            entry = rows.setdefault(
                str(row["task_id"]),
                {"task_id": str(row["task_id"]), "left": None, "right": None},
            )
            entry[side] = _example_row(row, include_retrieval=include_retrieval)
    out = []
    for entry in rows.values():
        classification, delta = _classify_pair(entry["left"], entry["right"])
        entry["classification"] = classification
        entry["delta"] = delta
        out.append(entry)
    return out


def _classify_pair(left: dict[str, Any] | None, right: dict[str, Any] | None) -> tuple[str, float | None]:
    if left is None:
        return "only_in_right", None
    if right is None:
        return "only_in_left", None
    delta = None
    if left.get("score") is not None and right.get("score") is not None:
        delta = float(right["score"]) - float(left["score"])
    lp = bool(left.get("passed"))
    rp = bool(right.get("passed"))
    if lp and not rp:
        return "regressed_RvL", delta
    if not lp and rp:
        return "breakthrough_RvL", delta
    if lp and rp:
        return "stable_pass", delta
    return "both_fail", delta


def _candidate_row(row: sqlite3.Row) -> dict[str, Any]:
    out = _row(row)
    out["config"] = _loads(out.pop("config_json", None), {})
    out["proposal"] = _loads(out.pop("proposal_json", None), {})
    return out


def _example_row(row: sqlite3.Row, *, include_retrieval: bool) -> dict[str, Any]:
    out = _row(row)
    summary = _loads(out.pop("summary_json", None), {})
    out["question"] = summary.get("question")
    out["gold"] = summary.get("gold")
    out["prediction"] = summary.get("prediction")
    if include_retrieval:
        out["retrieval"] = _retrieval_from_summary(summary)
    return out


def _trace_row(row: sqlite3.Row) -> dict[str, Any]:
    out = _row(row)
    out["summary"] = _loads(out.pop("summary_json", None), {})
    out["diff"] = _loads(out.pop("diff_json", None), None)
    return out


def _span_row(row: sqlite3.Row) -> dict[str, Any]:
    out = _row(row)
    out["input"] = _loads(out.pop("input_json", None), None)
    out["output"] = _loads(out.pop("output_json", None), None)
    out["metadata"] = _loads(out.pop("metadata_json", None), {})
    return out


def _retrieval_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    retrieved = summary.get("retrieved")
    if isinstance(retrieved, list):
        return [item for item in retrieved if isinstance(item, dict)]
    return []


def _artifact_text(row: sqlite3.Row, *, max_chars: int) -> tuple[str, bool]:
    data = row["content_blob"]
    encoding = row["content_encoding"]
    preview_only = encoding.endswith("+preview")
    if encoding in {"gzip", "gzip+preview"}:
        data = gzip.decompress(data)
    text = bytes(data).decode("utf-8", errors="replace")
    if max_chars and len(text) > max_chars:
        return text[:max_chars], True
    return text, preview_only


def _row(row: sqlite3.Row | None) -> dict[str, Any]:
    return {} if row is None else {key: row[key] for key in row.keys()}


def _row_without_blob(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys() if key != "content_blob"}


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _mod_id(iteration: int) -> str:
    return f"{_run_id()}:mod:{int(iteration):03d}"


def _call_id(iteration: int) -> str:
    return f"{_run_id()}:call:{int(iteration):03d}"


def _run_id() -> str:
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = 'run_id'"
            ).fetchone()
            if row and row["value"]:
                return str(row["value"])
    except Exception:  # noqa: BLE001 - fallback only
        pass
    return _DB_PATH.parent.name


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
