"""Structured per-run SQLite store.

This is the first migration step away from summary-driven optimizer state:
the existing writers keep producing their files, while ``RunStore`` mirrors
the same facts into ``runs/<run_id>/runstore.db`` with idempotent upserts.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from optimizer1.schemas import CandidateResult


SCHEMA_VERSION = "optimizer1.runstore.v1"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS iterations (
    iteration INTEGER PRIMARY KEY,
    status TEXT,
    as_of_iteration INTEGER,
    base_iteration INTEGER,
    base_candidate_id TEXT,
    created_at TEXT,
    committed_at TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    iteration INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    scaffold_name TEXT,
    passrate REAL,
    average_score REAL,
    token_consuming INTEGER,
    result_path TEXT,
    config_json TEXT,
    proposal_json TEXT,
    PRIMARY KEY (iteration, candidate_id)
);

CREATE TABLE IF NOT EXISTS proposer_calls (
    iteration INTEGER PRIMARY KEY,
    returncode INTEGER,
    timed_out INTEGER,
    call_dir TEXT,
    workspace_dir TEXT,
    metrics_json TEXT,
    usage_json TEXT,
    selection_policy TEXT,
    proposer_agent TEXT,
    extra_json TEXT
);

CREATE TABLE IF NOT EXISTS iteration_diffs (
    iteration INTEGER PRIMARY KEY,
    diff_text TEXT,
    files_changed_json TEXT,
    insertions INTEGER,
    deletions INTEGER
);

CREATE TABLE IF NOT EXISTS task_results (
    iteration INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    score REAL,
    passed INTEGER,
    token_consuming INTEGER,
    trace_path TEXT,
    PRIMARY KEY (iteration, candidate_id, task_id)
);

CREATE TABLE IF NOT EXISTS file_accesses (
    iteration INTEGER NOT NULL,
    path TEXT NOT NULL,
    access_type TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    read_calls INTEGER,
    read_lines INTEGER,
    line_ranges_json TEXT,
    PRIMARY KEY (iteration, path, access_type, tool_name)
);

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    iteration INTEGER,
    candidate_id TEXT,
    hypothesis TEXT,
    expected_effect_json TEXT,
    risk_json TEXT,
    evidence_refs_json TEXT
);

CREATE TABLE IF NOT EXISTS proposal_outcomes (
    proposal_id TEXT PRIMARY KEY,
    passrate_delta REAL,
    token_delta INTEGER,
    breakthrough_count INTEGER,
    regression_count INTEGER,
    outcome_summary TEXT
);

CREATE TABLE IF NOT EXISTS frontier_members (
    as_of_iteration INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    role TEXT,
    passrate REAL,
    token_consuming INTEGER,
    PRIMARY KEY (as_of_iteration, iteration, candidate_id)
);

CREATE TABLE IF NOT EXISTS state_snapshots (
    iteration INTEGER PRIMARY KEY,
    state_md TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class RunStore:
    """Idempotent structured store for one optimizer run."""

    def __init__(
        self,
        run_dir: Path,
        *,
        benchmark: str | None = None,
        initialize: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.db_path = self.run_dir / "runstore.db"
        self.benchmark = benchmark
        if initialize:
            self._ensure_schema()
            if benchmark:
                self.set_metadata("benchmark", benchmark)
            self.set_metadata("schema", SCHEMA_VERSION)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def set_metadata(self, key: str, value: object) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                (str(key), "" if value is None else str(value)),
            )

    def begin_iteration(
        self,
        iteration: int,
        *,
        as_of_iteration: int | None = None,
        base_iteration: int | None = None,
        base_candidate_id: str | None = None,
        status: str = "running",
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO iterations (
                    iteration, status, as_of_iteration, base_iteration,
                    base_candidate_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(iteration) DO UPDATE SET
                    status = excluded.status,
                    as_of_iteration = COALESCE(excluded.as_of_iteration, iterations.as_of_iteration),
                    base_iteration = COALESCE(excluded.base_iteration, iterations.base_iteration),
                    base_candidate_id = COALESCE(excluded.base_candidate_id, iterations.base_candidate_id)
                """,
                (
                    int(iteration),
                    status,
                    None if as_of_iteration is None else int(as_of_iteration),
                    None if base_iteration is None else int(base_iteration),
                    base_candidate_id,
                    now,
                ),
            )

    def commit_iteration(self, iteration: int, *, status: str = "committed") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO iterations(iteration, status, created_at, committed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(iteration) DO UPDATE SET
                    status = excluded.status,
                    committed_at = excluded.committed_at
                """,
                (int(iteration), status, _utc_now(), _utc_now()),
            )

    def record_candidates(
        self,
        iteration: int,
        candidates: Iterable[CandidateResult],
        *,
        proposals_by_candidate: dict[str, Any] | None = None,
    ) -> None:
        rows = []
        for candidate in candidates:
            proposal = (
                proposals_by_candidate.get(candidate.candidate_id)
                if proposals_by_candidate
                else None
            )
            rows.append(
                (
                    int(iteration),
                    candidate.candidate_id,
                    candidate.scaffold_name,
                    float(candidate.passrate),
                    float(candidate.average_score),
                    int(candidate.token_consuming),
                    candidate.result_path,
                    _json(candidate.config),
                    _json(proposal or {}),
                )
            )
        if not rows:
            return
        self.begin_iteration(iteration, status="recorded")
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO candidates (
                    iteration, candidate_id, scaffold_name, passrate,
                    average_score, token_consuming, result_path, config_json,
                    proposal_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def record_eval(self, iteration: int, candidates: Iterable[CandidateResult]) -> None:
        rows = []
        for candidate in candidates:
            payload = _read_json(Path(candidate.result_path))
            tasks = payload.get("tasks") if isinstance(payload, dict) else None
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                task_id = str(task.get("task_id") or "")
                if not task_id:
                    continue
                rows.append(
                    (
                        int(iteration),
                        candidate.candidate_id,
                        task_id,
                        _float_or_none(task.get("score")),
                        1 if bool(task.get("passed", False)) else 0,
                        _task_tokens(task),
                        candidate.result_path,
                    )
                )
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO task_results (
                    iteration, candidate_id, task_id, score, passed,
                    token_consuming, trace_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def record_proposer_call(
        self,
        iteration: int,
        *,
        result: Any,
        selection_policy: str,
        proposer_agent: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        extra_payload = dict(extra or {})
        tool_access = getattr(result, "tool_access", {}) or {}
        if not isinstance(tool_access, dict):
            tool_access = {}
        call_dir = extra_payload.get("call_dir")
        workspace_dir = extra_payload.get("workspace_dir")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO proposer_calls (
                    iteration, returncode, timed_out, call_dir, workspace_dir,
                    metrics_json, usage_json, selection_policy, proposer_agent,
                    extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(iteration),
                    _int_or_none(getattr(result, "returncode", None)),
                    1 if bool(getattr(result, "timed_out", False)) else 0,
                    None if call_dir is None else str(call_dir),
                    None if workspace_dir is None else str(workspace_dir),
                    _json(getattr(result, "metrics", {}) or {}),
                    _json(getattr(result, "usage", None)),
                    selection_policy,
                    proposer_agent,
                    _json(extra_payload),
                ),
            )
        self._record_tool_access(iteration, tool_access)

    def record_diff(self, iteration: int, diff_text: str) -> None:
        stats = diff_stats(diff_text)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO iteration_diffs (
                    iteration, diff_text, files_changed_json, insertions, deletions
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(iteration),
                    diff_text,
                    _json(stats["files_changed"]),
                    int(stats["insertions"]),
                    int(stats["deletions"]),
                ),
            )
        self.record_changed_files(iteration, stats["files_changed"])

    def record_changed_files(self, iteration: int, paths: Iterable[str]) -> None:
        rows = [
            (int(iteration), str(path), "changed", "diff", 0, 0, _json([]))
            for path in sorted({str(path) for path in paths if str(path)})
        ]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO file_accesses (
                    iteration, path, access_type, tool_name, read_calls,
                    read_lines, line_ranges_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def update_frontier(
        self,
        *,
        as_of_iteration: int,
        candidates: Iterable[CandidateResult],
        frontier: Iterable[CandidateResult],
    ) -> None:
        all_candidates = list(candidates)
        frontier_by_id = {item.candidate_id: item for item in frontier}
        if not all_candidates:
            return
        highest = max(all_candidates, key=lambda item: (item.passrate, item.candidate_id))
        lowest_token = min(
            all_candidates,
            key=lambda item: (item.token_consuming, -item.passrate),
        )
        rows = []
        for candidate in frontier_by_id.values():
            iteration = _candidate_iteration(candidate.candidate_id)
            if iteration is None:
                continue
            role = "balanced"
            if candidate.candidate_id == highest.candidate_id:
                role = "highest_passrate"
            elif candidate.candidate_id == lowest_token.candidate_id:
                role = "low_token"
            rows.append(
                (
                    int(as_of_iteration),
                    int(iteration),
                    candidate.candidate_id,
                    role,
                    float(candidate.passrate),
                    int(candidate.token_consuming),
                )
            )
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM frontier_members WHERE as_of_iteration = ?",
                (int(as_of_iteration),),
            )
            if rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO frontier_members (
                        as_of_iteration, iteration, candidate_id, role,
                        passrate, token_consuming
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def iteration_diff(
        self,
        iteration: int,
        *,
        as_of_iteration: int | None = None,
    ) -> dict[str, Any] | None:
        if as_of_iteration is not None and int(iteration) > int(as_of_iteration):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM iteration_diffs WHERE iteration = ?",
                (int(iteration),),
            ).fetchone()
        if row is None:
            return None
        return {
            "iteration": int(row["iteration"]),
            "diff_text": row["diff_text"] or "",
            "files_changed": _loads(row["files_changed_json"], []),
            "insertions": int(row["insertions"] or 0),
            "deletions": int(row["deletions"] or 0),
        }

    def proposer_call(
        self,
        iteration: int,
        *,
        as_of_iteration: int | None = None,
    ) -> dict[str, Any] | None:
        if as_of_iteration is not None and int(iteration) > int(as_of_iteration):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM proposer_calls WHERE iteration = ?",
                (int(iteration),),
            ).fetchone()
        if row is None:
            return None
        return {
            "iteration": int(row["iteration"]),
            "returncode": row["returncode"],
            "timed_out": bool(row["timed_out"]),
            "call_dir": row["call_dir"],
            "workspace_dir": row["workspace_dir"],
            "metrics": _loads(row["metrics_json"], {}),
            "usage": _loads(row["usage_json"], None),
            "selection_policy": row["selection_policy"],
            "proposer_agent": row["proposer_agent"],
            "extra": _loads(row["extra_json"], {}),
        }

    def iteration_meta(
        self,
        iteration: int | None = None,
        *,
        as_of_iteration: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[object] = []
        if iteration is not None:
            clauses.append("iteration = ?")
            params.append(int(iteration))
        if as_of_iteration is not None:
            clauses.append("iteration <= ?")
            params.append(int(as_of_iteration))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM iterations {where} ORDER BY iteration ASC",
                tuple(params),
            ).fetchall()
        return [
            {
                "iteration": int(row["iteration"]),
                "status": row["status"],
                "as_of_iteration": _int_or_none(row["as_of_iteration"]),
                "base_iteration": _int_or_none(row["base_iteration"]),
                "base_candidate_id": row["base_candidate_id"],
                "created_at": row["created_at"],
                "committed_at": row["committed_at"],
            }
            for row in rows
        ]

    def candidate_outcome(
        self,
        iteration: int,
        candidate_id: str | None = None,
        *,
        as_of_iteration: int | None = None,
        max_examples: int = 8,
    ) -> dict[str, Any] | None:
        if as_of_iteration is not None and int(iteration) > int(as_of_iteration):
            return None
        chosen = candidate_id or self._headline_candidate(iteration)
        if chosen is None:
            return None
        with self._connect() as conn:
            cand = conn.execute(
                """
                SELECT * FROM candidates
                WHERE iteration = ? AND candidate_id = ?
                """,
                (int(iteration), chosen),
            ).fetchone()
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) AS task_count,
                    AVG(passed) AS passrate,
                    AVG(score) AS average_score,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) AS passed_count,
                    SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) AS failed_count,
                    SUM(COALESCE(token_consuming, 0)) AS token_consuming
                FROM task_results
                WHERE iteration = ? AND candidate_id = ?
                """,
                (int(iteration), chosen),
            ).fetchone()
            failures = conn.execute(
                """
                SELECT iteration, candidate_id, task_id, score, passed,
                       token_consuming, trace_path
                FROM task_results
                WHERE iteration = ? AND candidate_id = ? AND passed = 0
                ORDER BY score ASC, task_id ASC
                LIMIT ?
                """,
                (int(iteration), chosen, max(0, int(max_examples))),
            ).fetchall()
        if cand is None and int(stats["task_count"] or 0) == 0:
            return None
        return {
            "iteration": int(iteration),
            "candidate_id": chosen,
            "candidate": _candidate_row(cand) if cand is not None else None,
            "task_count": int(stats["task_count"] or 0),
            "passrate": _float_or_none(stats["passrate"]),
            "average_score": _float_or_none(stats["average_score"]),
            "passed_count": int(stats["passed_count"] or 0),
            "failed_count": int(stats["failed_count"] or 0),
            "task_token_consuming": int(stats["token_consuming"] or 0),
            "failed_tasks": [_task_row(row) for row in failures],
        }

    def task_history(
        self,
        task_id: str,
        *,
        as_of_iteration: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["task_id = ?"]
        params: list[object] = [task_id]
        if as_of_iteration is not None:
            clauses.append("iteration <= ?")
            params.append(int(as_of_iteration))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM task_results
                WHERE {' AND '.join(clauses)}
                ORDER BY iteration ASC, candidate_id ASC
                """,
                tuple(params),
            ).fetchall()
        return [_task_row(row) for row in rows]

    def compare_iterations(
        self,
        left: int,
        right: int,
        *,
        left_candidate_id: str | None = None,
        right_candidate_id: str | None = None,
        as_of_iteration: int | None = None,
    ) -> list[dict[str, Any]]:
        if as_of_iteration is not None and (
            int(left) > int(as_of_iteration) or int(right) > int(as_of_iteration)
        ):
            return []
        left_cand = left_candidate_id or self._headline_candidate(left)
        right_cand = right_candidate_id or self._headline_candidate(right)
        if left_cand is None or right_cand is None:
            return []
        rows: dict[str, dict[str, Any]] = {}
        with self._connect() as conn:
            for side, iteration, candidate_id in (
                ("left", left, left_cand),
                ("right", right, right_cand),
            ):
                for row in conn.execute(
                    """
                    SELECT task_id, score, passed, token_consuming, trace_path
                    FROM task_results
                    WHERE iteration = ? AND candidate_id = ?
                    """,
                    (int(iteration), candidate_id),
                ).fetchall():
                    entry = rows.setdefault(
                        str(row["task_id"]),
                        {"task_id": str(row["task_id"]), "left": None, "right": None},
                    )
                    entry[side] = {
                        "iteration": int(iteration),
                        "candidate_id": candidate_id,
                        "score": _float_or_none(row["score"]),
                        "passed": bool(row["passed"]),
                        "token_consuming": _int_or_none(row["token_consuming"]),
                        "trace_path": row["trace_path"],
                    }
        out = []
        for entry in rows.values():
            classification, delta = _classify_pair(entry["left"], entry["right"])
            entry["classification"] = classification
            entry["delta"] = delta
            out.append(entry)
        rank = {
            "regressed_RvL": 0,
            "only_in_left": 1,
            "both_fail": 2,
            "only_in_right": 3,
            "stable_pass": 4,
            "breakthrough_RvL": 5,
        }
        out.sort(key=lambda item: (rank.get(item["classification"], 99), item["task_id"]))
        return out

    def frontier(self, *, as_of_iteration: int | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if as_of_iteration is None:
                row = conn.execute(
                    "SELECT MAX(as_of_iteration) AS as_of_iteration FROM frontier_members"
                ).fetchone()
                chosen_as_of = _int_or_none(row["as_of_iteration"]) if row else None
            else:
                row = conn.execute(
                    """
                    SELECT MAX(as_of_iteration) AS as_of_iteration
                    FROM frontier_members
                    WHERE as_of_iteration <= ?
                    """,
                    (int(as_of_iteration),),
                ).fetchone()
                chosen_as_of = _int_or_none(row["as_of_iteration"]) if row else None
            if chosen_as_of is None:
                return []
            rows = conn.execute(
                """
                SELECT *
                FROM frontier_members
                WHERE as_of_iteration = ?
                ORDER BY role ASC, passrate DESC, token_consuming ASC, candidate_id ASC
                """,
                (chosen_as_of,),
            ).fetchall()
        return [
            {
                "as_of_iteration": int(row["as_of_iteration"]),
                "iteration": int(row["iteration"]),
                "candidate_id": row["candidate_id"],
                "role": row["role"],
                "passrate": _float_or_none(row["passrate"]),
                "token_consuming": _int_or_none(row["token_consuming"]),
            }
            for row in rows
        ]

    def modification_file_history(
        self,
        path: str,
        *,
        as_of_iteration: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["fa.path = ?"]
        params: list[object] = [path]
        if as_of_iteration is not None:
            clauses.append("fa.iteration <= ?")
            params.append(int(as_of_iteration))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    fa.iteration AS iteration,
                    fa.access_type AS access_type,
                    fa.tool_name AS tool_name,
                    fa.read_calls AS read_calls,
                    fa.read_lines AS read_lines,
                    fa.line_ranges_json AS line_ranges_json,
                    c.candidate_id AS candidate_id,
                    c.passrate AS passrate,
                    c.token_consuming AS token_consuming
                FROM file_accesses fa
                LEFT JOIN candidates c ON c.iteration = fa.iteration
                WHERE {' AND '.join(clauses)}
                ORDER BY fa.iteration ASC, fa.access_type ASC, fa.tool_name ASC
                """,
                tuple(params),
            ).fetchall()
        return [
            {
                "iteration": int(row["iteration"]),
                "path": path,
                "access_type": row["access_type"],
                "tool_name": row["tool_name"],
                "read_calls": int(row["read_calls"] or 0),
                "read_lines": int(row["read_lines"] or 0),
                "line_ranges": _loads(row["line_ranges_json"], []),
                "candidate_id": row["candidate_id"],
                "passrate": _float_or_none(row["passrate"]),
                "token_consuming": _int_or_none(row["token_consuming"]),
            }
            for row in rows
        ]

    def similar_changes(
        self,
        text_or_diff: str,
        *,
        as_of_iteration: int | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        query_terms = _terms(text_or_diff)
        if not query_terms:
            return []
        clauses: list[str] = []
        params: list[object] = []
        if as_of_iteration is not None:
            clauses.append("iteration <= ?")
            params.append(int(as_of_iteration))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM iteration_diffs {where} ORDER BY iteration ASC",
                tuple(params),
            ).fetchall()
        scored = []
        for row in rows:
            diff_text = row["diff_text"] or ""
            diff_terms = _terms(diff_text)
            if not diff_terms:
                continue
            overlap = len(query_terms & diff_terms)
            if overlap <= 0:
                continue
            score = overlap / max(1, len(query_terms | diff_terms))
            scored.append(
                {
                    "iteration": int(row["iteration"]),
                    "similarity": score,
                    "files_changed": _loads(row["files_changed_json"], []),
                    "insertions": int(row["insertions"] or 0),
                    "deletions": int(row["deletions"] or 0),
                }
            )
        scored.sort(key=lambda item: (-item["similarity"], item["iteration"]))
        return scored[: max(0, int(k))]

    def _headline_candidate(self, iteration: int) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT candidate_id
                FROM candidates
                WHERE iteration = ?
                ORDER BY passrate DESC, average_score DESC, candidate_id DESC
                LIMIT 1
                """,
                (int(iteration),),
            ).fetchone()
        return None if row is None else str(row["candidate_id"])

    def render_state_md(
        self,
        *,
        iteration: int,
        as_of_iteration: int | None = None,
        benchmark: str | None = None,
        base_iteration: int | None = None,
        base_candidate_id: str | None = None,
    ) -> str:
        del base_iteration, base_candidate_id
        as_of = int(as_of_iteration if as_of_iteration is not None else iteration - 1)
        frontier = self._top_passrate_frontier(as_of_iteration=as_of, limit=5)
        benchmark_value = benchmark or self._metadata("benchmark") or "unknown"
        lines = [
            "# Optimizer State",
            "",
            "schema: optimizer1.state.v1",
            f"current_iteration: {int(iteration)}",
            f"as_of_iteration: {as_of}",
            f"benchmark: {benchmark_value}",
            "",
            "## Objective",
            "",
            "primary: maximize passrate",
            "",
            "## Frontier",
            "",
            "| rank | iteration | candidate_id | passrate | average_score | token_consuming |",
            "|---:|---:|---|---:|---:|---:|",
        ]
        if frontier:
            for rank, row in enumerate(frontier, start=1):
                lines.append(
                    f"| {rank} | {row['iteration']} | {row['candidate_id']} | "
                    f"{_value(row['passrate'])} | {_value(row['average_score'])} | "
                    f"{_value(row['token_consuming'])} |"
                )
        else:
            lines.append("|  |  | none |  |  |  |")
        lines.extend(
            [
                "",
                "## Available Evidence Tools",
                "",
                "Raw artifact evidence:",
                "- `mcp__evidence-tools__evidence_artifact_list(iteration?, kind?, path_contains?, limit?)`",
                "- `mcp__evidence-tools__evidence_artifact_get(artifact_id?, path?, max_chars?)`",
                "- `mcp__evidence-tools__evidence_artifact_search(query, kind?, iteration?, limit?, max_chars_per_match?)`",
                "",
                "Structured fact evidence:",
                "- `mcp__evidence-tools__evidence_fact_state(as_of_iteration?)`",
                "- `mcp__evidence-tools__evidence_fact_candidate_outcome(iteration, candidate_id?, max_examples?, include_retrieval?)`",
                "- `mcp__evidence-tools__evidence_fact_compare_iterations(left, right, left_candidate_id?, right_candidate_id?, max_examples?, include_retrieval?)`",
                "- `mcp__evidence-tools__evidence_fact_task_history(task_id, as_of_iteration?, include_retrieval?)`",
                "- `mcp__evidence-tools__evidence_fact_trace(trace_id, include_spans?, max_spans?)`",
                "- `mcp__evidence-tools__evidence_fact_modification(iteration, include_diff?, max_diff_chars?)`",
                "- `mcp__evidence-tools__evidence_fact_proposer_call(iteration)`",
                "- `mcp__evidence-tools__evidence_fact_file_history(path, limit?)`",
                "",
                "Evidence-link tools:",
                "- `mcp__evidence-tools__evidence_link_for(source_type?, source_id?, target_type?, target_id?, relation?, limit?)`",
                "- `mcp__evidence-tools__evidence_link_explain_iteration(iteration, include_diff?, include_examples?)`",
                "- `mcp__evidence-tools__evidence_link_chain_task(task_id, as_of_iteration?, include_retrieval?)`",
                "",
                "## Rule",
                "",
                "This file is a state snapshot, not evidence and not a plan.",
            ]
        )
        return "\n".join(lines) + "\n"

    def _top_passrate_frontier(
        self,
        *,
        as_of_iteration: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT iteration, candidate_id, passrate, average_score, token_consuming
                FROM candidates
                WHERE iteration <= ?
                ORDER BY passrate DESC, average_score DESC, candidate_id DESC
                LIMIT ?
                """,
                (int(as_of_iteration), max(0, int(limit))),
            ).fetchall()
        return [
            {
                "iteration": int(row["iteration"]),
                "candidate_id": str(row["candidate_id"]),
                "passrate": row["passrate"],
                "average_score": row["average_score"],
                "token_consuming": row["token_consuming"],
            }
            for row in rows
        ]

    def record_state_snapshot(self, iteration: int, state_md: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO state_snapshots(iteration, state_md, created_at)
                VALUES (?, ?, ?)
                """,
                (int(iteration), state_md, _utc_now()),
            )

    def _latest_candidate_iteration(self, *, as_of_iteration: int) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(iteration) AS iteration
                FROM candidates
                WHERE iteration <= ?
                """,
                (int(as_of_iteration),),
            ).fetchone()
        return _int_or_none(row["iteration"]) if row else None

    def _metadata(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = ?",
                (key,),
            ).fetchone()
        return None if row is None else str(row["value"])

    def _record_tool_access(self, iteration: int, tool_access: dict[str, Any]) -> None:
        rows: list[tuple[int, str, str, str, int, int, str]] = []
        files_read = tool_access.get("files_read") or {}
        if isinstance(files_read, dict):
            for path, meta in files_read.items():
                details = meta if isinstance(meta, dict) else {}
                rows.append(
                    (
                        int(iteration),
                        str(path),
                        "read",
                        "agent",
                        _int_metric(details.get("reads") or details.get("read_calls")),
                        _int_metric(details.get("lines") or details.get("read_lines")),
                        _json(details.get("line_ranges") or []),
                    )
                )
        files_written = tool_access.get("files_written") or {}
        if isinstance(files_written, dict):
            for path, meta in files_written.items():
                details = meta if isinstance(meta, dict) else {}
                rows.append(
                    (
                        int(iteration),
                        str(path),
                        "write",
                        "agent",
                        _int_metric(details.get("writes") or details.get("write_calls")),
                        _int_metric(details.get("lines_written") or details.get("written_lines")),
                        _json(details.get("line_ranges") or []),
                    )
                )
        grep_requests = tool_access.get("grep_requests") or []
        if isinstance(grep_requests, list):
            for item in grep_requests:
                if isinstance(item, dict):
                    path = str(item.get("path") or item.get("glob") or item.get("pattern") or "")
                else:
                    path = str(item)
                if path:
                    rows.append((int(iteration), path, "grep", "agent", 1, 0, _json([])))
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO file_accesses (
                    iteration, path, access_type, tool_name, read_calls,
                    read_lines, line_ranges_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def diff_stats(diff_text: str) -> dict[str, Any]:
    files_changed: list[str] = []
    insertions = 0
    deletions = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                files_changed.append(parts[3].removeprefix("b/"))
        elif line.startswith("+") and not line.startswith("+++"):
            insertions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return {
        "files_changed": sorted(set(files_changed)),
        "insertions": insertions,
        "deletions": deletions,
    }


def _candidate_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "iteration": int(row["iteration"]),
        "candidate_id": row["candidate_id"],
        "scaffold_name": row["scaffold_name"],
        "passrate": _float_or_none(row["passrate"]),
        "average_score": _float_or_none(row["average_score"]),
        "token_consuming": _int_or_none(row["token_consuming"]),
        "result_path": row["result_path"],
        "config": _loads(row["config_json"], {}),
        "proposal": _loads(row["proposal_json"], {}),
    }


def _task_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "iteration": int(row["iteration"]),
        "candidate_id": row["candidate_id"],
        "task_id": row["task_id"],
        "score": _float_or_none(row["score"]),
        "passed": bool(row["passed"]),
        "token_consuming": _int_or_none(row["token_consuming"]),
        "trace_path": row["trace_path"],
    }


def _classify_pair(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> tuple[str, float | None]:
    if left is None and right is not None:
        return ("only_in_right", None)
    if left is not None and right is None:
        return ("only_in_left", None)
    assert left is not None and right is not None
    left_score = left.get("score")
    right_score = right.get("score")
    delta = (
        None
        if left_score is None or right_score is None
        else float(right_score) - float(left_score)
    )
    if left["passed"] and not right["passed"]:
        return ("regressed_RvL", delta)
    if not left["passed"] and right["passed"]:
        return ("breakthrough_RvL", delta)
    if left["passed"] and right["passed"]:
        return ("stable_pass", delta)
    return ("both_fail", delta)


def _terms(text: str) -> set[str]:
    return {
        item.lower()
        for item in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", text)
        if len(item) >= 3
    }


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "breakthrough": 0,
        "regressed": 0,
        "stable_pass": 0,
        "persistent_fail": 0,
    }
    for row in rows:
        classification = row.get("classification")
        if classification == "breakthrough_RvL":
            counts["breakthrough"] += 1
        elif classification == "regressed_RvL":
            counts["regressed"] += 1
        elif classification == "stable_pass":
            counts["stable_pass"] += 1
        elif classification == "both_fail":
            counts["persistent_fail"] += 1
    return counts


def _sample_task_ids(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    samples = {"breakthrough": [], "regressed": [], "persistent_fail": []}
    mapping = {
        "breakthrough_RvL": "breakthrough",
        "regressed_RvL": "regressed",
        "both_fail": "persistent_fail",
    }
    for row in rows:
        key = mapping.get(str(row.get("classification")))
        if key and len(samples[key]) < 5:
            samples[key].append(str(row.get("task_id")))
    return samples


def _value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric(outcome: dict[str, Any] | None, key: str) -> str:
    if not outcome:
        return "null"
    return _value(outcome.get(key))


def _candidate_metric(outcome: dict[str, Any] | None, key: str) -> str:
    if not outcome:
        return "null"
    candidate = outcome.get("candidate")
    if not isinstance(candidate, dict):
        return "null"
    return _value(candidate.get(key))


def _bullet_items(items: list[Any]) -> list[str]:
    if not items:
        return ["  - none"]
    return [f"  - {item}" for item in items]


def _indented_items(items: list[str], *, indent: int) -> list[str]:
    prefix = " " * indent
    if not items:
        return [f"{prefix}- none"]
    return [f"{prefix}- {item}" for item in items]


def _candidate_iteration(candidate_id: str) -> int | None:
    match = re.search(r"iter(\d+)", candidate_id)
    if not match:
        return None
    return int(match.group(1))


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _loads(raw: str | None, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _task_tokens(task: dict[str, Any]) -> int | None:
    for key in ("token_consuming", "tokens", "total_tokens"):
        value = _int_or_none(task.get(key))
        if value is not None:
            return value
    prompt = _int_or_none(task.get("prompt_tokens"))
    completion = _int_or_none(task.get("completion_tokens"))
    if prompt is None and completion is None:
        return None
    return int(prompt or 0) + int(completion or 0)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_metric(value: Any) -> int:
    parsed = _int_or_none(value)
    return int(parsed or 0)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
