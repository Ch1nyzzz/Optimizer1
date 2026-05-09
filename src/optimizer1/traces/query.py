"""Structured queries over the trace harness index.

Read-only, deterministic SQL over `traces/index.db`. The proposer is the
primary consumer: it asks "what tasks have been failing?", "what files
changed in iter N?", "what was the outcome of candidate X?" — and gets
JSON back instead of having to grep markdown or write SQL.

All "diff-mode" predicates (persistent_failures / regressions /
breakthroughs) are defined over `diffs.status`, i.e. relative to the
trace harness baseline. This keeps semantics consistent with the
diagnostic markdown the renderer already produces.

Statuses (see `optimizer1.traces.diff`):
    baseline / regressed / breakthrough / stable_pass / persistent_fail / no_baseline

The class is intentionally non-mutating — construction only opens the
DB; no writes happen here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .diff import (
    STATUS_BREAKTHROUGH,
    STATUS_PERSISTENT_FAIL,
    STATUS_REGRESSED,
)


class TraceQuery:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"trace index not found: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- queries -------------------------------------------------

    def task_history(self, task_id: str) -> list[dict[str, Any]]:
        """One row per (iter, candidate) trace recorded for `task_id`,
        sorted ascending by iteration. Joins traces × diffs so each row
        carries the harness status and (where applicable) score delta.
        """

        sql = """
            SELECT
                t.iteration    AS iteration,
                t.candidate_id AS candidate_id,
                t.passed       AS passed,
                t.score        AS score,
                d.status       AS status,
                d.baseline_score AS baseline_score,
                d.delta        AS delta
            FROM traces t
            LEFT JOIN diffs d USING (trace_id)
            WHERE t.task_id = ?
            ORDER BY t.iteration ASC, t.candidate_id ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (task_id,)).fetchall()
        return [
            {
                "iteration": int(r["iteration"]),
                "candidate_id": r["candidate_id"],
                "passed": bool(r["passed"]),
                "score": float(r["score"]),
                "status": r["status"],
                "baseline_score": (
                    float(r["baseline_score"])
                    if r["baseline_score"] is not None
                    else None
                ),
                "delta": float(r["delta"]) if r["delta"] is not None else None,
            }
            for r in rows
        ]

    def persistent_failures(self, *, min_streak: int = 3) -> list[dict[str, Any]]:
        """Tasks whose most-recent contiguous run of `persistent_fail`
        statuses is at least `min_streak` long.

        Streak boundary is per-(task, candidate): we look at the trailing
        suffix of (task, candidate) iters ordered by iteration descending
        and count how many are persistent_fail before hitting any other
        status. The longest streak across candidates wins.
        """

        sql = """
            SELECT
                t.task_id      AS task_id,
                t.candidate_id AS candidate_id,
                t.iteration    AS iteration,
                d.status       AS status
            FROM traces t
            LEFT JOIN diffs d USING (trace_id)
            ORDER BY t.task_id ASC, t.candidate_id ASC, t.iteration DESC
        """
        out: dict[str, dict[str, Any]] = {}
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()

        # Group by (task_id, candidate_id), walk suffix until status differs.
        prev_key: tuple[str, str] | None = None
        streak_iters: list[int] = []
        for row in rows:
            key = (row["task_id"], row["candidate_id"])
            if key != prev_key:
                if prev_key is not None:
                    self._maybe_update_failure(out, prev_key[0], streak_iters, min_streak)
                prev_key = key
                streak_iters = []
            if row["status"] == STATUS_PERSISTENT_FAIL:
                streak_iters.append(int(row["iteration"]))
            elif streak_iters:
                # Suffix broken: lock in what we have and stop counting
                # within this (task, candidate) group.
                self._maybe_update_failure(out, key[0], streak_iters, min_streak)
                streak_iters = []
                prev_key = ("__sealed__", row["candidate_id"])
        if prev_key is not None and prev_key[0] != "__sealed__":
            self._maybe_update_failure(out, prev_key[0], streak_iters, min_streak)

        result = list(out.values())
        result.sort(key=lambda x: (-x["current_streak"], x["task_id"]))
        return result

    @staticmethod
    def _maybe_update_failure(
        out: dict[str, dict[str, Any]],
        task_id: str,
        streak_iters: list[int],
        min_streak: int,
    ) -> None:
        if len(streak_iters) < min_streak:
            return
        existing = out.get(task_id)
        candidate = {
            "task_id": task_id,
            "current_streak": len(streak_iters),
            "first_iter": min(streak_iters),
            "last_iter": max(streak_iters),
        }
        if existing is None or candidate["current_streak"] > existing["current_streak"]:
            out[task_id] = candidate

    def breakthroughs(self, *, since_iter: int = 0) -> list[dict[str, Any]]:
        """Traces with status='breakthrough' at iter >= since_iter."""

        sql = """
            SELECT
                t.iteration    AS iteration,
                t.candidate_id AS candidate_id,
                t.task_id      AS task_id,
                t.score        AS score,
                d.delta        AS delta
            FROM traces t
            JOIN diffs d USING (trace_id)
            WHERE d.status = ? AND t.iteration >= ?
            ORDER BY t.iteration ASC, t.task_id ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (STATUS_BREAKTHROUGH, since_iter)).fetchall()
        return [
            {
                "iteration": int(r["iteration"]),
                "candidate_id": r["candidate_id"],
                "task_id": r["task_id"],
                "score": float(r["score"]),
                "delta": float(r["delta"]) if r["delta"] is not None else None,
            }
            for r in rows
        ]

    def regressions(self, *, window: int = 3) -> list[dict[str, Any]]:
        """Traces with status='regressed' within the last `window` iters.

        `window` is counted from the max iteration present in the
        traces table. Useful for "what regressed recently?" queries.
        """

        with self._connect() as conn:
            max_iter_row = conn.execute(
                "SELECT MAX(iteration) AS mi FROM traces"
            ).fetchone()
            max_iter = (max_iter_row["mi"] if max_iter_row else None) or 0
            cutoff = max(0, int(max_iter) - int(window) + 1)
            rows = conn.execute(
                """
                SELECT
                    t.iteration    AS iteration,
                    t.candidate_id AS candidate_id,
                    t.task_id      AS task_id,
                    t.score        AS score,
                    d.baseline_score AS baseline_score,
                    d.delta        AS delta
                FROM traces t
                JOIN diffs d USING (trace_id)
                WHERE d.status = ? AND t.iteration >= ?
                ORDER BY t.iteration DESC, t.task_id ASC
                """,
                (STATUS_REGRESSED, cutoff),
            ).fetchall()
        return [
            {
                "iteration": int(r["iteration"]),
                "candidate_id": r["candidate_id"],
                "task_id": r["task_id"],
                "score": float(r["score"]),
                "baseline_score": (
                    float(r["baseline_score"])
                    if r["baseline_score"] is not None
                    else None
                ),
                "delta": float(r["delta"]) if r["delta"] is not None else None,
            }
            for r in rows
        ]

    def file_history(self, path: str) -> list[dict[str, Any]]:
        """Iters in which `path` appears in diff.patch, with that iter's
        aggregated outcome (passrate + status counts across all traces).
        """

        sql = """
            SELECT
                fm.iteration   AS iteration,
                COUNT(DISTINCT t.candidate_id) AS candidate_count,
                AVG(t.passed)  AS passrate,
                SUM(CASE WHEN d.status = 'regressed'        THEN 1 ELSE 0 END) AS regressed,
                SUM(CASE WHEN d.status = 'breakthrough'     THEN 1 ELSE 0 END) AS breakthrough,
                SUM(CASE WHEN d.status = 'stable_pass'      THEN 1 ELSE 0 END) AS stable_pass,
                SUM(CASE WHEN d.status = 'persistent_fail'  THEN 1 ELSE 0 END) AS persistent_fail,
                SUM(CASE WHEN d.status = 'baseline'         THEN 1 ELSE 0 END) AS baseline,
                SUM(CASE WHEN d.status = 'no_baseline'      THEN 1 ELSE 0 END) AS no_baseline
            FROM file_modifications fm
            LEFT JOIN traces t USING (iteration)
            LEFT JOIN diffs d USING (trace_id)
            WHERE fm.path = ?
            GROUP BY fm.iteration
            ORDER BY fm.iteration ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (path,)).fetchall()
        return [
            {
                "iteration": int(r["iteration"]),
                "candidate_count": int(r["candidate_count"] or 0),
                "passrate": (
                    float(r["passrate"]) if r["passrate"] is not None else None
                ),
                "status_counts": {
                    "regressed": int(r["regressed"] or 0),
                    "breakthrough": int(r["breakthrough"] or 0),
                    "stable_pass": int(r["stable_pass"] or 0),
                    "persistent_fail": int(r["persistent_fail"] or 0),
                    "baseline": int(r["baseline"] or 0),
                    "no_baseline": int(r["no_baseline"] or 0),
                },
            }
            for r in rows
        ]

    def candidate_outcome(
        self,
        iteration: int,
        candidate_id: str,
    ) -> dict[str, Any]:
        """Full per-(iter, candidate) summary: counts, passrate, mean
        score, jsonl pointer (for drill-in), and modified file list.
        """

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS n_traces,
                    AVG(t.passed)  AS passrate,
                    AVG(t.score)   AS mean_score,
                    MIN(t.jsonl_path) AS jsonl_path,
                    SUM(CASE WHEN d.status = 'regressed'        THEN 1 ELSE 0 END) AS regressed,
                    SUM(CASE WHEN d.status = 'breakthrough'     THEN 1 ELSE 0 END) AS breakthrough,
                    SUM(CASE WHEN d.status = 'stable_pass'      THEN 1 ELSE 0 END) AS stable_pass,
                    SUM(CASE WHEN d.status = 'persistent_fail'  THEN 1 ELSE 0 END) AS persistent_fail,
                    SUM(CASE WHEN d.status = 'baseline'         THEN 1 ELSE 0 END) AS baseline,
                    SUM(CASE WHEN d.status = 'no_baseline'      THEN 1 ELSE 0 END) AS no_baseline
                FROM traces t
                LEFT JOIN diffs d USING (trace_id)
                WHERE t.iteration = ? AND t.candidate_id = ?
                """,
                (iteration, candidate_id),
            ).fetchone()
            modified_paths = [
                r["path"]
                for r in conn.execute(
                    "SELECT path FROM file_modifications WHERE iteration = ? "
                    "ORDER BY path ASC",
                    (iteration,),
                ).fetchall()
            ]

        if row is None or (row["n_traces"] or 0) == 0:
            return {
                "iteration": iteration,
                "candidate_id": candidate_id,
                "n_traces": 0,
                "passrate": None,
                "mean_score": None,
                "jsonl_path": None,
                "modified_paths": modified_paths,
                "status_counts": {
                    "regressed": 0,
                    "breakthrough": 0,
                    "stable_pass": 0,
                    "persistent_fail": 0,
                    "baseline": 0,
                    "no_baseline": 0,
                },
            }
        return {
            "iteration": iteration,
            "candidate_id": candidate_id,
            "n_traces": int(row["n_traces"] or 0),
            "passrate": float(row["passrate"]) if row["passrate"] is not None else None,
            "mean_score": (
                float(row["mean_score"]) if row["mean_score"] is not None else None
            ),
            "jsonl_path": row["jsonl_path"],
            "modified_paths": modified_paths,
            "status_counts": {
                "regressed": int(row["regressed"] or 0),
                "breakthrough": int(row["breakthrough"] or 0),
                "stable_pass": int(row["stable_pass"] or 0),
                "persistent_fail": int(row["persistent_fail"] or 0),
                "baseline": int(row["baseline"] or 0),
                "no_baseline": int(row["no_baseline"] or 0),
            },
        }
