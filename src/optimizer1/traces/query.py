"""Structured queries over the trace harness index.

Read-only, deterministic SQL over `traces/index.db`. The proposer is the
primary consumer: it asks "what tasks have been failing?", "what files
changed in iter N?", "what was the outcome of candidate X?" — and gets
JSON back instead of having to grep markdown or write SQL.

The query surface is intentionally minimal and orthogonal — primitives
the calling agent composes itself rather than convenience wrappers
that hard-code policy. The cross-iter ``compare_iterations`` primitive
covers any "vs baseline / vs frontier-best / vs patch base" question;
``iteration_metadata`` exposes the full ``iteration_meta`` row so the
agent can decide which iters are interesting.

Statuses on the legacy ``diffs`` table (see ``optimizer1.traces.diff``)
are still available through ``task_history`` / ``candidate_outcome`` /
``file_history`` for "vs harness baseline" analyses:
    baseline / regressed / breakthrough / stable_pass / persistent_fail / no_baseline

The class is intentionally non-mutating — construction only opens the
DB; no writes happen here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


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
        *,
        max_examples: int = 8,
    ) -> dict[str, Any]:
        """Full per-(iter, candidate) summary: counts, passrate, mean
        score, artifact pointers, modified files, and representative
        task examples.
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
            diff_path = self._diff_path_for_iteration(conn, iteration)
            example_limit = max(0, int(max_examples))
            regressed_tasks = self._candidate_task_examples(
                conn,
                iteration=iteration,
                candidate_id=candidate_id,
                statuses=("regressed",),
                limit=example_limit,
            )
            breakthrough_tasks = self._candidate_task_examples(
                conn,
                iteration=iteration,
                candidate_id=candidate_id,
                statuses=("breakthrough",),
                limit=example_limit,
            )
            failed_tasks = self._candidate_task_examples(
                conn,
                iteration=iteration,
                candidate_id=candidate_id,
                passed=False,
                limit=example_limit,
            )

        if row is None or (row["n_traces"] or 0) == 0:
            return {
                "iteration": iteration,
                "candidate_id": candidate_id,
                "n_traces": 0,
                "passrate": None,
                "mean_score": None,
                "jsonl_path": None,
                "diff_path": diff_path,
                "modified_paths": modified_paths,
                "regressed_tasks": [],
                "breakthrough_tasks": [],
                "failed_tasks": [],
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
            "diff_path": diff_path,
            "modified_paths": modified_paths,
            "regressed_tasks": regressed_tasks,
            "breakthrough_tasks": breakthrough_tasks,
            "failed_tasks": failed_tasks,
            "status_counts": {
                "regressed": int(row["regressed"] or 0),
                "breakthrough": int(row["breakthrough"] or 0),
                "stable_pass": int(row["stable_pass"] or 0),
                "persistent_fail": int(row["persistent_fail"] or 0),
                "baseline": int(row["baseline"] or 0),
                "no_baseline": int(row["no_baseline"] or 0),
            },
        }

    def _diff_path_for_iteration(
        self,
        conn: sqlite3.Connection,
        iteration: int,
    ) -> str | None:
        """Best-effort path to the iteration's diff.patch.

        In a live proposer workspace the run-level trace DB is copied to
        ``workspace/traces`` while selected reference bundles live under
        ``workspace/reference_iterations``. In retained run artifacts,
        the DB lives under ``runs/<run>/traces`` and diffs live under
        ``runs/<run>/proposer_calls``. Prefer whichever path exists in
        the current filesystem, then fall back to recorded metadata.
        """

        root = self.db_path.parent.parent
        candidates = [
            root / "reference_iterations" / f"iter_{iteration:03d}" / "diff.patch",
            root / "proposer_calls" / f"iter_{iteration:03d}" / "diff.patch",
        ]
        meta_row = conn.execute(
            "SELECT proposer_call_dir FROM iteration_meta WHERE iteration = ?",
            (int(iteration),),
        ).fetchone()
        proposer_call_dir = (
            meta_row["proposer_call_dir"]
            if meta_row is not None and meta_row["proposer_call_dir"]
            else None
        )
        if proposer_call_dir:
            candidates.append(Path(str(proposer_call_dir)) / "diff.patch")

        for path in candidates:
            if path.exists():
                return str(path)
        if proposer_call_dir:
            return str(Path(str(proposer_call_dir)) / "diff.patch")
        return None

    def _candidate_task_examples(
        self,
        conn: sqlite3.Connection,
        *,
        iteration: int,
        candidate_id: str,
        statuses: tuple[str, ...] | None = None,
        passed: bool | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        where = ["t.iteration = ?", "t.candidate_id = ?"]
        params: list[object] = [int(iteration), candidate_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            where.append(f"d.status IN ({placeholders})")
            params.extend(statuses)
        if passed is not None:
            where.append("t.passed = ?")
            params.append(1 if passed else 0)

        sql = f"""
            SELECT
                t.task_id      AS task_id,
                t.passed       AS passed,
                t.score        AS score,
                t.jsonl_path   AS jsonl_path,
                t.jsonl_lineno AS jsonl_lineno,
                d.status       AS status,
                d.baseline_score AS baseline_score,
                d.delta        AS delta
            FROM traces t
            LEFT JOIN diffs d USING (trace_id)
            WHERE {' AND '.join(where)}
            ORDER BY
                CASE d.status
                    WHEN 'regressed' THEN 0
                    WHEN 'persistent_fail' THEN 1
                    WHEN 'no_baseline' THEN 2
                    WHEN 'breakthrough' THEN 3
                    WHEN 'stable_pass' THEN 4
                    WHEN 'baseline' THEN 5
                    ELSE 6
                END,
                t.score ASC,
                t.task_id ASC
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            {
                "task_id": str(row["task_id"]),
                "passed": bool(row["passed"]),
                "score": float(row["score"]),
                "status": row["status"],
                "baseline_score": (
                    float(row["baseline_score"])
                    if row["baseline_score"] is not None
                    else None
                ),
                "delta": float(row["delta"]) if row["delta"] is not None else None,
                "jsonl_path": row["jsonl_path"],
                "jsonl_lineno": int(row["jsonl_lineno"]),
            }
            for row in rows
        ]

    # ---- new orthogonal primitives -----------------------------------

    def list_tasks(self, *, iteration: int | None = None) -> list[str]:
        """Distinct task_ids, optionally scoped to one iteration.

        Returned ascending. Empty list when no traces match.
        """

        if iteration is None:
            sql = "SELECT DISTINCT task_id FROM traces ORDER BY task_id ASC"
            params: tuple[object, ...] = ()
        else:
            sql = (
                "SELECT DISTINCT task_id FROM traces WHERE iteration = ? "
                "ORDER BY task_id ASC"
            )
            params = (int(iteration),)
        with self._connect() as conn:
            return [str(row["task_id"]) for row in conn.execute(sql, params)]

    def iteration_metadata(
        self,
        *,
        iters: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Return iteration_meta rows (full set or restricted to ``iters``).

        Each row is the canonical ``iteration_meta`` shape: iteration,
        patch_base, budget, selection_policy, advanced_frontier (bool),
        on_pareto_frontier (bool), passrate, mean_score,
        proposer_call_dir. Iters with no row return nothing.
        """

        if iters is None:
            sql = "SELECT * FROM iteration_meta ORDER BY iteration ASC"
            params: tuple[object, ...] = ()
        else:
            normalized = [int(item) for item in iters]
            if not normalized:
                return []
            placeholders = ",".join("?" for _ in normalized)
            sql = (
                f"SELECT * FROM iteration_meta WHERE iteration IN ({placeholders}) "
                "ORDER BY iteration ASC"
            )
            params = tuple(normalized)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "iteration": int(row["iteration"]),
                "patch_base": (
                    int(row["patch_base"]) if row["patch_base"] is not None else None
                ),
                "budget": row["budget"],
                "selection_policy": row["selection_policy"],
                "advanced_frontier": (
                    bool(row["advanced_frontier"])
                    if row["advanced_frontier"] is not None
                    else None
                ),
                "on_pareto_frontier": (
                    bool(row["on_pareto_frontier"])
                    if row["on_pareto_frontier"] is not None
                    else None
                ),
                "passrate": (
                    float(row["passrate"]) if row["passrate"] is not None else None
                ),
                "mean_score": (
                    float(row["mean_score"]) if row["mean_score"] is not None else None
                ),
                "proposer_call_dir": row["proposer_call_dir"],
            }
            for row in rows
        ]

    def compare_iterations(
        self,
        left: int,
        right: int,
        *,
        left_candidate_id: str | None = None,
        right_candidate_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Per-task pass/fail comparison between two iterations.

        For each task that appears in either iter, the row carries the
        per-side (passed, score, candidate_id) plus a ``classification``:

          - ``regressed_RvL``    — passed in left, fails in right
          - ``breakthrough_RvL`` — failed in left, passes in right
          - ``stable_pass``      — both passed
          - ``both_fail``        — both failed
          - ``only_in_left``     — task absent in right
          - ``only_in_right``    — task absent in left

        ``delta`` = ``right.score - left.score`` (None when one side is missing).

        Candidate selection per side: when the corresponding
        ``*_candidate_id`` is None, the iter's highest-passrate
        candidate is used (tie-break by candidate_id desc — matches the
        frontier rule). Pass an explicit id to compare specific
        candidates.

        Output order: regressed_RvL → only_in_left → both_fail →
        only_in_right → stable_pass → breakthrough_RvL, then by
        task_id asc within each bucket. The ordering surfaces the
        most diagnostically interesting cases first.
        """

        left_cand = left_candidate_id or self._headline_candidate(left)
        right_cand = right_candidate_id or self._headline_candidate(right)

        rows: dict[str, dict[str, Any]] = {}
        with self._connect() as conn:
            if left_cand is not None:
                left_rows = conn.execute(
                    "SELECT task_id, passed, score, candidate_id FROM traces "
                    "WHERE iteration = ? AND candidate_id = ?",
                    (int(left), left_cand),
                ).fetchall()
                for row in left_rows:
                    rows[str(row["task_id"])] = {
                        "task_id": str(row["task_id"]),
                        "left": {
                            "passed": bool(row["passed"]),
                            "score": float(row["score"]),
                            "candidate_id": str(row["candidate_id"]),
                        },
                        "right": None,
                    }
            if right_cand is not None:
                right_rows = conn.execute(
                    "SELECT task_id, passed, score, candidate_id FROM traces "
                    "WHERE iteration = ? AND candidate_id = ?",
                    (int(right), right_cand),
                ).fetchall()
                for row in right_rows:
                    entry = rows.setdefault(
                        str(row["task_id"]),
                        {
                            "task_id": str(row["task_id"]),
                            "left": None,
                            "right": None,
                        },
                    )
                    entry["right"] = {
                        "passed": bool(row["passed"]),
                        "score": float(row["score"]),
                        "candidate_id": str(row["candidate_id"]),
                    }

        out: list[dict[str, Any]] = []
        for entry in rows.values():
            left_side = entry["left"]
            right_side = entry["right"]
            classification, delta = _classify_pair(left_side, right_side)
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

    def _headline_candidate(self, iteration: int) -> str | None:
        """Pick the highest-passrate candidate for one iteration.

        Tie-break by candidate_id desc (matches the frontier rule).
        Returns None when the iteration has no traces.
        """

        sql = """
            SELECT candidate_id,
                   AVG(passed) AS passrate
            FROM traces
            WHERE iteration = ?
            GROUP BY candidate_id
            ORDER BY passrate DESC, candidate_id DESC
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (int(iteration),)).fetchone()
        if row is None:
            return None
        return str(row["candidate_id"])


def _classify_pair(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> tuple[str, float | None]:
    if left is None and right is not None:
        return ("only_in_right", None)
    if left is not None and right is None:
        return ("only_in_left", None)
    assert left is not None and right is not None
    delta = right["score"] - left["score"]
    if left["passed"] and not right["passed"]:
        return ("regressed_RvL", delta)
    if not left["passed"] and right["passed"]:
        return ("breakthrough_RvL", delta)
    if left["passed"] and right["passed"]:
        return ("stable_pass", delta)
    return ("both_fail", delta)
