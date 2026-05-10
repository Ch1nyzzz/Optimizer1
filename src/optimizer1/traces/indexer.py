"""SQLite indexer — materializes jsonl traces into a queryable DB.

Layout: `runs/<run>/traces/index.db`

Three tables:

  - ``traces``: one row per trace. Includes `(jsonl_path, jsonl_lineno)`
    pointers so callers can jump back to the full Trace (with spans)
    on demand.
  - ``diffs``:  one row per trace. Contains `status`, `baseline_trace`,
    `baseline_score`, `delta`. Always populated (status ``baseline`` or
    ``no_baseline`` cover the cases where no diff is computed).
  - ``manifest``: key/value mirror of `traces/manifest.json` so
    consumers can read either source.

The indexer is **idempotent at iteration granularity**: re-running
``materialize_iteration`` for the same iteration replaces only that
iteration's rows. This lets the harness materialize after every
iteration without worrying about cleanup on retries.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .baseline import Baseline
from .diff import (
    STATUS_BASELINE,
    STATUS_NO_BASELINE,
    BaselineEntry,
    classify,
)
from .schema import Trace, read_jsonl


_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id     TEXT PRIMARY KEY,
    iteration    INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    benchmark    TEXT NOT NULL,
    passed       INTEGER NOT NULL,
    score        REAL    NOT NULL,
    jsonl_path   TEXT NOT NULL,
    jsonl_lineno INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_traces_task_iter
    ON traces(task_id, iteration);
CREATE INDEX IF NOT EXISTS ix_traces_iter_cand
    ON traces(iteration, candidate_id);

CREATE TABLE IF NOT EXISTS diffs (
    trace_id        TEXT PRIMARY KEY,
    baseline_trace  TEXT,
    status          TEXT NOT NULL,
    baseline_score  REAL,
    delta           REAL,
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
);

CREATE INDEX IF NOT EXISTS ix_diffs_status ON diffs(status);

CREATE TABLE IF NOT EXISTS file_modifications (
    iteration INTEGER NOT NULL,
    path      TEXT    NOT NULL,
    PRIMARY KEY (iteration, path)
);

CREATE INDEX IF NOT EXISTS ix_file_mods_path ON file_modifications(path);

CREATE TABLE IF NOT EXISTS iteration_diffs (
    iteration  INTEGER PRIMARY KEY,
    diff_text  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS diff_embeddings (
    iteration  INTEGER NOT NULL,
    model      TEXT    NOT NULL,
    dim        INTEGER NOT NULL,
    embedding  BLOB    NOT NULL,
    PRIMARY KEY (iteration, model)
);

CREATE TABLE IF NOT EXISTS iteration_meta (
    iteration          INTEGER PRIMARY KEY,
    patch_base         INTEGER,
    budget             TEXT,
    selection_policy   TEXT,
    advanced_frontier  INTEGER,
    on_pareto_frontier INTEGER,
    passrate           REAL,
    mean_score         REAL,
    proposer_call_dir  TEXT
);

CREATE TABLE IF NOT EXISTS manifest (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class Indexer:
    def __init__(
        self,
        db_path: Path,
        *,
        baseline: Baseline | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        # Mutable so the harness can swap in a self-baseline (current
        # run's iter_0) once it has been recorded.
        self.baseline = baseline or Baseline.empty()

    # ---- connection helpers --------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript(_SCHEMA)
        return conn

    # ---- public API ----------------------------------------------

    def write_manifest(self, manifest: dict[str, object]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO manifest(key, value) VALUES (?, ?)",
                [(str(k), None if v is None else str(v)) for k, v in manifest.items()],
            )

    def materialize_iteration(
        self,
        *,
        iteration: int,
        jsonl_paths: Iterable[Path],
        treat_as_baseline: bool = False,
    ) -> None:
        """Index every trace produced for `iteration`.

        Existing rows for this iteration are deleted first, so retries
        and re-runs of the same iteration cleanly replace data.

        When `treat_as_baseline=True`, every trace gets status
        ``baseline`` regardless of `self.baseline` — the iteration is
        itself the reference. The harness sets this for iter_0 of a
        run that has no external baseline.
        """

        # Read all traces (keep their jsonl_path + line number for
        # later drill-down).
        records: list[tuple[Trace, Path, int]] = []
        for path in jsonl_paths:
            for lineno, trace in enumerate(read_jsonl(path), start=1):
                records.append((trace, path, lineno))

        with self._connect() as conn:
            # Idempotency: drop rows for this iteration before writing.
            # Delete diffs first (uses traces as the source of trace_ids
            # for this iteration), then traces.
            conn.execute(
                "DELETE FROM diffs WHERE trace_id IN "
                "(SELECT trace_id FROM traces WHERE iteration = ?)",
                (iteration,),
            )
            conn.execute("DELETE FROM traces WHERE iteration = ?", (iteration,))

            trace_rows = []
            diff_rows = []
            for trace, path, lineno in records:
                summary = trace.summary or {}
                passed = bool(summary.get("passed", False))
                score = float(summary.get("score") or 0.0)
                trace_rows.append(
                    (
                        trace.trace_id,
                        trace.iteration,
                        trace.candidate_id,
                        trace.task_id,
                        trace.benchmark,
                        1 if passed else 0,
                        score,
                        str(path),
                        lineno,
                    )
                )
                diff_rows.append(
                    self._build_diff_row(
                        trace_id=trace.trace_id,
                        task_id=trace.task_id,
                        passed=passed,
                        score=score,
                        treat_as_baseline=treat_as_baseline,
                    )
                )

            conn.executemany(
                "INSERT INTO traces "
                "(trace_id, iteration, candidate_id, task_id, benchmark, passed, score, jsonl_path, jsonl_lineno) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                trace_rows,
            )
            conn.executemany(
                "INSERT INTO diffs "
                "(trace_id, baseline_trace, status, baseline_score, delta) "
                "VALUES (?, ?, ?, ?, ?)",
                diff_rows,
            )

    def record_file_modifications(
        self,
        *,
        iteration: int,
        paths: Iterable[str],
    ) -> None:
        """Replace the file_modifications rows for one iteration.

        Idempotent: re-recording the same iteration overwrites previous
        rows. Empty `paths` clears the iteration's rows.
        """

        normalized = sorted({p for p in (str(p).strip() for p in paths) if p})
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM file_modifications WHERE iteration = ?",
                (iteration,),
            )
            if normalized:
                conn.executemany(
                    "INSERT INTO file_modifications(iteration, path) VALUES (?, ?)",
                    [(iteration, p) for p in normalized],
                )

    def record_diff_text(self, *, iteration: int, diff_text: str) -> None:
        """Insert or replace the raw diff text for one iteration.

        Called by the harness after each evaluation. Embedding is a
        separate concern handled lazily by the MCP layer; storing the
        text unconditionally lets ``trace_similar`` re-embed any
        history at call time without optimizer-level setup.
        """

        if not diff_text or not diff_text.strip():
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO iteration_diffs(iteration, diff_text) "
                "VALUES (?, ?)",
                (int(iteration), diff_text),
            )

    def record_diff_embedding(
        self,
        *,
        iteration: int,
        model: str,
        dim: int,
        embedding: bytes,
    ) -> None:
        """Insert or replace one (iteration, model) embedding row.

        Called by the MCP ``trace_similar`` tool when it lazily embeds
        a diff that has no cached vector for the active model.
        """

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO diff_embeddings("
                "iteration, model, dim, embedding) "
                "VALUES (?, ?, ?, ?)",
                (int(iteration), model, int(dim), embedding),
            )

    def upsert_iteration_meta(
        self,
        *,
        iteration: int,
        patch_base: int | None = None,
        budget: str | None = None,
        selection_policy: str | None = None,
        advanced_frontier: bool | None = None,
        on_pareto_frontier: bool | None = None,
        passrate: float | None = None,
        mean_score: float | None = None,
        proposer_call_dir: str | None = None,
    ) -> None:
        """Insert or update the iteration_meta row for one iteration.

        Fields passed as ``None`` preserve the existing value (via
        ``COALESCE`` on conflict). Use a non-``None`` sentinel to set a
        field; passing ``None`` does not clear it.
        """

        adv = None if advanced_frontier is None else (1 if advanced_frontier else 0)
        on_fr = None if on_pareto_frontier is None else (1 if on_pareto_frontier else 0)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO iteration_meta (
                    iteration, patch_base, budget, selection_policy,
                    advanced_frontier, on_pareto_frontier,
                    passrate, mean_score, proposer_call_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(iteration) DO UPDATE SET
                    patch_base         = COALESCE(excluded.patch_base,         iteration_meta.patch_base),
                    budget             = COALESCE(excluded.budget,             iteration_meta.budget),
                    selection_policy   = COALESCE(excluded.selection_policy,   iteration_meta.selection_policy),
                    advanced_frontier  = COALESCE(excluded.advanced_frontier,  iteration_meta.advanced_frontier),
                    on_pareto_frontier = COALESCE(excluded.on_pareto_frontier, iteration_meta.on_pareto_frontier),
                    passrate           = COALESCE(excluded.passrate,           iteration_meta.passrate),
                    mean_score         = COALESCE(excluded.mean_score,         iteration_meta.mean_score),
                    proposer_call_dir  = COALESCE(excluded.proposer_call_dir,  iteration_meta.proposer_call_dir)
                """,
                (
                    int(iteration),
                    None if patch_base is None else int(patch_base),
                    budget,
                    selection_policy,
                    adv,
                    on_fr,
                    None if passrate is None else float(passrate),
                    None if mean_score is None else float(mean_score),
                    proposer_call_dir,
                ),
            )

    def refresh_pareto_frontier(self, iter_to_on_frontier: dict[int, bool]) -> None:
        """Bulk-update ``on_pareto_frontier`` for every known iteration.

        ``iter_to_on_frontier`` maps iteration → bool. Iterations not
        present in the map are explicitly set to ``0`` (off-frontier),
        so callers should pass the full picture each time the frontier
        is recomputed. Iterations missing from ``iteration_meta`` get
        a row inserted with the flag set; other fields stay ``NULL``.
        """

        with self._connect() as conn:
            conn.execute("UPDATE iteration_meta SET on_pareto_frontier = 0")
            for iteration, on_frontier in iter_to_on_frontier.items():
                flag = 1 if on_frontier else 0
                conn.execute(
                    """
                    INSERT INTO iteration_meta (iteration, on_pareto_frontier)
                    VALUES (?, ?)
                    ON CONFLICT(iteration) DO UPDATE SET
                        on_pareto_frontier = excluded.on_pareto_frontier
                    """,
                    (int(iteration), flag),
                )

    # ---- internals -----------------------------------------------

    def _build_diff_row(
        self,
        *,
        trace_id: str,
        task_id: str,
        passed: bool,
        score: float,
        treat_as_baseline: bool,
    ) -> tuple[str, str | None, str, float | None, float | None]:
        if treat_as_baseline:
            return (trace_id, None, STATUS_BASELINE, None, None)

        baseline_entry: BaselineEntry | None = self.baseline.lookup(task_id)
        if baseline_entry is None:
            return (trace_id, None, STATUS_NO_BASELINE, None, None)

        status = classify(curr_passed=passed, baseline=baseline_entry)
        delta = score - baseline_entry.score
        return (
            trace_id,
            baseline_entry.trace_id,
            status,
            baseline_entry.score,
            delta,
        )
