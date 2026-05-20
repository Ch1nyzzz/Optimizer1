"""Unified evidence database for optimizer runs.

``EvidenceStore`` is an additive store: existing ``runstore.db`` and
``traces/index.db`` continue to be written, while this module imports their
structured facts into ``runs/<run>/evidence_store.db``. Raw artifacts are kept as
lightweight indexes/previews only; the run directory remains the source of truth
for full files. The schema keeps trace/evaluation/modification facts and
explicit links between modifications and the evidence the proposer touched.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from optimizer1.traces.schema import read_jsonl, trace_to_dict


SCHEMA_VERSION = "optimizer1.evidence_store.v1"

INLINE_ARTIFACT_MAX_BYTES = 1_000_000
PREVIEW_ARTIFACT_MAX_BYTES = 64_000

_RUN_LEVEL_ARTIFACT_NAMES = {
    "best_candidates.json",
    "candidate_score_table.json",
    "diff_summary.jsonl",
    "evolution_summary.jsonl",
    "iteration_index.json",
    "optimizer_summary.json",
    "pareto_frontier.json",
    "pending_eval.json",
    "retrieval_diagnostics_summary.json",
    "run_summary.json",
    "state.md",
    "test_frontier_summary.json",
}

_CALL_LEVEL_ARTIFACT_NAMES = {
    "access_policy.json",
    "assignment.json",
    "diff.patch",
    "diff_digest.md",
    "pending_eval.json",
    "pending_eval.raw.json",
    "source_snapshot_manifest.json",
    "workspace_manifest.json",
}

_EVAL_ARTIFACT_NAMES = {
    "candidate_result.compact.json",
    "eval_summary.json",
}

_AGENT_ARTIFACT_NAMES = {
    "metrics.json",
    "meta.json",
    "prompt.md",
    "stderr.txt",
    "stdout.md",
    "tool_access.json",
}

_SKIP_ARTIFACT_PARTS = {
    ".git",
    ".mypy_cache",
    ".optimizer1_mcp_src",
    ".pytest_cache",
    "__pycache__",
    "generated",
    "reference_iterations",
    "source_snapshot",
    "workspace",
}

_SKIP_ARTIFACT_SUFFIXES = {
    ".db",
    ".db-journal",
    ".db-shm",
    ".db-wal",
    ".journal",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite3",
    ".wal",
    ".shm",
}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    iteration INTEGER,
    candidate_id TEXT,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    content_encoding TEXT NOT NULL,
    content_blob BLOB NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_artifacts_kind_iter
    ON artifacts(kind, iteration);
CREATE INDEX IF NOT EXISTS ix_artifacts_path
    ON artifacts(path);

CREATE TABLE IF NOT EXISTS iterations (
    iteration INTEGER PRIMARY KEY,
    status TEXT,
    as_of_iteration INTEGER,
    base_iteration INTEGER,
    base_candidate_id TEXT,
    patch_base INTEGER,
    budget TEXT,
    selection_policy TEXT,
    advanced_frontier INTEGER,
    on_pareto_frontier INTEGER,
    passrate REAL,
    mean_score REAL,
    proposer_call_dir TEXT,
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

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    iteration INTEGER NOT NULL,
    candidate_id TEXT,
    proposal_name TEXT,
    scaffold_name TEXT,
    hypothesis TEXT,
    changes TEXT,
    expected_effect_json TEXT,
    risk_json TEXT,
    evidence_refs_json TEXT,
    proposal_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_proposals_iter_candidate
    ON proposals(iteration, candidate_id);

CREATE TABLE IF NOT EXISTS proposal_outcomes (
    proposal_id TEXT PRIMARY KEY,
    iteration INTEGER NOT NULL,
    candidate_id TEXT,
    base_iteration INTEGER,
    base_candidate_id TEXT,
    passrate_delta REAL,
    average_score_delta REAL,
    token_delta INTEGER,
    breakthrough_count INTEGER NOT NULL,
    regression_count INTEGER NOT NULL,
    stable_pass_count INTEGER NOT NULL,
    persistent_fail_count INTEGER NOT NULL,
    task_count INTEGER NOT NULL,
    outcome_summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    iteration INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    score REAL,
    passed INTEGER,
    token_consuming INTEGER,
    result_path TEXT,
    trace_id TEXT,
    PRIMARY KEY (iteration, candidate_id, task_id)
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    iteration INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    benchmark TEXT,
    passed INTEGER,
    score REAL,
    summary_json TEXT,
    diff_json TEXT,
    jsonl_artifact_id TEXT,
    jsonl_lineno INTEGER,
    jsonl_path TEXT
);

CREATE INDEX IF NOT EXISTS ix_traces_iter_candidate
    ON traces(iteration, candidate_id);
CREATE INDEX IF NOT EXISTS ix_traces_task
    ON traces(task_id);

CREATE TABLE IF NOT EXISTS trace_spans (
    trace_id TEXT NOT NULL,
    span_ord INTEGER NOT NULL,
    span_id TEXT,
    parent_span_id TEXT,
    kind TEXT,
    input_json TEXT,
    output_json TEXT,
    metadata_json TEXT,
    PRIMARY KEY (trace_id, span_ord)
);

CREATE TABLE IF NOT EXISTS trace_diffs (
    trace_id TEXT PRIMARY KEY,
    baseline_trace TEXT,
    status TEXT,
    baseline_score REAL,
    delta REAL
);

CREATE TABLE IF NOT EXISTS modifications (
    iteration INTEGER PRIMARY KEY,
    diff_text TEXT,
    files_changed_json TEXT,
    insertions INTEGER,
    deletions INTEGER,
    diff_artifact_id TEXT,
    pending_eval_artifact_id TEXT,
    hypothesis TEXT,
    proposal_json TEXT
);

CREATE TABLE IF NOT EXISTS modified_files (
    iteration INTEGER NOT NULL,
    path TEXT NOT NULL,
    PRIMARY KEY (iteration, path)
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

CREATE TABLE IF NOT EXISTS tool_accesses (
    access_id TEXT PRIMARY KEY,
    iteration INTEGER NOT NULL,
    path TEXT NOT NULL,
    access_type TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    read_calls INTEGER,
    read_lines INTEGER,
    line_ranges_json TEXT
);

CREATE INDEX IF NOT EXISTS ix_tool_accesses_iter_type
    ON tool_accesses(iteration, access_type);

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
    artifact_id TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS evidence_links (
    link_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    provenance TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_evidence_links_source
    ON evidence_links(source_type, source_id);
CREATE INDEX IF NOT EXISTS ix_evidence_links_target
    ON evidence_links(target_type, target_id);
"""


class EvidenceStore:
    """Unified, idempotent evidence store for one optimizer run."""

    def __init__(self, run_dir: Path, *, initialize: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.db_path = self.run_dir / "evidence_store.db"
        if initialize:
            self._ensure_schema()
            self.set_metadata("schema", SCHEMA_VERSION)
            self.set_metadata("run_id", self.run_id)

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

    def refresh(self, *, iteration: int | None = None) -> None:
        """Import all currently available evidence for this run.

        ``iteration`` limits raw artifact scanning to the current call
        directory plus run-level files. Structured source DBs are imported
        wholesale because they are already compact and idempotent.
        """

        self._import_runstore()
        self._import_trace_index()
        self._import_trace_jsonl(iteration=iteration)
        self._import_run_artifacts(iteration=iteration)
        self._rebuild_proposal_outcomes()
        self._rebuild_links()
        self.set_metadata("last_refresh_at", _utc_now())

    # ---- source database import ---------------------------------

    def _import_runstore(self) -> None:
        path = self.run_dir / "runstore.db"
        if not path.exists():
            return
        with sqlite3.connect(path) as src, self._connect() as dest:
            src.row_factory = sqlite3.Row
            for row in _select(src, "SELECT * FROM iterations"):
                dest.execute(
                    """
                    INSERT INTO iterations (
                        iteration, status, as_of_iteration, base_iteration,
                        base_candidate_id, created_at, committed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(iteration) DO UPDATE SET
                        status = excluded.status,
                        as_of_iteration = excluded.as_of_iteration,
                        base_iteration = excluded.base_iteration,
                        base_candidate_id = excluded.base_candidate_id,
                        created_at = COALESCE(excluded.created_at, iterations.created_at),
                        committed_at = COALESCE(excluded.committed_at, iterations.committed_at)
                    """,
                    (
                        row["iteration"],
                        row["status"],
                        row["as_of_iteration"],
                        row["base_iteration"],
                        row["base_candidate_id"],
                        row["created_at"],
                        row["committed_at"],
                    ),
                )
            for row in _select(src, "SELECT * FROM candidates"):
                dest.execute(
                    """
                    INSERT OR REPLACE INTO candidates (
                        iteration, candidate_id, scaffold_name, passrate,
                        average_score, token_consuming, result_path,
                        config_json, proposal_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(row),
                )
                proposal = _loads(row["proposal_json"], {})
                if proposal:
                    self._upsert_proposal_payload(
                        dest,
                        iteration=row["iteration"],
                        candidate_id=row["candidate_id"],
                        proposal=proposal,
                    )
                    self._upsert_modification_payload(dest, row["iteration"], proposal)
            for row in _select(src, "SELECT * FROM proposals"):
                proposal = {
                    "hypothesis": row["hypothesis"],
                    "expected_effect": _loads(row["expected_effect_json"], None),
                    "risk": _loads(row["risk_json"], None),
                    "evidence_refs": _loads(row["evidence_refs_json"], None),
                }
                self._upsert_proposal_payload(
                    dest,
                    iteration=row["iteration"],
                    candidate_id=row["candidate_id"],
                    proposal={key: value for key, value in proposal.items() if value is not None},
                    proposal_id=row["proposal_id"],
                )
            for row in _select(src, "SELECT * FROM task_results"):
                dest.execute(
                    """
                    INSERT OR REPLACE INTO eval_results (
                        iteration, candidate_id, task_id, score, passed,
                        token_consuming, result_path, trace_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["iteration"],
                        row["candidate_id"],
                        row["task_id"],
                        row["score"],
                        row["passed"],
                        row["token_consuming"],
                        row["trace_path"],
                        None,
                    ),
                )
            for row in _select(src, "SELECT * FROM proposer_calls"):
                dest.execute(
                    """
                    INSERT OR REPLACE INTO proposer_calls (
                        iteration, returncode, timed_out, call_dir, workspace_dir,
                        metrics_json, usage_json, selection_policy, proposer_agent,
                        extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(row),
                )
            for row in _select(src, "SELECT * FROM iteration_diffs"):
                dest.execute(
                    """
                    INSERT INTO modifications (
                        iteration, diff_text, files_changed_json, insertions, deletions
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(iteration) DO UPDATE SET
                        diff_text = excluded.diff_text,
                        files_changed_json = excluded.files_changed_json,
                        insertions = excluded.insertions,
                        deletions = excluded.deletions
                    """,
                    tuple(row),
                )
                for changed in _loads(row["files_changed_json"], []):
                    dest.execute(
                        "INSERT OR REPLACE INTO modified_files(iteration, path) VALUES (?, ?)",
                        (row["iteration"], str(changed)),
                    )
            for row in _select(src, "SELECT * FROM file_accesses"):
                access_id = _stable_id(
                    "access",
                    self.run_id,
                    row["iteration"],
                    row["path"],
                    row["access_type"],
                    row["tool_name"],
                )
                dest.execute(
                    """
                    INSERT OR REPLACE INTO tool_accesses (
                        access_id, iteration, path, access_type, tool_name,
                        read_calls, read_lines, line_ranges_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        access_id,
                        row["iteration"],
                        row["path"],
                        row["access_type"],
                        row["tool_name"],
                        row["read_calls"],
                        row["read_lines"],
                        row["line_ranges_json"],
                    ),
                )
            for row in _select(src, "SELECT * FROM frontier_members"):
                dest.execute(
                    """
                    INSERT OR REPLACE INTO frontier_members (
                        as_of_iteration, iteration, candidate_id, role,
                        passrate, token_consuming
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    tuple(row),
                )
            for row in _select(src, "SELECT * FROM state_snapshots"):
                artifact = self._record_artifact_with_conn(
                    dest,
                    kind="state_snapshot",
                    path=f"runstore:state_snapshots/{row['iteration']}.md",
                    data=(row["state_md"] or "").encode("utf-8"),
                    iteration=row["iteration"],
                    candidate_id=None,
                )
                dest.execute(
                    """
                    INSERT OR REPLACE INTO state_snapshots (
                        iteration, state_md, artifact_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (row["iteration"], row["state_md"], artifact, row["created_at"]),
                )

    def _import_trace_index(self) -> None:
        path = self.run_dir / "traces" / "index.db"
        if not path.exists():
            return
        with sqlite3.connect(path) as src, self._connect() as dest:
            src.row_factory = sqlite3.Row
            for row in _select(src, "SELECT * FROM iteration_meta"):
                dest.execute(
                    """
                    INSERT INTO iterations (
                        iteration, patch_base, budget, selection_policy,
                        advanced_frontier, on_pareto_frontier, passrate,
                        mean_score, proposer_call_dir
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(iteration) DO UPDATE SET
                        patch_base = excluded.patch_base,
                        budget = excluded.budget,
                        selection_policy = excluded.selection_policy,
                        advanced_frontier = excluded.advanced_frontier,
                        on_pareto_frontier = excluded.on_pareto_frontier,
                        passrate = excluded.passrate,
                        mean_score = excluded.mean_score,
                        proposer_call_dir = excluded.proposer_call_dir
                    """,
                    tuple(row),
                )
            for row in _select(
                src,
                """
                SELECT t.*, d.baseline_trace, d.status, d.baseline_score, d.delta
                FROM traces t LEFT JOIN diffs d ON d.trace_id = t.trace_id
                """,
            ):
                dest.execute(
                    """
                    INSERT INTO traces (
                        trace_id, iteration, candidate_id, task_id, benchmark,
                        passed, score, jsonl_path, jsonl_lineno
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trace_id) DO UPDATE SET
                        iteration = excluded.iteration,
                        candidate_id = excluded.candidate_id,
                        task_id = excluded.task_id,
                        benchmark = excluded.benchmark,
                        passed = excluded.passed,
                        score = excluded.score,
                        jsonl_path = excluded.jsonl_path,
                        jsonl_lineno = excluded.jsonl_lineno
                    """,
                    (
                        row["trace_id"],
                        row["iteration"],
                        row["candidate_id"],
                        row["task_id"],
                        row["benchmark"],
                        row["passed"],
                        row["score"],
                        row["jsonl_path"],
                        row["jsonl_lineno"],
                    ),
                )
                dest.execute(
                    """
                    INSERT OR REPLACE INTO trace_diffs (
                        trace_id, baseline_trace, status, baseline_score, delta
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        row["trace_id"],
                        row["baseline_trace"],
                        row["status"],
                        row["baseline_score"],
                        row["delta"],
                    ),
                )
                dest.execute(
                    """
                    UPDATE eval_results
                    SET trace_id = ?
                    WHERE iteration = ? AND candidate_id = ? AND task_id = ?
                    """,
                    (
                        row["trace_id"],
                        row["iteration"],
                        row["candidate_id"],
                        row["task_id"],
                    ),
                )
            for row in _select(src, "SELECT * FROM file_modifications"):
                dest.execute(
                    "INSERT OR REPLACE INTO modified_files(iteration, path) VALUES (?, ?)",
                    tuple(row),
                )
            for row in _select(src, "SELECT * FROM iteration_diffs"):
                dest.execute(
                    """
                    INSERT INTO modifications(iteration, diff_text)
                    VALUES (?, ?)
                    ON CONFLICT(iteration) DO UPDATE SET
                        diff_text = COALESCE(modifications.diff_text, excluded.diff_text)
                    """,
                    tuple(row),
                )

    # ---- raw artifact import -------------------------------------

    def _import_trace_jsonl(self, *, iteration: int | None) -> None:
        spans_root = self.run_dir / "traces" / "spans"
        if not spans_root.exists():
            return
        pattern = f"iter_{iteration:03d}" if iteration is not None else "iter_*"
        for iter_dir in sorted(spans_root.glob(pattern)):
            for path in sorted(iter_dir.glob("*.jsonl")):
                artifact_id = self._record_file_artifact(path, kind="trace_jsonl")
                for lineno, trace in enumerate(read_jsonl(path), start=1):
                    payload = trace_to_dict(trace)
                    summary = payload.get("summary") or {}
                    passed = 1 if bool(summary.get("passed", False)) else 0
                    score = _float_or_none(summary.get("score"))
                    with self._connect() as conn:
                        conn.execute(
                            """
                            INSERT INTO traces (
                                trace_id, iteration, candidate_id, task_id,
                                benchmark, passed, score, summary_json, diff_json,
                                jsonl_artifact_id, jsonl_lineno, jsonl_path
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(trace_id) DO UPDATE SET
                                iteration = excluded.iteration,
                                candidate_id = excluded.candidate_id,
                                task_id = excluded.task_id,
                                benchmark = excluded.benchmark,
                                passed = excluded.passed,
                                score = excluded.score,
                                summary_json = excluded.summary_json,
                                diff_json = excluded.diff_json,
                                jsonl_artifact_id = excluded.jsonl_artifact_id,
                                jsonl_lineno = excluded.jsonl_lineno,
                                jsonl_path = excluded.jsonl_path
                            """,
                            (
                                trace.trace_id,
                                trace.iteration,
                                trace.candidate_id,
                                trace.task_id,
                                trace.benchmark,
                                passed,
                                score,
                                _json(payload.get("summary") or {}),
                                _json(payload.get("diff")),
                                artifact_id,
                                lineno,
                                _rel(path, self.run_dir),
                            ),
                        )
                        conn.execute(
                            """
                            UPDATE eval_results
                            SET trace_id = ?
                            WHERE iteration = ? AND candidate_id = ? AND task_id = ?
                            """,
                            (
                                trace.trace_id,
                                trace.iteration,
                                trace.candidate_id,
                                trace.task_id,
                            ),
                        )
                        conn.execute(
                            "DELETE FROM trace_spans WHERE trace_id = ?",
                            (trace.trace_id,),
                        )
                        for idx, span in enumerate(trace.spans):
                            metadata = dict(span.metadata or {})
                            parent = metadata.get("parent_span_id") or metadata.get("parent_id")
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO trace_spans (
                                    trace_id, span_ord, span_id, parent_span_id,
                                    kind, input_json, output_json, metadata_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    trace.trace_id,
                                    idx,
                                    span.id,
                                    None if parent is None else str(parent),
                                    span.kind,
                                    _json(span.input),
                                    _json(span.output),
                                    _json(metadata),
                                ),
                            )

    def _import_run_artifacts(self, *, iteration: int | None) -> None:
        for name in sorted(_RUN_LEVEL_ARTIFACT_NAMES):
            path = self.run_dir / name
            if path.exists():
                self._record_file_artifact(path, kind=_artifact_kind(path))
        call_root = self.run_dir / "proposer_calls"
        if not call_root.exists():
            return
        pattern = f"iter_{iteration:03d}" if iteration is not None else "iter_*"
        for call_dir in sorted(call_root.glob(pattern)):
            for path in self._iter_allowed_call_artifacts(call_dir):
                artifact_id = self._record_file_artifact(path, kind=_artifact_kind(path))
                self._attach_artifact_to_modification(path, artifact_id)

    def _iter_allowed_call_artifacts(self, call_dir: Path) -> Iterable[Path]:
        for path in sorted(call_dir.rglob("*")):
            if path.is_file() and _is_allowed_call_artifact(path, call_dir):
                yield path

    def _record_file_artifact(self, path: Path, *, kind: str) -> str:
        stat = path.stat()
        sha = _file_sha256(path)
        with path.open("rb") as fh:
            data = fh.read(INLINE_ARTIFACT_MAX_BYTES + 1)
        if len(data) > INLINE_ARTIFACT_MAX_BYTES:
            data = data[:PREVIEW_ARTIFACT_MAX_BYTES]
            content_encoding = "gzip+preview"
        else:
            content_encoding = "gzip"
        return self._record_artifact(
            kind=kind,
            path=_rel(path, self.run_dir),
            data=data,
            sha=sha,
            byte_count=stat.st_size,
            content_encoding=content_encoding,
            iteration=_iteration_from_path(path),
            candidate_id=_candidate_from_trace_path(path),
        )

    def _record_text_artifact(
        self,
        *,
        kind: str,
        path: str,
        text: str,
        iteration: int | None = None,
    ) -> str:
        return self._record_artifact(
            kind=kind,
            path=path,
            data=text.encode("utf-8"),
            content_encoding="gzip",
            iteration=iteration,
            candidate_id=None,
        )

    def _record_artifact(
        self,
        *,
        kind: str,
        path: str,
        data: bytes,
        iteration: int | None,
        candidate_id: str | None,
        sha: str | None = None,
        byte_count: int | None = None,
        content_encoding: str = "gzip",
    ) -> str:
        sha = sha or hashlib.sha256(data).hexdigest()
        byte_count = len(data) if byte_count is None else int(byte_count)
        artifact_id = _stable_id("artifact", self.run_id, path, sha)
        with self._connect() as conn:
            self._insert_artifact_row(
                conn,
                artifact_id=artifact_id,
                kind=kind,
                path=path,
                data=data,
                sha=sha,
                byte_count=byte_count,
                content_encoding=content_encoding,
                iteration=iteration,
                candidate_id=candidate_id,
            )
        return artifact_id

    def _record_artifact_with_conn(
        self,
        conn: sqlite3.Connection,
        *,
        kind: str,
        path: str,
        data: bytes,
        iteration: int | None,
        candidate_id: str | None,
    ) -> str:
        sha = hashlib.sha256(data).hexdigest()
        artifact_id = _stable_id("artifact", self.run_id, path, sha)
        self._insert_artifact_row(
            conn,
            artifact_id=artifact_id,
            kind=kind,
            path=path,
            data=data,
            sha=sha,
            byte_count=len(data),
            content_encoding="gzip",
            iteration=iteration,
            candidate_id=candidate_id,
        )
        return artifact_id

    def _insert_artifact_row(
        self,
        conn: sqlite3.Connection,
        *,
        artifact_id: str,
        kind: str,
        path: str,
        data: bytes,
        sha: str,
        byte_count: int,
        content_encoding: str,
        iteration: int | None,
        candidate_id: str | None,
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts (
                artifact_id, run_id, iteration, candidate_id, kind, path,
                sha256, byte_count, content_encoding, content_blob,
                imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                self.run_id,
                iteration,
                candidate_id,
                kind,
                path,
                sha,
                byte_count,
                content_encoding,
                gzip.compress(data),
                _utc_now(),
            ),
        )

    def _attach_artifact_to_modification(self, path: Path, artifact_id: str) -> None:
        iteration = _iteration_from_path(path)
        if iteration is None:
            return
        column = None
        if path.name == "diff.patch":
            column = "diff_artifact_id"
        elif path.name == "pending_eval.json":
            column = "pending_eval_artifact_id"
            payload = _read_json(path)
            if isinstance(payload, dict):
                self._upsert_modification_payload_from_pending(iteration, payload)
        if column is not None:
            with self._connect() as conn:
                conn.execute(
                    f"""
                    INSERT INTO modifications(iteration, {column})
                    VALUES (?, ?)
                    ON CONFLICT(iteration) DO UPDATE SET
                        {column} = excluded.{column}
                    """,
                    (iteration, artifact_id),
                )

    # ---- links ----------------------------------------------------

    def _rebuild_links(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM evidence_links WHERE run_id = ?", (self.run_id,))
            for row in conn.execute("SELECT * FROM proposals").fetchall():
                proposal_id = str(row["proposal_id"])
                iteration = int(row["iteration"])
                self._link(
                    conn,
                    source_type="proposer_call",
                    source_id=_call_id(self.run_id, iteration),
                    target_type="proposal",
                    target_id=proposal_id,
                    relation="proposes",
                    provenance="iteration",
                    confidence=1.0,
                )
                self._link(
                    conn,
                    source_type="proposal",
                    source_id=proposal_id,
                    target_type="modification",
                    target_id=_mod_id(self.run_id, iteration),
                    relation="implemented_by",
                    provenance="iteration",
                    confidence=1.0,
                )
                if row["candidate_id"]:
                    self._link(
                        conn,
                        source_type="proposal",
                        source_id=proposal_id,
                        target_type="candidate",
                        target_id=_candidate_id(
                            self.run_id,
                            iteration,
                            str(row["candidate_id"]),
                        ),
                        relation="generates_candidate",
                        provenance="candidate_id",
                        confidence=1.0,
                    )
            for row in conn.execute("SELECT iteration, candidate_id FROM candidates").fetchall():
                candidate_entity = _candidate_id(
                    self.run_id,
                    row["iteration"],
                    str(row["candidate_id"]),
                )
                self._link(
                    conn,
                    source_type="modification",
                    source_id=_mod_id(self.run_id, row["iteration"]),
                    target_type="candidate",
                    target_id=candidate_entity,
                    relation="produces_candidate",
                    provenance="iteration",
                    confidence=1.0,
                )
                for eval_row in conn.execute(
                    """
                    SELECT task_id FROM eval_results
                    WHERE iteration = ? AND candidate_id = ?
                    """,
                    (row["iteration"], row["candidate_id"]),
                ).fetchall():
                    self._link(
                        conn,
                        source_type="candidate",
                        source_id=candidate_entity,
                        target_type="eval_result",
                        target_id=_eval_id(
                            self.run_id,
                            row["iteration"],
                            str(row["candidate_id"]),
                            str(eval_row["task_id"]),
                        ),
                        relation="has_eval_result",
                        provenance="candidate_eval",
                        confidence=1.0,
                    )
            for row in conn.execute("SELECT iteration, path FROM modified_files").fetchall():
                self._link(
                    conn,
                    source_type="modification",
                    source_id=_mod_id(self.run_id, row["iteration"]),
                    target_type="file",
                    target_id=str(row["path"]),
                    relation="changes",
                    provenance="diff",
                    confidence=1.0,
                )
            for row in conn.execute("SELECT * FROM tool_accesses").fetchall():
                relation = {
                    "read": "reads_evidence",
                    "grep": "searches_evidence",
                    "write": "writes_file",
                    "changed": "changes",
                }.get(str(row["access_type"]), str(row["access_type"]))
                access_id = str(row["access_id"])
                self._link(
                    conn,
                    source_type="modification",
                    source_id=_mod_id(self.run_id, row["iteration"]),
                    target_type="tool_access",
                    target_id=access_id,
                    relation=relation,
                    provenance="agent_tool_access",
                    confidence=1.0,
                )
                artifact = self._artifact_for_access(conn, str(row["path"]))
                if artifact is not None:
                    self._link(
                        conn,
                        source_type="tool_access",
                        source_id=access_id,
                        target_type="artifact",
                        target_id=artifact["artifact_id"],
                        relation="touches_artifact",
                        provenance="path_match",
                        confidence=0.8,
                    )
                    self._link(
                        conn,
                        source_type="modification",
                        source_id=_mod_id(self.run_id, row["iteration"]),
                        target_type="artifact",
                        target_id=artifact["artifact_id"],
                        relation=relation,
                        provenance="agent_tool_access_path_match",
                        confidence=0.8,
                    )
            for row in conn.execute("SELECT trace_id, jsonl_artifact_id FROM traces").fetchall():
                artifact_id = row["jsonl_artifact_id"]
                if artifact_id:
                    self._link(
                        conn,
                        source_type="trace",
                        source_id=str(row["trace_id"]),
                        target_type="artifact",
                        target_id=str(artifact_id),
                        relation="stored_in",
                        provenance="trace_jsonl",
                        confidence=1.0,
                    )
            for row in conn.execute(
                "SELECT iteration, candidate_id, task_id, trace_id FROM eval_results WHERE trace_id IS NOT NULL"
            ).fetchall():
                self._link(
                    conn,
                    source_type="eval_result",
                    source_id=_eval_id(
                        self.run_id,
                        row["iteration"],
                        row["candidate_id"],
                        row["task_id"],
                    ),
                    target_type="trace",
                    target_id=str(row["trace_id"]),
                    relation="has_trace",
                    provenance="trace_index",
                    confidence=1.0,
                )
            for row in conn.execute("SELECT iteration FROM modifications").fetchall():
                self._link(
                    conn,
                    source_type="proposer_call",
                    source_id=_call_id(self.run_id, row["iteration"]),
                    target_type="modification",
                    target_id=_mod_id(self.run_id, row["iteration"]),
                    relation="produces",
                    provenance="iteration",
                    confidence=1.0,
                )

    def _rebuild_proposal_outcomes(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM proposal_outcomes")
            proposals = conn.execute(
                """
                SELECT p.proposal_id, p.iteration, p.candidate_id,
                       c.passrate, c.average_score, c.token_consuming
                FROM proposals p
                LEFT JOIN candidates c
                  ON c.iteration = p.iteration AND c.candidate_id = p.candidate_id
                WHERE p.candidate_id IS NOT NULL
                ORDER BY p.iteration ASC, p.candidate_id ASC
                """
            ).fetchall()
            for proposal in proposals:
                base = self._base_candidate_for_iteration(
                    conn,
                    iteration=int(proposal["iteration"]),
                )
                counts = {
                    "breakthrough_count": 0,
                    "regression_count": 0,
                    "stable_pass_count": 0,
                    "persistent_fail_count": 0,
                    "task_count": 0,
                }
                if base is not None:
                    counts = self._proposal_task_counts(
                        conn,
                        left_iteration=int(base["iteration"]),
                        left_candidate_id=str(base["candidate_id"]),
                        right_iteration=int(proposal["iteration"]),
                        right_candidate_id=str(proposal["candidate_id"]),
                    )
                passrate_delta = _delta(
                    proposal["passrate"],
                    base["passrate"] if base else None,
                )
                average_score_delta = _delta(
                    proposal["average_score"],
                    base["average_score"] if base else None,
                )
                token_delta = _int_delta(
                    proposal["token_consuming"],
                    base["token_consuming"] if base else None,
                )
                summary = {
                    "base_iteration": None if base is None else int(base["iteration"]),
                    "base_candidate_id": None if base is None else str(base["candidate_id"]),
                    "passrate_delta": passrate_delta,
                    "average_score_delta": average_score_delta,
                    "token_delta": token_delta,
                    **counts,
                }
                conn.execute(
                    """
                    INSERT OR REPLACE INTO proposal_outcomes (
                        proposal_id, iteration, candidate_id, base_iteration,
                        base_candidate_id, passrate_delta, average_score_delta,
                        token_delta, breakthrough_count, regression_count,
                        stable_pass_count, persistent_fail_count, task_count,
                        outcome_summary_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        proposal["proposal_id"],
                        proposal["iteration"],
                        proposal["candidate_id"],
                        summary["base_iteration"],
                        summary["base_candidate_id"],
                        passrate_delta,
                        average_score_delta,
                        token_delta,
                        counts["breakthrough_count"],
                        counts["regression_count"],
                        counts["stable_pass_count"],
                        counts["persistent_fail_count"],
                        counts["task_count"],
                        _json(summary),
                    ),
                )

    def _base_candidate_for_iteration(
        self,
        conn: sqlite3.Connection,
        *,
        iteration: int,
    ) -> sqlite3.Row | None:
        meta = conn.execute(
            """
            SELECT base_iteration, base_candidate_id, patch_base
            FROM iterations
            WHERE iteration = ?
            """,
            (int(iteration),),
        ).fetchone()
        base_iteration = None
        base_candidate_id = None
        if meta is not None:
            base_iteration = meta["base_iteration"]
            base_candidate_id = meta["base_candidate_id"]
            if base_iteration is None:
                base_iteration = meta["patch_base"]
        if base_iteration is None:
            candidate_row = conn.execute(
                """
                SELECT MAX(iteration) AS iteration
                FROM candidates
                WHERE iteration < ?
                """,
                (int(iteration),),
            ).fetchone()
            base_iteration = candidate_row["iteration"] if candidate_row else None
        if base_iteration is None:
            return None
        if base_candidate_id:
            row = conn.execute(
                """
                SELECT iteration, candidate_id, passrate, average_score, token_consuming
                FROM candidates
                WHERE iteration = ? AND candidate_id = ?
                """,
                (int(base_iteration), str(base_candidate_id)),
            ).fetchone()
            if row is not None:
                return row
        return conn.execute(
            """
            SELECT iteration, candidate_id, passrate, average_score, token_consuming
            FROM candidates
            WHERE iteration = ?
            ORDER BY passrate DESC, average_score DESC, candidate_id DESC
            LIMIT 1
            """,
            (int(base_iteration),),
        ).fetchone()

    def _proposal_task_counts(
        self,
        conn: sqlite3.Connection,
        *,
        left_iteration: int,
        left_candidate_id: str,
        right_iteration: int,
        right_candidate_id: str,
    ) -> dict[str, int]:
        rows: dict[str, dict[str, bool | None]] = {}
        for side, iteration, candidate_id in (
            ("left", left_iteration, left_candidate_id),
            ("right", right_iteration, right_candidate_id),
        ):
            for row in conn.execute(
                """
                SELECT task_id, passed
                FROM eval_results
                WHERE iteration = ? AND candidate_id = ?
                """,
                (iteration, candidate_id),
            ).fetchall():
                entry = rows.setdefault(str(row["task_id"]), {"left": None, "right": None})
                entry[side] = bool(row["passed"])
        counts = {
            "breakthrough_count": 0,
            "regression_count": 0,
            "stable_pass_count": 0,
            "persistent_fail_count": 0,
            "task_count": len(rows),
        }
        for item in rows.values():
            left = item["left"]
            right = item["right"]
            if left is False and right is True:
                counts["breakthrough_count"] += 1
            elif left is True and right is False:
                counts["regression_count"] += 1
            elif left is True and right is True:
                counts["stable_pass_count"] += 1
            elif left is False and right is False:
                counts["persistent_fail_count"] += 1
        return counts

    def _artifact_for_access(self, conn: sqlite3.Connection, access_path: str) -> sqlite3.Row | None:
        candidates = _path_candidates(access_path)
        for candidate in candidates:
            row = conn.execute(
                """
                SELECT artifact_id, path FROM artifacts
                WHERE path = ? OR path LIKE ?
                ORDER BY byte_count DESC
                LIMIT 1
                """,
                (candidate, f"%/{candidate}"),
            ).fetchone()
            if row is not None:
                return row
        return None

    def _link(
        self,
        conn: sqlite3.Connection,
        *,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation: str,
        provenance: str,
        confidence: float,
    ) -> None:
        link_id = _stable_id(
            "link",
            self.run_id,
            source_type,
            source_id,
            target_type,
            target_id,
            relation,
            provenance,
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO evidence_links (
                link_id, run_id, source_type, source_id, target_type,
                target_id, relation, provenance, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                link_id,
                self.run_id,
                source_type,
                source_id,
                target_type,
                target_id,
                relation,
                provenance,
                float(confidence),
                _utc_now(),
            ),
        )

    # ---- proposal metadata ---------------------------------------

    def _upsert_modification_payload_from_pending(
        self,
        iteration: int,
        pending: dict[str, Any],
    ) -> None:
        candidates = pending.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return
        first = candidates[0]
        if isinstance(first, dict):
            with self._connect() as conn:
                self._upsert_modification_payload(conn, iteration, first)

    def _upsert_modification_payload(
        self,
        conn: sqlite3.Connection,
        iteration: int,
        proposal: dict[str, Any],
    ) -> None:
        hypothesis = proposal.get("hypothesis") or proposal.get("description")
        conn.execute(
            """
            INSERT INTO modifications(iteration, hypothesis, proposal_json)
            VALUES (?, ?, ?)
            ON CONFLICT(iteration) DO UPDATE SET
                hypothesis = COALESCE(excluded.hypothesis, modifications.hypothesis),
                proposal_json = COALESCE(excluded.proposal_json, modifications.proposal_json)
            """,
            (
                int(iteration),
                None if hypothesis is None else str(hypothesis),
                _json(proposal),
            ),
        )

    def _upsert_proposal_payload(
        self,
        conn: sqlite3.Connection,
        *,
        iteration: int,
        candidate_id: str | None,
        proposal: dict[str, Any],
        proposal_id: str | None = None,
    ) -> None:
        cid = None if candidate_id is None else str(candidate_id)
        proposal_id = proposal_id or _proposal_id(self.run_id, iteration, cid)
        hypothesis = proposal.get("hypothesis") or proposal.get("description")
        changes = proposal.get("changes") or proposal.get("implementation")
        conn.execute(
            """
            INSERT INTO proposals (
                proposal_id, iteration, candidate_id, proposal_name,
                scaffold_name, hypothesis, changes, expected_effect_json,
                risk_json, evidence_refs_json, proposal_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(proposal_id) DO UPDATE SET
                iteration = excluded.iteration,
                candidate_id = COALESCE(excluded.candidate_id, proposals.candidate_id),
                proposal_name = COALESCE(excluded.proposal_name, proposals.proposal_name),
                scaffold_name = COALESCE(excluded.scaffold_name, proposals.scaffold_name),
                hypothesis = COALESCE(excluded.hypothesis, proposals.hypothesis),
                changes = COALESCE(excluded.changes, proposals.changes),
                expected_effect_json = COALESCE(excluded.expected_effect_json, proposals.expected_effect_json),
                risk_json = COALESCE(excluded.risk_json, proposals.risk_json),
                evidence_refs_json = COALESCE(excluded.evidence_refs_json, proposals.evidence_refs_json),
                proposal_json = excluded.proposal_json
            """,
            (
                proposal_id,
                int(iteration),
                cid,
                _optional_str(proposal.get("name") or proposal.get("proposal_name")),
                _optional_str(proposal.get("scaffold_name")),
                _optional_str(hypothesis),
                _optional_str(changes),
                _json_or_none(proposal.get("expected_effect")),
                _json_or_none(proposal.get("risk")),
                _json_or_none(proposal.get("evidence_refs") or proposal.get("evidence")),
                _json(proposal),
            ),
        )


def _select(conn: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError:
        return []


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_allowed_call_artifact(path: Path, call_dir: Path) -> bool:
    try:
        rel = path.relative_to(call_dir)
    except ValueError:
        return False
    parts = rel.parts
    if not parts:
        return False
    if any(part in _SKIP_ARTIFACT_PARTS for part in parts):
        return False
    if path.name in {"runstore.db", "index.db", "evidence_store.db"}:
        return False
    if path.suffix in _SKIP_ARTIFACT_SUFFIXES:
        return False
    if len(parts) == 1:
        return path.name in _CALL_LEVEL_ARTIFACT_NAMES
    if parts[0] == "eval":
        return path.name in _EVAL_ARTIFACT_NAMES
    if parts[0] == "agent":
        return path.name in _AGENT_ARTIFACT_NAMES
    return False


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _delta(right: Any, left: Any) -> float | None:
    right_value = _float_or_none(right)
    left_value = _float_or_none(left)
    if right_value is None or left_value is None:
        return None
    return right_value - left_value


def _int_delta(right: Any, left: Any) -> int | None:
    try:
        if right is None or left is None:
            return None
        return int(right) - int(left)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _json_or_none(value: Any) -> str | None:
    return None if value is None else _json(value)


def _stable_id(*parts: object) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mod_id(run_id: str, iteration: int) -> str:
    return f"{run_id}:mod:{int(iteration):03d}"


def _call_id(run_id: str, iteration: int) -> str:
    return f"{run_id}:call:{int(iteration):03d}"


def _proposal_id(run_id: str, iteration: int, candidate_id: str | None) -> str:
    suffix = candidate_id or "pending"
    return f"{run_id}:proposal:{int(iteration):03d}:{suffix}"


def _candidate_id(run_id: str, iteration: int, candidate_id: str) -> str:
    return f"{run_id}:candidate:{int(iteration):03d}:{candidate_id}"


def _eval_id(run_id: str, iteration: int, candidate_id: str, task_id: str) -> str:
    return f"{run_id}:eval:{int(iteration):03d}:{candidate_id}:{task_id}"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iteration_from_path(path: Path) -> int | None:
    match = re.search(r"iter_(\d{3,})", str(path))
    if not match:
        return None
    return int(match.group(1))


def _candidate_from_trace_path(path: Path) -> str | None:
    if path.suffix != ".jsonl":
        return None
    if "traces" not in path.parts or "spans" not in path.parts:
        return None
    return path.stem


def _artifact_kind(path: Path) -> str:
    name = path.name
    if name == "diff.patch":
        return "diff_patch"
    if name == "pending_eval.json":
        return "pending_eval"
    if name == "tool_access.json":
        return "tool_access"
    if name == "state.md":
        return "state_md"
    if name.endswith(".jsonl"):
        return "jsonl"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".md"):
        return "markdown"
    if name.endswith(".patch") or name.endswith(".diff"):
        return "diff_patch"
    return "file"


def _path_candidates(path: str) -> list[str]:
    text = path.strip()
    if not text:
        return []
    out = [text.lstrip("/")]
    for marker in (
        "workspace/",
        "summaries/",
        "traces/",
        "reference_iterations/",
        "source_snapshot/",
    ):
        idx = text.find(marker)
        if idx >= 0:
            out.append(text[idx:])
    # Docker paths often start at /workspace; host paths may contain
    # proposer_calls/.../workspace. Matching suffixes handles both.
    return list(dict.fromkeys(out))
