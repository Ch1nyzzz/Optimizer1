"""Markdown diagnostic renderer.

Reads `traces/index.db` plus per-iteration `spans/iter_NNN/*.jsonl` and
writes `traces/diagnostic/iter_NNN.md`. The markdown is the single
artifact the proposer (or a human) reads to understand "what happened
this iteration vs the baseline" — it is intentionally NOT inlined into
the proposer prompt; the prompt only cites the file path.

Sections (in order — most actionable first):
  - Diff Summary (counts + Δ averages per status)
  - REGRESSED        top-K by |Δ|
  - PERSISTENT_FAIL  top-K by |Δ|
  - BREAKTHROUGH     top-K by |Δ|
  - NO_BASELINE      counts only (new task ids without a baseline match)
  - STABLE_PASS      counts only

Baseline runs (manifest.is_baseline_run = True) collapse all of the
above into a single counts-only header — there is no diff to render.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .diff import (
    STATUS_BASELINE,
    STATUS_BREAKTHROUGH,
    STATUS_NO_BASELINE,
    STATUS_PERSISTENT_FAIL,
    STATUS_REGRESSED,
    STATUS_STABLE_PASS,
)
from .schema import Trace, read_jsonl


@dataclass(frozen=True)
class RenderConfig:
    top_k: int = 5
    truncate_question: int = 200
    truncate_prediction: int = 200
    truncate_doc_content: int = 120
    retrieval_top: int = 3


@dataclass(frozen=True)
class _TraceRow:
    trace_id: str
    task_id: str
    candidate_id: str
    iteration: int
    passed: bool
    score: float
    status: str
    baseline_score: float | None
    delta: float | None
    jsonl_path: str
    jsonl_lineno: int


_DETAIL_ORDER = (
    STATUS_REGRESSED,
    STATUS_PERSISTENT_FAIL,
    STATUS_BREAKTHROUGH,
)
_COUNTS_ONLY = (STATUS_NO_BASELINE, STATUS_STABLE_PASS)


class Renderer:
    def __init__(
        self,
        traces_dir: Path,
        *,
        config: RenderConfig | None = None,
    ) -> None:
        self.root = Path(traces_dir)
        self.config = config or RenderConfig()

    @property
    def db_path(self) -> Path:
        return self.root / "index.db"

    @property
    def diagnostic_dir(self) -> Path:
        return self.root / "diagnostic"

    # ------------------------------------------------------------------

    def render_iteration(self, iteration: int) -> Path:
        rows = self._query_rows(iteration)
        manifest = self._read_manifest()

        out_path = self.diagnostic_dir / f"iter_{iteration:03d}.md"
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)

        # An iteration is a "baseline iteration" iff every recorded
        # trace was tagged baseline. This handles two cases naturally:
        #   - iter_0 of a run with no external baseline (all 'baseline')
        #   - any iteration of an empty/zero-trace run
        # iter>=1 normally has diff statuses → falls through to
        # _render_diff_run.
        all_baseline = bool(rows) and all(
            row.status == STATUS_BASELINE for row in rows
        )
        if not rows or all_baseline:
            text = self._render_baseline_run(iteration, rows, manifest)
        else:
            text = self._render_diff_run(iteration, rows, manifest)

        out_path.write_text(text, encoding="utf-8")
        return out_path

    # ---- queries -----------------------------------------------------

    def _query_rows(self, iteration: int) -> list[_TraceRow]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            raw = conn.execute(
                "SELECT t.trace_id, t.task_id, t.candidate_id, t.iteration, "
                "       t.passed, t.score, "
                "       d.status, d.baseline_score, d.delta, "
                "       t.jsonl_path, t.jsonl_lineno "
                "FROM traces t JOIN diffs d USING (trace_id) "
                "WHERE t.iteration = ? "
                "ORDER BY t.task_id",
                (iteration,),
            ).fetchall()
        return [
            _TraceRow(
                trace_id=str(row["trace_id"]),
                task_id=str(row["task_id"]),
                candidate_id=str(row["candidate_id"]),
                iteration=int(row["iteration"]),
                passed=bool(row["passed"]),
                score=float(row["score"] or 0.0),
                status=str(row["status"]),
                baseline_score=(
                    float(row["baseline_score"])
                    if row["baseline_score"] is not None
                    else None
                ),
                delta=float(row["delta"]) if row["delta"] is not None else None,
                jsonl_path=str(row["jsonl_path"]),
                jsonl_lineno=int(row["jsonl_lineno"]),
            )
            for row in raw
        ]

    def _read_manifest(self) -> dict[str, str | None]:
        if not self.db_path.exists():
            return {}
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT key, value FROM manifest").fetchall()
        return {str(row["key"]): row["value"] for row in rows}

    # ---- baseline rendering ------------------------------------------

    def _render_baseline_run(
        self,
        iteration: int,
        rows: list[_TraceRow],
        manifest: dict,
    ) -> str:
        lines = [
            f"# Iteration {iteration} — Baseline Run",
            "",
            f"Benchmark: {manifest.get('benchmark') or 'unknown'}  •  "
            f"Traces: {len(rows)}",
            "",
            "_This run is itself a baseline (no `--trace-baseline` was set), "
            "so no diff is computed. Use these traces as the reference for "
            "future runs._",
            "",
            "## Counts",
            f"- passed: {sum(1 for r in rows if r.passed)}",
            f"- failed: {sum(1 for r in rows if not r.passed)}",
            "",
        ]
        return "\n".join(lines)

    # ---- diff rendering ----------------------------------------------

    def _render_diff_run(
        self,
        iteration: int,
        rows: list[_TraceRow],
        manifest: dict,
    ) -> str:
        baseline_path = manifest.get("baseline_path") or "(none)"
        by_status: dict[str, list[_TraceRow]] = {}
        for row in rows:
            by_status.setdefault(row.status, []).append(row)

        lines = [
            f"# Iteration {iteration} — Diagnostic vs baseline",
            "",
            f"Benchmark: {manifest.get('benchmark') or 'unknown'}  •  "
            f"Baseline: `{baseline_path}`  •  "
            f"Traces: {len(rows)}",
            "",
            "## Diff Summary",
        ]
        lines.extend(self._summary_lines(by_status))
        lines.append("")

        # Detailed sections — top_k each.
        traces_cache = self._load_traces_for_rows(rows)
        for status in _DETAIL_ORDER:
            section_rows = by_status.get(status, [])
            if not section_rows:
                continue
            lines.append("---")
            lines.append("")
            lines.extend(
                self._detail_section(status, section_rows, traces_cache)
            )
            lines.append("")

        # Counts-only sections.
        for status in _COUNTS_ONLY:
            section_rows = by_status.get(status, [])
            if not section_rows:
                continue
            lines.append("---")
            lines.append("")
            header = self._status_header(status)
            lines.append(f"## {header}")
            lines.append("")
            lines.append(f"_{len(section_rows)} traces — counts only._")
            lines.append("")

        return "\n".join(lines)

    # ---- summary -----------------------------------------------------

    def _summary_lines(
        self, by_status: dict[str, list[_TraceRow]]
    ) -> list[str]:
        out: list[str] = []
        ordered = (
            STATUS_REGRESSED,
            STATUS_BREAKTHROUGH,
            STATUS_PERSISTENT_FAIL,
            STATUS_STABLE_PASS,
            STATUS_NO_BASELINE,
        )
        for status in ordered:
            rows = by_status.get(status, [])
            if not rows:
                continue
            count = len(rows)
            deltas = [r.delta for r in rows if r.delta is not None]
            if deltas:
                avg = sum(deltas) / len(deltas)
                tail = f"  (Δ avg {avg:+.3f})"
            else:
                tail = ""
            out.append(f"- {status}: {count}{tail}")
        return out

    # ---- per-status detail rendering ---------------------------------

    def _detail_section(
        self,
        status: str,
        rows: list[_TraceRow],
        traces_cache: dict[str, Trace],
    ) -> list[str]:
        marker = {
            STATUS_REGRESSED: "▼",
            STATUS_PERSISTENT_FAIL: "▼",
            STATUS_BREAKTHROUGH: "▲",
        }.get(status, "•")
        header = self._status_header(status)
        sorted_rows = sorted(
            rows,
            key=lambda r: abs(r.delta) if r.delta is not None else 0.0,
            reverse=True,
        )
        chosen = sorted_rows[: self.config.top_k]

        out = [f"## {marker} {header}  (top {len(chosen)} of {len(rows)})", ""]
        for row in chosen:
            trace = traces_cache.get(row.trace_id)
            out.extend(self._format_trace_block(row, trace))
            out.append("")
        return out

    def _format_trace_block(
        self,
        row: _TraceRow,
        trace: Trace | None,
    ) -> list[str]:
        flip = self._flip_label(row)
        score_arrow = self._score_arrow(row)
        block = [
            f"### {row.task_id}  {flip}  ({score_arrow})",
        ]
        if trace is None:
            block.append("_(trace body unavailable — jsonl missing)_")
            return block

        summary: dict[str, Any] = trace.summary or {}
        question = self._truncate(
            str(summary.get("question") or ""), self.config.truncate_question
        )
        gold = str(summary.get("gold") or "")
        prediction = self._truncate(
            str(summary.get("prediction") or ""), self.config.truncate_prediction
        )
        block.append(f"- Q: {question}")
        block.append(f"- Gold: {gold!r}")
        block.append(f"- Pred: {prediction!r}")

        retrieval_lines = self._retrieval_lines(trace)
        block.extend(retrieval_lines)

        prompt_tokens = summary.get("prompt_tokens")
        completion_tokens = summary.get("completion_tokens")
        if prompt_tokens is not None or completion_tokens is not None:
            block.append(
                f"- Tokens: {prompt_tokens or 0} prompt / "
                f"{completion_tokens or 0} completion"
            )
        return block

    def _retrieval_lines(self, trace: Trace) -> list[str]:
        retrieval_span = next(
            (span for span in trace.spans if span.kind == "retrieval"),
            None,
        )
        if retrieval_span is None or not isinstance(retrieval_span.output, dict):
            return []
        documents = retrieval_span.output.get("documents") or []
        if not isinstance(documents, list) or not documents:
            return ["- Retrieval: 0 docs"]
        total = retrieval_span.output.get("total_returned", len(documents))
        out = [f"- Retrieval ({total} docs, top-{min(self.config.retrieval_top, len(documents))}):"]
        for doc in documents[: self.config.retrieval_top]:
            if not isinstance(doc, dict):
                continue
            score = doc.get("score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
            source = str(doc.get("source") or "?")
            content = self._truncate(
                str(doc.get("content") or ""),
                self.config.truncate_doc_content,
            )
            rank = doc.get("rank") or "?"
            out.append(f"  - [{rank}] {score_str} {source}  {content!r}")
        return out

    # ---- formatting helpers ------------------------------------------

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = text.replace("\n", " ").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    @staticmethod
    def _flip_label(row: _TraceRow) -> str:
        if row.status == STATUS_REGRESSED:
            return "passed→failed"
        if row.status == STATUS_BREAKTHROUGH:
            return "failed→passed"
        if row.status == STATUS_PERSISTENT_FAIL:
            return "failed→failed"
        if row.status == STATUS_STABLE_PASS:
            return "passed→passed"
        if row.status == STATUS_NO_BASELINE:
            return "(no baseline)"
        return row.status

    @staticmethod
    def _score_arrow(row: _TraceRow) -> str:
        baseline = row.baseline_score
        delta = row.delta
        if baseline is None and delta is None:
            return f"score {row.score:.2f}"
        if baseline is None:
            return f"score {row.score:.2f}"
        return f"{baseline:.2f} → {row.score:.2f}, Δ {delta:+.2f}"

    @staticmethod
    def _status_header(status: str) -> str:
        return status.replace("_", " ").upper()

    # ---- jsonl loading -----------------------------------------------

    def _load_traces_for_rows(
        self,
        rows: Iterable[_TraceRow],
    ) -> dict[str, Trace]:
        # Group by jsonl_path to read each file once.
        by_path: dict[str, list[str]] = {}
        for row in rows:
            by_path.setdefault(row.jsonl_path, []).append(row.trace_id)

        cache: dict[str, Trace] = {}
        for path_str, wanted in by_path.items():
            path = Path(path_str)
            if not path.exists():
                continue
            wanted_set = set(wanted)
            for trace in read_jsonl(path):
                if trace.trace_id in wanted_set:
                    cache[trace.trace_id] = trace
        return cache
