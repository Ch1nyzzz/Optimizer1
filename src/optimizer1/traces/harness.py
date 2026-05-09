"""TraceHarness — top-level entry point used by the optimizer.

In M1 the harness is write-only: it ensures `traces/manifest.json` exists
and records each iteration's traces under `traces/spans/`. The indexer
(SQLite + diff) and renderer (markdown) come in M2 / M3.

Lifecycle:

    harness = TraceHarness(
        run_dir=run_dir,
        benchmark="longmemeval",
        baseline_path=Path("...optional..."),
    )
    harness.record_iteration(iteration=N, candidates=[...])

The optimizer holds one harness instance for the duration of a run and
calls `record_iteration` once per evaluated batch (mirrors
`write_post_eval_artifacts`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from optimizer1.schemas import CandidateResult

from .adapter import get_adapter
from .baseline import Baseline
from .embeddings import DiffEmbedder
from .indexer import Indexer
from .recorder import Recorder
from .renderer import RenderConfig, Renderer
from .schema import SCHEMA_VERSION

BACKEND_VERSION = "1.0"


class TraceHarness:
    def __init__(
        self,
        *,
        run_dir: Path,
        benchmark: str,
        baseline_path: Path | None = None,
        render_config: RenderConfig | None = None,
        diff_embedder: DiffEmbedder | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "traces"
        self.benchmark = benchmark
        self.baseline_path = Path(baseline_path) if baseline_path else None
        # No baseline path provided ⇒ this run *is* the baseline.
        self.is_baseline_run = self.baseline_path is None
        self.adapter = get_adapter(benchmark)
        self.recorder = Recorder(self.root)
        self.baseline = (
            Baseline.load(self.baseline_path)
            if self.baseline_path is not None
            else Baseline.empty()
        )
        self.indexer = Indexer(
            self.root / "index.db",
            baseline=self.baseline,
        )
        self.renderer = Renderer(self.root, config=render_config)
        self.diff_embedder = diff_embedder
        self._manifest_written = False
        # When no external baseline is provided, iter_0 of *this* run
        # acts as the implicit baseline. We lazy-load it the first time
        # an iter>=1 needs it and cache here.
        self._self_baseline_loaded = False

    # ---- manifest -------------------------------------------------

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def ensure_manifest(self) -> None:
        if self._manifest_written and self.manifest_path().exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "backend_version": BACKEND_VERSION,
            "schema_version": SCHEMA_VERSION,
            "benchmark": self.benchmark,
            "baseline_path": str(self.baseline_path) if self.baseline_path else None,
            "is_baseline_run": self.is_baseline_run,
        }
        self.manifest_path().write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Mirror manifest into the SQLite index so consumers can read
        # either source.
        self.indexer.write_manifest(manifest)
        self._manifest_written = True

    # ---- recording ------------------------------------------------

    def record_iteration(
        self,
        *,
        iteration: int,
        candidates: list[CandidateResult],
    ) -> list[Path]:
        """Build traces for every candidate and write them to disk.

        Returns the list of jsonl paths written (one per candidate).
        """

        self.ensure_manifest()
        written: list[Path] = []
        for candidate in candidates:
            payload = self._read_result(candidate)
            tasks = payload.get("tasks") if isinstance(payload, dict) else None
            if not isinstance(tasks, list):
                continue
            traces = []
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                traces.append(
                    self.adapter.build_trace(
                        iteration=iteration,
                        candidate_id=candidate.candidate_id,
                        task=task,
                    )
                )
            path = self.recorder.write(
                iteration=iteration,
                candidate_id=candidate.candidate_id,
                traces=traces,
            )
            written.append(path)

        if written:
            # iter_0 of a run with no external baseline IS the baseline.
            # iter_N>=1 diff against either the external baseline (if
            # supplied) or the current run's own iter_0 (lazy-loaded).
            treat_as_baseline = (
                iteration == 0 and self.baseline_path is None
            )
            if (
                not treat_as_baseline
                and self.baseline_path is None
                and not self._self_baseline_loaded
            ):
                self._load_self_iter0_baseline()
            self.indexer.materialize_iteration(
                iteration=iteration,
                jsonl_paths=written,
                treat_as_baseline=treat_as_baseline,
            )
            diff_text = self._read_diff_text(iteration)
            self.indexer.record_file_modifications(
                iteration=iteration,
                paths=_paths_from_diff_text(diff_text),
            )
            if self.diff_embedder is not None and diff_text:
                emb = self.diff_embedder.embed(diff_text)
                if emb is not None:
                    self.indexer.record_diff_embedding(
                        iteration=iteration,
                        model=emb.model,
                        dim=emb.dim,
                        diff_text=emb.diff_text,
                        embedding=emb.to_bytes(),
                    )
            self.renderer.render_iteration(iteration)
        return written

    def _read_diff_text(self, iteration: int) -> str:
        diff_path = (
            self.run_dir
            / "proposer_calls"
            / f"iter_{iteration:03d}"
            / "diff.patch"
        )
        if not diff_path.exists():
            return ""
        return diff_path.read_text(encoding="utf-8", errors="replace")

    def _load_self_iter0_baseline(self) -> None:
        """Populate `self.baseline` from this run's iter_0 traces.

        Used when no `--trace-baseline` was supplied, so the indexer
        can compute diffs for iter>=1 against the local scaffold.
        Idempotent.
        """

        if self._self_baseline_loaded:
            return
        iter0_dir = self.recorder.spans_dir(iteration=0)
        self.baseline = Baseline.from_jsonl_dir(iter0_dir)
        self.indexer.baseline = self.baseline
        self._self_baseline_loaded = True

    # ---- helpers --------------------------------------------------

    @staticmethod
    def _read_result(candidate: CandidateResult) -> dict[str, Any]:
        try:
            return json.loads(Path(candidate.result_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


def _paths_from_diff_text(diff_text: str) -> list[str]:
    if not diff_text:
        return []
    out: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) >= 4:
            out.append(parts[3].removeprefix("b/"))
    return sorted(set(out))
