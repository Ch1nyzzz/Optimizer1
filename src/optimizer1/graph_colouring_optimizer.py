"""Graph-colouring optimization entry point for source-backed C++ heuristics.

Mirrors SwebenchOptimizer's structure: subclass LocomoOptimizer, swap in the
graph-colouring evaluator and seed loader, copy the upstream graph-colouring
tree into the per-iteration source snapshot so the proposer can edit it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optimizer1.benchmark_workspaces import (
    BenchmarkWorkspaceSpec,
    GRAPH_COLOURING_WORKSPACE_SPEC,
)
from optimizer1.graph_colouring_overlay import apply_overlay as apply_graph_colouring_overlay
from optimizer1.graph_colouring import (
    DEFAULT_GRAPH_COLOURING_NAME,
    DEFAULT_GRAPH_COLOURING_SOURCE_PATH,
    DEFAULT_TARGET_ALGORITHM,
    DEFAULT_TIME_BUDGET_S,
    DEFAULT_COMPILE_TIMEOUT_S,
    DEFAULT_TRAIN_INSTANCES,
    DEFAULT_TEST_INSTANCES,
    GraphColouringSourceRunner,
    GraphInstance,
    load_graph_colouring_instances,
    run_graph_colouring_frontier,
)
from optimizer1.optimizer import LocomoOptimizer, OptimizerConfig
from optimizer1.schemas import CandidateResult


@dataclass(frozen=True)
class GraphColouringOptimizerConfig(OptimizerConfig):
    """Configuration for source-backed graph-colouring optimization."""

    graph_colouring_source_path: Path = DEFAULT_GRAPH_COLOURING_SOURCE_PATH
    target_algorithm: str = DEFAULT_TARGET_ALGORITHM
    time_budget_s: int = DEFAULT_TIME_BUDGET_S
    compile_timeout_s: int = DEFAULT_COMPILE_TIMEOUT_S
    train_instances: tuple[str, ...] = DEFAULT_TRAIN_INSTANCES
    test_instances: tuple[str, ...] = DEFAULT_TEST_INSTANCES
    force: bool = False
    scaffolds: tuple[str, ...] = (DEFAULT_GRAPH_COLOURING_NAME,)
    progressive_target_system: str = DEFAULT_GRAPH_COLOURING_NAME


class GraphColouringOptimizer(LocomoOptimizer):
    """Proposer loop for graph-colouring source-backed C++ heuristics."""

    workspace_spec: BenchmarkWorkspaceSpec = GRAPH_COLOURING_WORKSPACE_SPEC
    config: GraphColouringOptimizerConfig

    def __init__(self, config: GraphColouringOptimizerConfig) -> None:
        super().__init__(config)

    def _instance_names_for_split(self, split: str) -> tuple[str, ...]:
        if split == "test":
            return self.config.test_instances
        return self.config.train_instances

    def _load_examples(self) -> list[GraphInstance]:
        return load_graph_colouring_instances(
            source_dir=self.config.graph_colouring_source_path,
            instance_names=self._instance_names_for_split(self.config.split),
            split=self.config.split,
        )

    def _run_seed_frontier(self) -> dict[str, Any]:
        return run_graph_colouring_frontier(
            out_dir=self.run_dir,
            source_path=self.config.graph_colouring_source_path,
            instance_names=self._instance_names_for_split(self.config.split),
            split=self.config.split,
            target_algorithm=self.config.target_algorithm,
            time_budget_s=self.config.time_budget_s,
            compile_timeout_s=self.config.compile_timeout_s,
            max_eval_workers=self.config.max_eval_workers,
            dry_run=self.config.dry_run,
            force=self.config.force,
            pareto_quality_threshold=self.config.pareto_quality_threshold,
        )

    def _benchmark_prompt_name(self) -> str:
        return "graph-colouring DIMACS heuristic optimization"

    def _raw_data_policy_name(self) -> str:
        return "graph-colouring metadata, known chromatic numbers, or upstream solutions"

    def _evaluate_proposed(
        self,
        iteration: int,
        proposed: list[dict[str, Any]],
        examples: list[GraphInstance],
    ) -> list[CandidateResult]:
        runner = GraphColouringSourceRunner(
            instances=examples,
            out_dir=self.run_dir,
            target_algorithm=self.config.target_algorithm,
            time_budget_s=self.config.time_budget_s,
            compile_timeout_s=self.config.compile_timeout_s,
            max_eval_workers=self.config.max_eval_workers,
            dry_run=self.config.dry_run,
            force=self.config.force,
            default_source_path=self.config.graph_colouring_source_path,
        )
        results: list[CandidateResult] = []
        for raw in proposed:
            if not isinstance(raw, dict):
                continue
            candidate = dict(raw)
            agent_name = str(
                candidate.get("agent_name")
                or candidate.get("scaffold_name")
                or DEFAULT_GRAPH_COLOURING_NAME
            )
            candidate.setdefault("agent_name", agent_name)
            candidate.setdefault("source_family", DEFAULT_GRAPH_COLOURING_NAME)
            candidate.setdefault("target_algorithm", self.config.target_algorithm)
            self._normalize_candidate_source_project_path(candidate)

            violations = self._candidate_code_policy_violations(candidate)
            if violations:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_policy_rejected",
                        "candidate": candidate,
                        "violations": violations,
                    }
                )
                continue

            candidate_name = str(candidate.get("name") or agent_name)
            candidate_id = f"iter{iteration:03d}_{candidate_name}"
            try:
                result = runner.evaluate_candidate(
                    candidate=candidate,
                    candidate_id=candidate_id,
                    agent_name=agent_name,
                )
            except Exception as exc:  # noqa: BLE001 - log and continue
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_eval_failed",
                        "candidate": candidate,
                        "candidate_id": candidate_id,
                        "error": str(exc),
                    }
                )
                continue
            results.append(result)
            self._append_summary(iteration=iteration, candidate=result, proposal=candidate)
        return results

    def _normalize_candidate_source_project_path(self, candidate: dict[str, Any]) -> None:
        """Keep proposer-edited graph-colouring snapshots ahead of the default."""

        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        if candidate.get("source_project_path"):
            return
        for key in (
            "source_project_path",
            "graph_colouring_source_path",
            "upstream_source_path",
        ):
            if extra.get(key):
                candidate["source_project_path"] = str(extra[key])
                return
        candidate["source_project_path"] = str(self.config.graph_colouring_source_path)

    def _copy_upstream_source_context(self, source_family: str, dest_dir: Path) -> None:
        super()._copy_upstream_source_context(source_family, dest_dir)
        if source_family != DEFAULT_GRAPH_COLOURING_NAME:
            return
        source = self.config.graph_colouring_source_path
        if not source.exists() or not source.is_dir():
            return
        # The upstream repo is 305MB on disk because of data/network-repo. The
        # proposer never needs that, the generated artifacts, or the legacy
        # fetchers, so prune them on copy and keep the editable surface tight.
        upstream_dest = dest_dir / "upstream_source" / "graph-colouring"
        self._copy_tree_if_exists(
            source,
            upstream_dest,
            ignore_names=(
                "build",
                "output",
                "results",
                "snapshots-colouring",
                "plots",
                "reports",
                "legacy",
                "bonus",
                "network-repo",
            ),
        )
        # Inject the `evolved` algorithm slot so the proposer sees and edits
        # the same scaffold the eval harness will compile against. Idempotent:
        # if the snapshot tree already has the overlay (e.g. a parent_iter
        # candidate was replayed) the call is a no-op and proposer edits to
        # evolved.cpp are preserved.
        if upstream_dest.exists():
            apply_graph_colouring_overlay(upstream_dest)

    def _candidate_policy_scan_paths(self, candidate: dict[str, Any]) -> list[Path]:
        out = super()._candidate_policy_scan_paths(candidate)
        source_path = self._candidate_graph_source_path(candidate)
        if source_path is not None and source_path.exists():
            for ext in ("*.cpp", "*.h", "*.hpp"):
                out.extend(sorted(source_path.rglob(ext)))
        return sorted(set(out))

    def _candidate_code_policy_violations(self, candidate: Any) -> list[dict[str, str]]:
        violations = super()._candidate_code_policy_violations(candidate)
        if not isinstance(candidate, dict):
            return violations
        # Block obvious cheats. The CSV pathname strings from the upstream's
        # legacy lookup_known_optimal_from_metadata are NOT a violation in
        # themselves — that lookup is dead code under our harness layout —
        # so we only flag a candidate that explicitly embeds a hand-written
        # table of chromatic numbers under the `known_optimal_table`
        # identifier, which is the canonical way the upstream's metadata
        # CSV is shaped and the most common shape for a hardcoded lookup.
        forbidden = {
            "known_optimal_table": (
                "candidate code must not embed a known-optimal lookup table"
            ),
        }
        for path in self._candidate_policy_scan_paths(candidate):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            for marker, reason in forbidden.items():
                if marker.lower() in text:
                    violations.append(
                        {
                            "path": str(path),
                            "marker": marker,
                            "reason": reason,
                        }
                    )
        return violations

    def _candidate_graph_source_path(self, candidate: dict[str, Any]) -> Path | None:
        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        for key in (
            "source_project_path",
            "graph_colouring_source_path",
            "upstream_source_path",
        ):
            value = candidate.get(key) or extra.get(key)
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            return path
        return None

    def _build_source_snapshot_workspace(
        self,
        *,
        iteration: int,
        source_family: str,
        call_dir: Path,
        target_system: str | None = None,
        snapshot_root: Path | None = None,
        generated_dir: Path | None = None,
        base_iter: int | None = None,
    ) -> Path:
        snapshot_root = super()._build_source_snapshot_workspace(
            iteration=iteration,
            source_family=source_family,
            call_dir=call_dir,
            target_system=target_system,
            snapshot_root=snapshot_root,
            generated_dir=generated_dir,
            base_iter=base_iter,
        )
        candidate_dir = snapshot_root / "candidate"
        upstream = candidate_dir / "upstream_source" / "graph-colouring"
        if base_iter is not None:
            parent_upstream = (
                self._iteration_dir(base_iter)
                / "source_snapshot"
                / "candidate"
                / "upstream_source"
                / "graph-colouring"
            )
            if parent_upstream.exists():
                if upstream.exists():
                    shutil.rmtree(upstream)
                shutil.copytree(
                    parent_upstream,
                    upstream,
                    ignore=shutil.ignore_patterns(
                        "build", "__pycache__", "*.pyc"
                    ),
                )
        readme = candidate_dir / "SNAPSHOT.md"
        readme.write_text(
            "\n".join(
                [
                    "# Graph-colouring Source Snapshot Candidate",
                    "",
                    f"Iteration: {iteration}",
                    f"Target system: {target_system or source_family}",
                    "",
                    "This directory is a writable candidate-specific source snapshot.",
                    "Edit `upstream_source/graph-colouring/src/algorithms/` and",
                    "`upstream_source/graph-colouring/src/benchmark_runner.cpp` to",
                    "improve the heuristic; do NOT edit `src/io/`, the Makefile, or",
                    "anything that would change the CSV output schema.",
                    "",
                    "Two-axis evaluation:",
                    "- colors_used (PRIMARY, lower is better — fewer colours)",
                    "- runtime_ms (TIEBREAKER, lower is better — only matters when",
                    "  colors_used ties)",
                    "",
                    "Write `pending_eval.json` with exactly one candidate. Point",
                    "`source_project_path` at",
                    "`source_snapshot/candidate/upstream_source/graph-colouring`",
                    "or the edited absolute path visible in the workspace.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        manifest_path = snapshot_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["graph_colouring_source"] = str(upstream)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        call_manifest_path = call_dir / "source_snapshot_manifest.json"
        if call_manifest_path.exists():
            call_manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return snapshot_root
