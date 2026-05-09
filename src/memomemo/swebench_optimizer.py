"""SWE-bench optimization entry point for source-backed coding agents."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memomemo.benchmark_workspaces import BenchmarkWorkspaceSpec, SWEBENCH_WORKSPACE_SPEC
from memomemo.optimizer import LocomoOptimizer, OptimizerConfig
from memomemo.schemas import CandidateResult
from memomemo.swebench import (
    DEFAULT_MINI_SWE_AGENT_NAME,
    DEFAULT_MINI_SWE_AGENT_SOURCE_PATH,
    MiniSweAgentSourceRunner,
    SwebenchInstance,
    load_swebench_instances,
    run_swebench_frontier,
)


@dataclass(frozen=True)
class SwebenchOptimizerConfig(OptimizerConfig):
    """Configuration for source-backed mini-SWE-agent optimization."""

    data_path: Path | None = None
    mini_swe_agent_source_path: Path = DEFAULT_MINI_SWE_AGENT_SOURCE_PATH
    mini_swe_agent_command: str = ""
    mini_swe_agent_eval_command: str = ""
    force: bool = False
    scaffolds: tuple[str, ...] = (DEFAULT_MINI_SWE_AGENT_NAME,)
    progressive_target_system: str = DEFAULT_MINI_SWE_AGENT_NAME


class SwebenchOptimizer(LocomoOptimizer):
    """Proposer loop for SWE-bench-style coding-agent candidates."""

    workspace_spec: BenchmarkWorkspaceSpec = SWEBENCH_WORKSPACE_SPEC
    config: SwebenchOptimizerConfig

    def __init__(self, config: SwebenchOptimizerConfig) -> None:
        super().__init__(config)

    def _load_examples(self) -> list[SwebenchInstance]:
        return load_swebench_instances(
            self.config.data_path,
            split=self.config.split,
            limit=self.config.limit,
        )

    def _run_seed_frontier(self) -> dict[str, Any]:
        return run_swebench_frontier(
            out_dir=self.run_dir,
            data_path=self.config.data_path,
            split=self.config.split,
            limit=self.config.limit,
            source_project_path=self.config.mini_swe_agent_source_path,
            command=self.config.mini_swe_agent_command,
            eval_command=self.config.mini_swe_agent_eval_command,
            timeout_s=self.config.eval_timeout_s,
            max_eval_workers=self.config.max_eval_workers,
            dry_run=self.config.dry_run,
            force=self.config.force,
            pareto_quality_threshold=self.config.pareto_quality_threshold,
        )

    def _benchmark_prompt_name(self) -> str:
        return "SWE-bench coding-agent issue resolution"

    def _raw_data_policy_name(self) -> str:
        return "SWE-bench gold patches, test patches, and evaluation results"

    def _evaluate_proposed(
        self,
        iteration: int,
        proposed: list[dict[str, Any]],
        examples: list[SwebenchInstance],
    ) -> list[CandidateResult]:
        runner = MiniSweAgentSourceRunner(
            instances=examples,
            out_dir=self.run_dir,
            timeout_s=self.config.eval_timeout_s,
            max_eval_workers=self.config.max_eval_workers,
            dry_run=self.config.dry_run,
            force=self.config.force,
        )
        results: list[CandidateResult] = []
        for raw in proposed:
            if not isinstance(raw, dict):
                continue
            candidate = dict(raw)
            agent_name = str(
                candidate.get("agent_name")
                or candidate.get("scaffold_name")
                or DEFAULT_MINI_SWE_AGENT_NAME
            )
            candidate.setdefault("agent_name", agent_name)
            candidate.setdefault("source_family", DEFAULT_MINI_SWE_AGENT_NAME)
            self._normalize_candidate_source_project_path(candidate)
            candidate.setdefault("command", self.config.mini_swe_agent_command)
            candidate.setdefault("eval_command", self.config.mini_swe_agent_eval_command)

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
        """Keep proposer-edited mini-SWE-agent snapshots ahead of the default source."""

        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        if candidate.get("source_project_path"):
            return
        for key in ("source_project_path", "upstream_source_path", "mini_swe_agent_source_path"):
            if extra.get(key):
                candidate["source_project_path"] = str(extra[key])
                return
        candidate["source_project_path"] = str(self.config.mini_swe_agent_source_path)

    def _copy_upstream_source_context(self, source_family: str, dest_dir: Path) -> None:
        super()._copy_upstream_source_context(source_family, dest_dir)
        if source_family != DEFAULT_MINI_SWE_AGENT_NAME:
            return
        source = self.config.mini_swe_agent_source_path
        if not source.exists() or not source.is_dir():
            return
        self._copy_tree_if_exists(
            source,
            dest_dir / "upstream_source" / "mini-swe-agent",
        )

    def _candidate_policy_scan_paths(self, candidate: dict[str, Any]) -> list[Path]:
        out = super()._candidate_policy_scan_paths(candidate)
        source_path = self._candidate_mini_source_path(candidate)
        if source_path is not None and source_path.exists():
            out.extend(sorted(source_path.rglob("*.py")))
        return sorted(set(out))

    def _candidate_code_policy_violations(self, candidate: Any) -> list[dict[str, str]]:
        violations = super()._candidate_code_policy_violations(candidate)
        if not isinstance(candidate, dict):
            return violations
        forbidden = {
            "test_patch": "runtime code must not read SWE-bench gold test patches",
            "gold_patch": "runtime code must not read SWE-bench gold patches",
            "swebench.harness": "candidate code must not call or modify the SWE-bench scorer",
            "candidate_results": "runtime code must not read previous candidate results",
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

    def _candidate_mini_source_path(self, candidate: dict[str, Any]) -> Path | None:
        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        for key in ("source_project_path", "upstream_source_path", "mini_swe_agent_source_path"):
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
        upstream = candidate_dir / "upstream_source" / "mini-swe-agent"
        # CuraII: when a parent base is supplied, replace the freshly
        # baseline-seeded mini-swe-agent source with the parent iteration's
        # archived candidate source so the proposer edits on top of a
        # previously evaluated candidate rather than restarting from baseline.
        if base_iter is not None:
            parent_upstream = (
                self._iteration_dir(base_iter)
                / "source_snapshot"
                / "candidate"
                / "upstream_source"
                / "mini-swe-agent"
            )
            if parent_upstream.exists():
                if upstream.exists():
                    shutil.rmtree(upstream)
                shutil.copytree(
                    parent_upstream,
                    upstream,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
        readme = candidate_dir / "SNAPSHOT.md"
        readme.write_text(
            "\n".join(
                [
                    "# mini-SWE-agent Source Snapshot Candidate",
                    "",
                    f"Iteration: {iteration}",
                    f"Target system: {target_system or source_family}",
                    "",
                    "This directory is a writable candidate-specific source snapshot.",
                    "Edit `upstream_source/mini-swe-agent` to optimize the coding agent.",
                    "Do not edit evaluator/scorer files or read gold patches/test patches.",
                    "",
                    "Write `pending_eval.json` with exactly one candidate. Point",
                    "`source_project_path` at `source_snapshot/candidate/upstream_source/mini-swe-agent`",
                    "or the edited absolute path visible in the workspace.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        manifest_path = snapshot_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["mini_swe_agent_source"] = str(upstream)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        call_manifest_path = call_dir / "source_snapshot_manifest.json"
        if call_manifest_path.exists():
            call_manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return snapshot_root
