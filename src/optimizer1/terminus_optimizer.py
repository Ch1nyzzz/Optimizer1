"""Terminal-Bench 2.0 optimization entry point for source-backed agent scaffolds.

This mirrors ``swebench_optimizer.py``: the proposer evolves an agent class
(seeded from Terminus-KIRA, ``baseline_kira.py``) inside a writable snapshot
of the vendored ``terminal_bench_2`` reference project, and the candidate is
scored by running Harbor on Terminal-Bench 2.0. Everything that is not the
scaffold itself — the outer loop, all selection policies, the per-benchmark
proposer skill, the trace MCP tools — is inherited from ``LocomoOptimizer``
unchanged, so Optimal1's deltas (pareto base resampling + trace harness)
layer on exactly as they do for SWE-bench mini.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optimizer1.benchmark_workspaces import BenchmarkWorkspaceSpec, TERMINUS_WORKSPACE_SPEC
from optimizer1.optimizer import LocomoOptimizer, OptimizerConfig
from optimizer1.schemas import CandidateResult
from optimizer1.terminus import (
    DEFAULT_TERMINUS_AGENT_NAME,
    DEFAULT_TERMINUS_AGENT_TIMEOUT_MULTIPLIER,
    DEFAULT_TERMINUS_CONCURRENCY,
    DEFAULT_TERMINUS_DATASET,
    DEFAULT_TERMINUS_ENV,
    DEFAULT_TERMINUS_HARBOR_COMMAND,
    DEFAULT_TERMINUS_JOB_TIMEOUT_S,
    DEFAULT_TERMINUS_REASONING_EFFORT,
    DEFAULT_TERMINUS_SECONDARY_BASELINE_IMPORT_PATH,
    DEFAULT_TERMINUS_SECONDARY_BASELINE_NAME,
    DEFAULT_TERMINUS_SEED_AGENT_IMPORT_PATH,
    DEFAULT_TERMINUS_SOLVER_API_KEY_ENV,
    DEFAULT_TERMINUS_SOLVER_BASE_URL,
    DEFAULT_TERMINUS_SOLVER_MODEL,
    DEFAULT_TERMINUS_SOURCE_FAMILY,
    DEFAULT_TERMINUS_SOURCE_PATH,
    DEFAULT_TERMINUS_TRIALS,
    TerminusHarborRunner,
    TerminusInstance,
    load_terminus_instances,
    run_terminus_frontier,
)


@dataclass(frozen=True)
class TerminusOptimizerConfig(OptimizerConfig):
    """Configuration for source-backed Terminal-Bench 2.0 agent-scaffold evolution."""

    tasks_path: Path | None = None
    test_tasks: tuple[str, ...] = ()
    terminus_source_path: Path = DEFAULT_TERMINUS_SOURCE_PATH
    dataset: str = DEFAULT_TERMINUS_DATASET
    harbor_environment: str = DEFAULT_TERMINUS_ENV
    harbor_command: str = DEFAULT_TERMINUS_HARBOR_COMMAND
    solver_model: str = DEFAULT_TERMINUS_SOLVER_MODEL
    solver_base_url: str = DEFAULT_TERMINUS_SOLVER_BASE_URL
    solver_api_key_env: str = DEFAULT_TERMINUS_SOLVER_API_KEY_ENV
    reasoning_effort: str = DEFAULT_TERMINUS_REASONING_EFFORT
    rollout_trials: int = DEFAULT_TERMINUS_TRIALS
    rollout_concurrency: int = DEFAULT_TERMINUS_CONCURRENCY
    agent_timeout_multiplier: float = DEFAULT_TERMINUS_AGENT_TIMEOUT_MULTIPLIER
    seed_agent_import_path: str = DEFAULT_TERMINUS_SEED_AGENT_IMPORT_PATH
    include_secondary_baseline: bool = True
    secondary_baseline_name: str = DEFAULT_TERMINUS_SECONDARY_BASELINE_NAME
    secondary_baseline_import_path: str = DEFAULT_TERMINUS_SECONDARY_BASELINE_IMPORT_PATH
    job_timeout_s: int = DEFAULT_TERMINUS_JOB_TIMEOUT_S
    force: bool = False
    scaffolds: tuple[str, ...] = (DEFAULT_TERMINUS_AGENT_NAME,)
    progressive_target_system: str = DEFAULT_TERMINUS_AGENT_NAME


class TerminusOptimizer(LocomoOptimizer):
    """Proposer loop for Terminal-Bench 2.0 agent-scaffold candidates."""

    workspace_spec: BenchmarkWorkspaceSpec = TERMINUS_WORKSPACE_SPEC
    config: TerminusOptimizerConfig

    def __init__(self, config: TerminusOptimizerConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------- data + seed

    def _load_examples_for_split(self, split: str, limit: int = 0) -> list[TerminusInstance]:
        return load_terminus_instances(
            self.config.tasks_path,
            split=split,
            limit=limit,
            test_tasks=tuple(self.config.test_tasks) or None,
        )

    def _run_seed_frontier(self) -> dict[str, Any]:
        return run_terminus_frontier(
            out_dir=self.run_dir,
            tasks_path=self.config.tasks_path,
            split=self.config.split,
            limit=self.config.limit,
            source_path=self.config.terminus_source_path,
            dataset=self.config.dataset,
            env=self.config.harbor_environment,
            harbor_command=self.config.harbor_command,
            solver_model=self.config.solver_model,
            solver_base_url=self.config.solver_base_url,
            solver_api_key_env=self.config.solver_api_key_env,
            reasoning_effort=self.config.reasoning_effort,
            trials=self.config.rollout_trials,
            concurrency=self.config.rollout_concurrency,
            agent_timeout_multiplier=self.config.agent_timeout_multiplier,
            seed_agent_import_path=self.config.seed_agent_import_path,
            include_secondary_baseline=self.config.include_secondary_baseline,
            secondary_baseline_name=self.config.secondary_baseline_name,
            secondary_baseline_import_path=self.config.secondary_baseline_import_path,
            test_tasks=tuple(self.config.test_tasks) or None,
            timeout_s=self.config.job_timeout_s,
            dry_run=self.config.dry_run,
            force=self.config.force,
            pareto_quality_threshold=self.config.pareto_quality_threshold,
        )

    def _benchmark_prompt_name(self) -> str:
        return "Terminal-Bench 2.0 long-horizon terminal task resolution"

    def _raw_data_policy_name(self) -> str:
        return "Terminal-Bench task verifier scripts, solution scripts, and reward files"

    # --------------------------------------------------------------- evaluation

    def _runner(self, examples: list[TerminusInstance]) -> TerminusHarborRunner:
        return TerminusHarborRunner(
            instances=examples,
            out_dir=self.run_dir,
            source_path=self.config.terminus_source_path,
            dataset=self.config.dataset,
            env=self.config.harbor_environment,
            harbor_command=self.config.harbor_command,
            solver_model=self.config.solver_model,
            solver_base_url=self.config.solver_base_url,
            solver_api_key_env=self.config.solver_api_key_env,
            reasoning_effort=self.config.reasoning_effort,
            trials=self.config.rollout_trials,
            concurrency=self.config.rollout_concurrency,
            agent_timeout_multiplier=self.config.agent_timeout_multiplier,
            seed_agent_import_path=self.config.seed_agent_import_path,
            timeout_s=self.config.job_timeout_s,
            dry_run=self.config.dry_run,
            force=self.config.force,
        )

    def _run_test_frontier(self, candidates: list[CandidateResult]) -> dict[str, Any]:
        """Re-evaluate the train Pareto frontier on the held-out test split.

        Unlike the LoCoMo/memory-scaffold base implementation this runs the
        candidate's *agent scaffold* through Harbor on the test tasks, so it
        works for source-backed Terminus candidates.
        """

        from optimizer1.pareto import ParetoPoint, save_frontier

        full_frontier = self._quality_frontier(candidates)
        candidate_limit = max(0, int(self.config.test_frontier_candidate_limit or 0))
        frontier = full_frontier[:candidate_limit] if candidate_limit else full_frontier
        test_dir = self.run_dir / "test_frontier"
        test_dir.mkdir(parents=True, exist_ok=True)
        examples = self._load_examples_for_split(self.config.test_split, self.config.test_limit)
        runner = TerminusHarborRunner(
            instances=examples,
            out_dir=test_dir,
            source_path=self.config.terminus_source_path,
            dataset=self.config.dataset,
            env=self.config.harbor_environment,
            harbor_command=self.config.harbor_command,
            solver_model=self.config.solver_model,
            solver_base_url=self.config.solver_base_url,
            solver_api_key_env=self.config.solver_api_key_env,
            reasoning_effort=self.config.reasoning_effort,
            trials=self.config.rollout_trials,
            concurrency=self.config.rollout_concurrency,
            agent_timeout_multiplier=self.config.agent_timeout_multiplier,
            seed_agent_import_path=self.config.seed_agent_import_path,
            timeout_s=self.config.job_timeout_s,
            dry_run=self.config.dry_run,
            force=self.config.force,
        )

        rows: list[dict[str, Any]] = []
        test_results: list[CandidateResult] = []
        failures: list[dict[str, Any]] = []
        for candidate in frontier:
            candidate_dict = dict(candidate.config) if isinstance(candidate.config, dict) else {}
            candidate_dict.setdefault("name", candidate.scaffold_name)
            candidate_dict.setdefault("agent_name", candidate.scaffold_name)
            self._normalize_candidate_source_project_path(candidate_dict)
            test_candidate_id = f"test_{candidate.candidate_id}"
            try:
                result = runner.evaluate_candidate(
                    candidate=candidate_dict,
                    candidate_id=test_candidate_id,
                    agent_name=str(candidate_dict.get("agent_name") or candidate.scaffold_name),
                )
            except Exception as exc:  # noqa: BLE001 - keep testing the rest of the frontier
                failure = {
                    "original_candidate_id": candidate.candidate_id,
                    "test_candidate_id": test_candidate_id,
                    "error": str(exc),
                }
                failures.append(failure)
                rows.append({"original_candidate": candidate.to_dict(), "error": str(exc)})
                self._append_event({"event": "test_frontier_candidate_failed", **failure})
                continue
            test_results.append(result)
            rows.append(
                {
                    "original_candidate": candidate.to_dict(),
                    "test_candidate": result.to_dict(),
                }
            )

        results_path = test_dir / "test_results.json"
        results_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        test_frontier_path = test_dir / "test_pareto_frontier.json"
        save_frontier(
            test_frontier_path,
            [
                ParetoPoint(
                    candidate_id=item.candidate_id,
                    scaffold_name=item.scaffold_name,
                    passrate=item.passrate,
                    token_consuming=item.token_consuming,
                    avg_token_consuming=item.avg_token_consuming,
                    average_score=item.average_score,
                    result_path=item.result_path,
                    config=item.config,
                )
                for item in test_results
            ],
            quality_gap_threshold=self.config.pareto_quality_threshold,
        )
        summary = {
            "split": self.config.test_split,
            "limit": self.config.test_limit,
            "count": len(examples),
            "train_frontier_total_count": len(full_frontier),
            "candidate_limit": candidate_limit,
            "train_frontier_count": len(frontier),
            "evaluated_count": len(test_results),
            "failed_count": len(failures),
            "test_dir": str(test_dir),
            "test_results_path": str(results_path),
            "test_pareto_frontier_path": str(test_frontier_path),
            "failures": failures,
        }
        summary_path = self.run_dir / "test_frontier_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        return summary

    def _evaluate_proposed(
        self,
        iteration: int,
        proposed: list[dict[str, Any]],
        examples: list[TerminusInstance],
    ) -> list[CandidateResult]:
        runner = self._runner(examples)
        results: list[CandidateResult] = []
        for raw in proposed:
            if not isinstance(raw, dict):
                continue
            candidate = dict(raw)
            agent_name = str(
                candidate.get("agent_name")
                or candidate.get("scaffold_name")
                or DEFAULT_TERMINUS_AGENT_NAME
            )
            candidate.setdefault("agent_name", agent_name)
            candidate.setdefault("source_family", DEFAULT_TERMINUS_SOURCE_FAMILY)
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

    # ----------------------------------------------------- source-snapshot glue

    def _terminus_source_path(self) -> Path:
        path = Path(self.config.terminus_source_path).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _normalize_candidate_source_project_path(self, candidate: dict[str, Any]) -> None:
        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        if candidate.get("source_project_path"):
            return
        for key in ("source_project_path", "terminus_source_path", "upstream_source_path"):
            if extra.get(key):
                candidate["source_project_path"] = str(extra[key])
                return
        candidate["source_project_path"] = str(self._terminus_source_path())

    def _copy_upstream_source_context(self, source_family: str, dest_dir: Path) -> None:
        super()._copy_upstream_source_context(source_family, dest_dir)
        if source_family not in {DEFAULT_TERMINUS_SOURCE_FAMILY, DEFAULT_TERMINUS_AGENT_NAME}:
            return
        source = self._terminus_source_path()
        if not source.exists() or not source.is_dir():
            return
        self._copy_tree_if_exists(source, dest_dir / "upstream_source" / "terminal-bench-2")

    def _candidate_policy_scan_paths(self, candidate: dict[str, Any]) -> list[Path]:
        out = super()._candidate_policy_scan_paths(candidate)
        source_path = self._candidate_terminus_source_path(candidate)
        if source_path is not None and source_path.exists():
            agents_dir = source_path / "agents"
            scan_root = agents_dir if agents_dir.exists() else source_path
            out.extend(sorted(scan_root.rglob("*.py")))
        return sorted(set(out))

    def _candidate_code_policy_violations(self, candidate: Any) -> list[dict[str, str]]:
        violations = super()._candidate_code_policy_violations(candidate)
        if not isinstance(candidate, dict):
            return violations
        forbidden = {
            "verifier_result": "runtime code must not read the harbor verifier result",
            "solution.sh": "runtime code must not read or run task solution scripts",
            "tests/verifier": "runtime code must not read task verifier scripts",
            "reward.txt": "runtime code must not read reward files",
            "candidate_results": "runtime code must not read previous candidate results",
        }
        for path in self._candidate_policy_scan_paths(candidate):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            for marker, reason in forbidden.items():
                if marker.lower() in text:
                    violations.append({"path": str(path), "marker": marker, "reason": reason})
        return violations

    def _candidate_terminus_source_path(self, candidate: dict[str, Any]) -> Path | None:
        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        for key in ("source_project_path", "terminus_source_path", "upstream_source_path"):
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
        upstream = candidate_dir / "upstream_source" / "terminal-bench-2"
        if base_iter is not None:
            parent_upstream = (
                self._iteration_dir(base_iter)
                / "source_snapshot"
                / "candidate"
                / "upstream_source"
                / "terminal-bench-2"
            )
            if parent_upstream.exists():
                if upstream.exists():
                    shutil.rmtree(upstream)
                shutil.copytree(
                    parent_upstream,
                    upstream,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "jobs", "logs"),
                )
        readme = candidate_dir / "SNAPSHOT.md"
        readme.write_text(
            "\n".join(
                [
                    "# Terminal-Bench 2.0 Agent-Scaffold Snapshot Candidate",
                    "",
                    f"Iteration: {iteration}",
                    f"Target system: {target_system or source_family}",
                    "",
                    "This directory is a writable candidate-specific snapshot of the",
                    "vendored `terminal_bench_2` reference project.",
                    "",
                    "Create your new agent at `upstream_source/terminal-bench-2/agents/<name>.py`,",
                    "copying `agents/baseline_kira.py` (Terminus-KIRA) as the parent. The class",
                    "MUST be named `AgentHarness` and subclass",
                    "`harbor.agents.terminus_2.terminus_2.Terminus2` (see `.claude/skills/`).",
                    "Do NOT edit `meta_harness.py`, `claude_wrapper.py`, the existing agents,",
                    "task verifier/solution scripts, or read reward files.",
                    "",
                    "Write `pending_eval.json` with exactly one candidate. Set",
                    "`terminus_agent_import_path` to `agents.<name>:AgentHarness` and",
                    "`source_project_path` to",
                    "`source_snapshot/candidate/upstream_source/terminal-bench-2` (or the",
                    "edited absolute path visible in the workspace).",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        manifest_path = snapshot_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["terminus_source"] = str(upstream)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            call_manifest_path = call_dir / "source_snapshot_manifest.json"
            if call_manifest_path.exists():
                call_manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        return snapshot_root


__all__ = ["TerminusOptimizer", "TerminusOptimizerConfig"]
