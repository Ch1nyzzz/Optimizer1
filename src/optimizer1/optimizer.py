"""Claude Code proposer optimization loop for OptiHarness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optimizer1.baseline import load_baseline_candidates
from optimizer1.benchmark_workspaces import (
    LOCOMO_WORKSPACE_SPEC,
    BenchmarkWorkspaceSpec,
    copy_benchmark_project_source,
)
from optimizer1.claude_runner import (
    DEFAULT_DOCKER_ENV_VARS,
    ProposerSandboxConfig,
    _float_metric,
    _int_metric,
    run_code_agent_prompt,
)
from optimizer1.dynamic import load_candidate_scaffold
from optimizer1.evaluation import EvaluationRunner, run_initial_frontier
from optimizer1.evidence_store import EvidenceStore
from optimizer1.locomo import default_data_path, load_locomo_examples, prepare_locomo, select_split
from optimizer1.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from optimizer1.optimization_cells import get_target_cells
from optimizer1.pareto import ParetoPoint, pareto_frontier, save_frontier
from optimizer1.post_eval import write_diff_digest, write_post_eval_artifacts
from optimizer1.run_store import RunStore, diff_stats
from optimizer1.traces import TraceHarness, has_adapter
from optimizer1.proposer_prompt import build_progressive_proposer_prompt
from optimizer1.scaffolds import DEFAULT_EVOLUTION_SEED_SCAFFOLDS, DEFAULT_SCAFFOLD_TOP_KS
from optimizer1.scaffolds.base import ScaffoldConfig
from optimizer1.schemas import CandidateResult, LocomoExample


DEFAULT_PROPOSER_DOCKER_IMAGE = "docker-claude:latest"


def _pending_candidates(payload: Any) -> list[Any]:
    """Accept either {"candidates": [...]} or a top-level candidate list."""

    if isinstance(payload, dict):
        candidates = payload.get("candidates") or []
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = []
    return candidates if isinstance(candidates, list) else []


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the Claude Code proposer loop."""

    run_id: str
    out_dir: Path
    iterations: int = 20
    split: str = "train"
    limit: int = 0
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = "EMPTY"
    eval_timeout_s: int = 300
    proposer_agent: str = "claude"
    claude_model: str = "deepseek-v4-pro[1m]"
    claude_effort: str = ""
    claude_base_url: str = "https://api.deepseek.com/anthropic"
    claude_auth_token: str | None = None
    claude_native_auth: bool = False
    # Codex proposer settings (used only when proposer_agent == "codex").
    # Codex authenticates from $CODEX_HOME/auth.json (default ~/.codex),
    # not via env tokens, so there is no codex_auth_token field.
    codex_model: str = "gpt-5.5"
    codex_reasoning_effort: str = "high"
    codex_home: str = ""
    propose_timeout_s: int = 2400
    dry_run: bool = False
    max_context_chars: int = 6000
    max_eval_workers: int = 1
    skip_scaffold_eval: bool = False
    resume: bool = False
    # When a proposer invocation reports an Anthropic usage-limit rejection
    # (HTTP 429 / rejected rate_limit_event), sleep until the limit window
    # resets and retry the same invocation instead of burning the iteration.
    wait_on_rate_limit: bool = True
    rate_limit_buffer_s: int = 120
    rate_limit_default_wait_s: int = 1800
    rate_limit_max_wait_s: int = 6 * 3600
    rate_limit_max_retries: int = 8
    rate_limit_poll_log_s: int = 300
    baseline_dir: Path | None = None
    scaffolds: tuple[str, ...] = DEFAULT_EVOLUTION_SEED_SCAFFOLDS
    scaffold_extra: dict[str, dict[str, object]] | None = None
    selection_policy: str = "default"
    include_optimization_direction: bool = False
    force_budget: str = ""
    progressive_target_system: str = "memgpt"
    progressive_initial_low_iterations: int = 5
    progressive_low_best_count: int = 1
    progressive_medium_best_count: int = 3
    bandit_prior_weight: float = 0.4
    bandit_prior_alpha: float = 2.0
    bandit_exploration_c: float = 0.15
    bandit_cost_lambda: float = 0.05
    bandit_line_scale: int = 500
    bandit_min_core_files: bool = True
    bandit_stagnation_threshold: int = 4
    bandit_reward_window: int = 8
    bandit_reward_sigma_floor: float = 0.02
    bandit_reward_clip: float = 2.0
    bandit_failed_iter_penalty: float = 0.5
    pareto_quality_threshold: float = 0.125
    proposer_sandbox: str = "none"
    proposer_docker_image: str = ""
    proposer_docker_workspace: str = "/workspace"
    proposer_docker_env: tuple[str, ...] = ()
    proposer_docker_mount: tuple[str, ...] = ()
    proposer_docker_user: str = ""
    proposer_docker_home: str = ""
    test_frontier: bool = False
    test_split: str = "test"
    test_limit: int = 0
    test_frontier_candidate_limit: int = 0
    trace_baseline_path: Path | None = None
    proposer_show_trace_harness_section: bool = True
    # When False, the cumulative cross-session summary (the workspace
    # ``summaries/`` directory of structured logs, and the corresponding
    # prompt section) is withheld from the proposer -- the no-summary probe.
    summaries_in_workspace: bool = True
    # Organized mode uses generated state.md + RunStore tools as the
    # proposer's historical interface. Summaries remain generated on the
    # run side for compatibility, but are not copied into the workspace.
    organized: bool = False
    organized_include_summaries: bool = False


class LocomoOptimizer:
    """Meta-harness-style proposer loop for LOCOMO memory scaffolds."""

    workspace_spec: BenchmarkWorkspaceSpec = LOCOMO_WORKSPACE_SPEC

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parents[2]
        self.run_dir = config.out_dir
        self.pending_eval_path = self.run_dir / "pending_eval.json"
        self.frontier_path = self.run_dir / "best_candidates.json"
        self.summary_path = self.run_dir / "evolution_summary.jsonl"
        self.generated_dir = self.run_dir / "generated"
        self.progressive_state_path = self.run_dir / "progressive_state.json"
        self.bandit_state_path = self.run_dir / "bandit_state.json"
        self.candidate_score_table_path = self.run_dir / "candidate_score_table.json"
        self.retrieval_diagnostics_summary_path = (
            self.run_dir / "retrieval_diagnostics_summary.json"
        )
        self.iteration_index_path = self.run_dir / "iteration_index.json"
        self.diff_summary_path = self.run_dir / "diff_summary.jsonl"
        self._validate_proposer_sandbox_policy()
        self._validate_proposer_agent()
        self.run_store = RunStore(
            self.run_dir,
            benchmark=self.workspace_spec.benchmark,
        )
        self.evidence_store = EvidenceStore(self.run_dir)
        self.trace_harness: TraceHarness = self._build_trace_harness()

    def _build_trace_harness(self) -> TraceHarness:
        benchmark = self.workspace_spec.benchmark
        if not has_adapter(benchmark):
            raise ValueError(
                f"No trace adapter is registered for benchmark "
                f"{benchmark!r}. Register one in optimizer1.traces.adapters."
            )
        return TraceHarness(
            run_dir=self.run_dir,
            benchmark=benchmark,
            baseline_path=self.config.trace_baseline_path,
        )

    def _validate_proposer_agent(self) -> None:
        agent = self.config.proposer_agent.strip().lower()
        if agent not in {"claude", "codex"}:
            raise ValueError(
                "proposer_agent must be 'claude' or 'codex'; got "
                f"proposer_agent={self.config.proposer_agent!r}"
            )

    def _validate_proposer_sandbox_policy(self) -> None:
        policy = self.config.selection_policy.strip().lower()
        if policy not in {
            "progressive",
            "bandit",
            "random",
            "recent",
            "best",
            "curai",
            "curaii",
            "pareto",
        }:
            return
        sandbox = self.config.proposer_sandbox.strip().lower()
        if sandbox != "docker":
            raise ValueError(
                f"{policy} selection policy requires --proposer-sandbox docker"
            )
        if not self._effective_proposer_docker_image():
            raise ValueError(
                f"{policy} selection policy requires --proposer-docker-image"
            )

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(self.generated_dir)
        examples = self._load_examples()

        candidates: list[CandidateResult] = []
        if self.config.baseline_dir is not None:
            # Only filter on top_k for scaffolds that define one in
            # DEFAULT_SCAFFOLD_TOP_KS (memgpt_source, bm25, ...). Source-only
            # scaffolds like mini_swe_agent_source carry no top_k in their
            # candidate config, so applying the filter would reject every
            # candidate and break --baseline-dir on the SWE-bench / Terminus
            # tasks. Passing None when the dict is empty also lets a user
            # reuse an arbitrary baseline that was built without top_k
            # variants.
            top_k_filter = {
                scaffold: DEFAULT_SCAFFOLD_TOP_KS[scaffold]
                for scaffold in self.config.scaffolds
                if scaffold in DEFAULT_SCAFFOLD_TOP_KS
            }
            baseline_candidates = load_baseline_candidates(
                self.config.baseline_dir,
                split=self.config.split,
                scaffolds=self.config.scaffolds,
                top_k_by_scaffold=top_k_filter or None,
            )
            if not baseline_candidates:
                raise ValueError(
                    f"No baseline candidates found for split '{self.config.split}' "
                    f"under {self.config.baseline_dir}"
                )
            for item in baseline_candidates:
                candidate = CandidateResult.from_dict(item)
                if int(candidate.count) != len(examples):
                    raise ValueError(
                        "Baseline candidate count does not match current evaluation set: "
                        f"{candidate.candidate_id} has count={candidate.count}, "
                        f"but split={self.config.split!r} limit={self.config.limit} "
                        f"selects {len(examples)} examples. Recompute the baseline with "
                        "the same split/limit before using it as --baseline-dir."
                    )
                candidates.append(candidate)
                self._append_summary(iteration=0, candidate=candidate)
        elif not (self.config.skip_scaffold_eval or self.config.resume):
            scaffold_summary = self._run_seed_frontier()
            for item in scaffold_summary.get("candidates", []):
                candidate = CandidateResult.from_dict(item)
                candidates.append(candidate)
                self._append_summary(iteration=0, candidate=candidate)
        else:
            if self.config.resume and not (self.run_dir / "candidate_results").exists():
                raise ValueError(
                    f"--resume: no candidate_results/ under {self.run_dir}; "
                    "nothing to resume from."
                )
            candidates.extend(self._load_existing_candidates())
            if self.config.resume and not candidates:
                raise ValueError(
                    f"--resume: candidate_results/ under {self.run_dir} is empty "
                    "or unreadable; nothing to resume from."
                )

        # Reusing scaffold/baseline rows is mutually exclusive with resume;
        # resume always reloads from candidate_results and never re-records
        # iteration 0, exactly like --skip-scaffold-eval.
        skip_iter0_recording = self.config.skip_scaffold_eval or self.config.resume

        start_iteration = 1
        if self.config.resume:
            start_iteration = self._resume_start_iteration(candidates)
            # Drop any stale per-iteration artifacts for the iterations we
            # are about to (re)run — these may exist as crashed/partial dirs
            # plus dangling evolution_summary / diff_summary rows.
            self._clean_stale_iteration_artifacts(start_iteration)
            if self.config.selection_policy in {
                "progressive",
                "curai",
                "curaii",
                "pareto",
            }:
                # Crashed iterations still bumped stagnation_count etc.;
                # rebuild progressive_state.json from the surviving rows so
                # the budget heuristics see the true streak.
                self._rederive_progressive_state(
                    candidates, start_iteration=start_iteration
                )

        if candidates and not skip_iter0_recording:
            best_ids = self._quality_frontier_ids(candidates)
            write_post_eval_artifacts(
                run_dir=self.run_dir,
                call_dir=None,
                iteration=0,
                candidates=candidates,
                frontier_ids=best_ids,
            )
            self.trace_harness.record_iteration(
                iteration=0,
                candidates=candidates,
                selection_policy=self.config.selection_policy,
            )
            self.run_store.record_eval(0, candidates)
            self.run_store.commit_iteration(0)
            self._refresh_evidence_store(0)

        self._save_best_candidates(candidates)
        self._refresh_run_indexes(candidates)

        for iteration in range(start_iteration, self.config.iterations + 1):
            previous_best_passrate = self._best_passrate(candidates)
            previous_frontier_ids = self._quality_frontier_ids(candidates)
            previous_best_quality = self._best_quality_value(candidates)
            bandit_policy: dict[str, Any] | None = None
            forced_budget = self.config.force_budget or None
            if self.config.selection_policy == "bandit":
                bandit_policy = self._bandit_policy_for_workspace(
                    iteration=iteration,
                    candidates=candidates,
                    force_budget=forced_budget,
                )
                budget = str(bandit_policy.get("budget") or "low")
            elif self.config.selection_policy in {"progressive", "curai", "curaii"}:
                budget = forced_budget or self._progressive_budget_for_iteration(iteration)
            elif self.config.selection_policy in {"random", "recent", "best"}:
                budget = forced_budget or "medium"
            else:
                # default and pareto both pin to fixed-high context every
                # iter; the only difference is that pareto resamples the
                # patch base from the current Pareto frontier instead of
                # always re-baselining from the clean snapshot.
                budget = forced_budget or "high"
            evaluated = self._run_progressive_proposer_iteration(
                iteration,
                candidates,
                examples,
                budget=budget,
                adaptive=self.config.selection_policy
                in {
                    "progressive",
                    "bandit",
                    "random",
                    "recent",
                    "best",
                    "curai",
                    "curaii",
                    "pareto",
                },
                selection_policy=self.config.selection_policy,
                bandit_policy=bandit_policy,
            )
            candidates.extend(evaluated)
            self._save_best_candidates(candidates)
            self._refresh_run_indexes(candidates)
            best_ids = self._quality_frontier_ids(candidates)
            write_post_eval_artifacts(
                run_dir=self.run_dir,
                call_dir=None,
                iteration=iteration,
                candidates=evaluated,
                frontier_ids=best_ids,
            )
            self.trace_harness.record_iteration(
                iteration=iteration,
                candidates=evaluated,
            )
            self.run_store.record_eval(iteration, evaluated)
            self._refresh_evidence_store(iteration)
            if self.config.selection_policy in {"progressive", "curai", "curaii", "pareto"}:
                # pareto's advance signal is "passrate strictly beat the
                # historical best"; curai / curaii / progressive treat
                # joining the top-K frontier as advance. Both routes
                # update progressive_state.json so the budget heuristics
                # read the stagnation_count uniformly.
                advance_signal_ids = (
                    None
                    if self.config.selection_policy == "pareto"
                    else previous_frontier_ids
                )
                advanced = self._update_progressive_state(
                    iteration=iteration,
                    budget=budget,
                    previous_best_passrate=previous_best_passrate,
                    previous_frontier_ids=advance_signal_ids,
                    candidates=candidates,
                    evaluated=evaluated,
                )
                # Mirror into iteration_meta so MCP queries see a single
                # source of truth for "did this iter advance".
                if self.trace_harness.indexer.db_path.exists():
                    self.trace_harness.indexer.upsert_iteration_meta(
                        iteration=iteration,
                        advanced_frontier=advanced,
                    )
            if self.config.selection_policy == "bandit":
                self._update_bandit_state(
                    iteration=iteration,
                    previous_best_passrate=previous_best_passrate,
                    previous_best_quality=previous_best_quality,
                    evaluated=evaluated,
                    call_dir=self._iteration_dir(iteration),
                )

        test_frontier_summary = (
            self._run_test_frontier(candidates)
            if self.config.test_frontier
            else None
        )

        final_summary = {
            "run_id": self.config.run_id,
            "out_dir": str(self.run_dir),
            "iterations": self.config.iterations,
            "candidate_count": len(candidates),
            "best_candidates_path": str(self.frontier_path),
            "selection_policy": self.config.selection_policy,
            "proposer_metrics": self._aggregate_proposer_metrics(),
        }
        if test_frontier_summary is not None:
            final_summary["test_frontier"] = test_frontier_summary
        if self.config.selection_policy == "bandit":
            final_summary["bandit_state_path"] = str(self.bandit_state_path)
            final_summary["bandit_policy"] = self._load_bandit_state().get("last_policy", {})
        (self.run_dir / "optimizer_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return final_summary

    def _run_default_proposer_iteration(
        self,
        iteration: int,
        examples: list[LocomoExample],
        existing_candidates: list[CandidateResult] | None = None,
    ) -> list[CandidateResult]:
        return self._run_progressive_proposer_iteration(
            iteration,
            existing_candidates or [],
            examples,
            budget="high",
            adaptive=False,
        )

    def _run_progressive_proposer_iteration(
        self,
        iteration: int,
        existing_candidates: list[CandidateResult],
        examples: list[LocomoExample],
        *,
        budget: str,
        adaptive: bool,
        selection_policy: str | None = None,
        bandit_policy: dict[str, Any] | None = None,
    ) -> list[CandidateResult]:
        if self.pending_eval_path.exists():
            self.pending_eval_path.unlink()
        call_dir = self.run_dir / "proposer_calls" / f"iter_{iteration:03d}"
        retry_note = ""
        max_attempts = 2
        result: Any | None = None
        workspace_dir = call_dir / "workspace"
        workspace_generated_dir = workspace_dir / "generated"
        reference_iterations: tuple[int, ...] = ()
        policy_name = selection_policy or ("progressive" if adaptive else "default")
        curaii_base_iter: int | None = None
        curaii_base_passrate: float | None = None
        curaii_base_average_score: float | None = None
        curaii_refs_override: tuple[int, ...] | None = None
        if policy_name == "curaii":
            curaii_base_iter, curaii_refs_override = self._curaii_select_for_budget(
                existing_candidates,
                iteration=iteration,
                budget=budget,
                baseline_passrate=self._seed_passrate(existing_candidates),
            )
        elif policy_name == "pareto":
            curaii_base_iter = self._pareto_select_base(
                existing_candidates,
                iteration=iteration,
                baseline_passrate=self._seed_passrate(existing_candidates),
            )
            curaii_refs_override = None
        if curaii_base_iter is not None:
            base_candidate = next(
                (
                    item
                    for item in existing_candidates
                    if _candidate_iteration(item.candidate_id)
                    == curaii_base_iter
                ),
                None,
            )
            if base_candidate is not None:
                curaii_base_passrate = base_candidate.passrate
                curaii_base_average_score = base_candidate.average_score
        state_base_iter = curaii_base_iter
        if self.config.organized and state_base_iter is None:
            state_base_iter = self._state_snapshot_base_iteration(
                existing_candidates,
                iteration=iteration,
            )
        self.run_store.begin_iteration(
            iteration,
            as_of_iteration=max(0, iteration - 1),
            base_iteration=state_base_iter if self.config.organized else curaii_base_iter,
            status="running",
        )
        if self.config.organized:
            self._write_state_md(
                iteration=iteration,
                as_of_iteration=max(0, iteration - 1),
                base_iteration=state_base_iter,
            )
        for attempt in range(1, max_attempts + 1):
            if bandit_policy:
                refs_override: tuple[int, ...] | None = tuple(
                    int(item) for item in bandit_policy.get("reference_iterations", ())
                )
            else:
                refs_override = curaii_refs_override
            workspace_dir, reference_iterations = self._build_progressive_workspace(
                iteration=iteration,
                budget=budget,
                existing_candidates=existing_candidates,
                call_dir=call_dir,
                reference_iterations_override=refs_override,
                bandit_policy=bandit_policy,
                base_iter=curaii_base_iter,
            )
            workspace_generated_dir = workspace_dir / "generated"
            workspace_source_snapshot_dir = workspace_dir / "source_snapshot"
            workspace_pending_eval_path = workspace_dir / "pending_eval.json"
            workspace_traces_dir = workspace_dir / "traces"
            # The proposer receives its self-contained benchmark skill via
            # the system-prompt channel (Claude --append-system-prompt /
            # Codex AGENTS.md); the user message carries only the
            # per-iteration assignment.
            prompt = build_progressive_proposer_prompt(
                run_id=self.config.run_id,
                iteration=iteration,
                run_dir=workspace_dir,
                pending_eval_path=workspace_pending_eval_path,
                summaries_dir=workspace_dir / "summaries",
                include_summaries=self._summaries_in_workspace_enabled(),
                reference_iterations_dir=workspace_dir / "reference_iterations",
                generated_dir=workspace_generated_dir,
                source_snapshot_dir=workspace_source_snapshot_dir,
                budget=budget,
                reference_iterations=reference_iterations,
                target_system=self.config.progressive_target_system,
                optimization_directions=(
                    self._optimization_direction_lines(self.config.progressive_target_system)
                    if self.config.include_optimization_direction
                    else ()
                ),
                split=self.config.split,
                limit=self.config.limit,
                selection_policy=policy_name,
                bandit_policy=bandit_policy,
                benchmark_name=self._benchmark_prompt_name(),
                current_base_iter=curaii_base_iter,
                current_base_passrate=curaii_base_passrate,
                current_base_average_score=curaii_base_average_score,
                state_path=workspace_dir / "state.md" if self.config.organized else None,
                organized=self.config.organized,
                trace_harness_dir=(
                    workspace_traces_dir
                    if self.config.proposer_show_trace_harness_section
                    else None
                ),
            )
            if retry_note:
                prompt = f"{prompt}\n\n{retry_note}"
            result = self._run_proposer_agent(
                prompt,
                log_dir=call_dir / "agent" / f"attempt_{attempt:02d}",
                name="proposer",
                cwd=workspace_dir,
            )
            self._append_proposer_result_event(
                iteration=iteration,
                result=result,
                selection_policy=policy_name,
                extra={
                    "budget": budget,
                    "reference_iterations": list(reference_iterations),
                    "target_system": self.config.progressive_target_system,
                    "call_dir": str(call_dir),
                    "workspace_dir": str(workspace_dir),
                    "attempt": attempt,
                    "bandit_policy_score_snapshot": (
                        bandit_policy.get("policy_score_snapshot", {})
                        if bandit_policy
                        else {}
                    ),
                },
            )
            access_violations = self._proposer_access_violations(
                result,
                workspace_dir=workspace_dir,
            )
            if not access_violations:
                break
            if attempt < max_attempts:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "proposer_access_retry",
                        "selection_policy": policy_name,
                        "budget": budget,
                        "attempt": attempt,
                        "violations": access_violations,
                    }
                )
                retry_note = self._access_retry_note(
                    violations=access_violations,
                    workspace_dir=workspace_dir,
                )
                continue
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "proposer_access_rejected",
                    "selection_policy": policy_name,
                    "budget": budget,
                    "attempt": attempt,
                    "violations": access_violations,
                }
            )
            return []

        assert result is not None
        self._archive_workspace_outputs(
            workspace_dir=workspace_dir,
            call_dir=call_dir,
            result=result,
        )
        if (
            result.returncode == 0
            and not result.timed_out
            and not self.pending_eval_path.exists()
        ):
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "proposer_missing_pending_retry",
                    "selection_policy": policy_name,
                    "budget": budget,
                    "attempt": max_attempts,
                }
            )
            repair_prompt = (
                f"{prompt}\n\n"
                "## Required Repair\n\n"
                "The previous proposer attempt exited without writing "
                f"`{workspace_pending_eval_path}`. Continue in the same "
                "workspace, make a concrete candidate source change if needed, "
                "and write exactly one valid `pending_eval.json`. Do not run "
                "the full harness evaluation."
            )
            result = self._run_proposer_agent(
                repair_prompt,
                log_dir=call_dir / "agent" / "missing_pending_retry",
                name="proposer",
                cwd=workspace_dir,
            )
            self._append_proposer_result_event(
                iteration=iteration,
                result=result,
                selection_policy=policy_name,
                extra={
                    "budget": budget,
                    "reference_iterations": list(reference_iterations),
                    "target_system": self.config.progressive_target_system,
                    "call_dir": str(call_dir),
                    "workspace_dir": str(workspace_dir),
                    "attempt": "missing_pending_retry",
                    "bandit_policy_score_snapshot": (
                        bandit_policy.get("policy_score_snapshot", {})
                        if bandit_policy
                        else {}
                    ),
                },
            )
            access_violations = self._proposer_access_violations(
                result,
                workspace_dir=workspace_dir,
            )
            if access_violations:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "proposer_access_rejected",
                        "selection_policy": policy_name,
                        "budget": budget,
                        "attempt": "missing_pending_retry",
                        "violations": access_violations,
                    }
                )
                return []
            self._archive_workspace_outputs(
                workspace_dir=workspace_dir,
                call_dir=call_dir,
                result=result,
            )
        if result.returncode != 0 or result.timed_out or not self.pending_eval_path.exists():
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "proposer_failed",
                    "selection_policy": policy_name,
                    "budget": budget,
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "stderr": result.stderr[:1000],
                    "proposer_metrics": getattr(result, "metrics", {}),
                }
            )
            return []

        try:
            pending = json.loads(self.pending_eval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "proposer_invalid_pending_retry",
                    "selection_policy": policy_name,
                    "budget": budget,
                    "attempt": "json_parse_retry",
                    "error": str(exc),
                }
            )
            repair_prompt = (
                f"{prompt}\n\n"
                "## Required Repair\n\n"
                f"The previous proposer wrote invalid JSON to `{workspace_pending_eval_path}`: "
                f"{exc}. Rewrite that file as exactly one valid JSON object with a "
                "`candidates` array containing one candidate. Escape backslashes inside "
                "strings as `\\\\`, or remove them. Do not run the full harness evaluation."
            )
            result = self._run_proposer_agent(
                repair_prompt,
                log_dir=call_dir / "agent" / "invalid_pending_retry",
                name="proposer",
                cwd=workspace_dir,
            )
            self._append_proposer_result_event(
                iteration=iteration,
                result=result,
                selection_policy=policy_name,
                extra={
                    "budget": budget,
                    "reference_iterations": list(reference_iterations),
                    "target_system": self.config.progressive_target_system,
                    "call_dir": str(call_dir),
                    "workspace_dir": str(workspace_dir),
                    "attempt": "invalid_pending_retry",
                    "bandit_policy_score_snapshot": (
                        bandit_policy.get("policy_score_snapshot", {})
                        if bandit_policy
                        else {}
                    ),
                },
            )
            access_violations = self._proposer_access_violations(
                result,
                workspace_dir=workspace_dir,
            )
            if access_violations:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "proposer_access_rejected",
                        "selection_policy": policy_name,
                        "budget": budget,
                        "attempt": "invalid_pending_retry",
                        "violations": access_violations,
                    }
                )
                return []
            self._archive_workspace_outputs(
                workspace_dir=workspace_dir,
                call_dir=call_dir,
                result=result,
            )
            if result.returncode != 0 or result.timed_out or not self.pending_eval_path.exists():
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "proposer_failed",
                        "selection_policy": policy_name,
                        "budget": budget,
                        "returncode": result.returncode,
                        "timed_out": result.timed_out,
                        "stderr": result.stderr[:1000],
                        "proposer_metrics": getattr(result, "metrics", {}),
                    }
                )
                return []
            try:
                pending = json.loads(self.pending_eval_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as retry_exc:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "proposer_invalid_pending_rejected",
                        "selection_policy": policy_name,
                        "budget": budget,
                        "attempt": "invalid_pending_retry",
                        "error": str(retry_exc),
                    }
                )
                return []
        proposed = _pending_candidates(pending)
        if len(proposed) != 1:
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "candidate_count_adjusted",
                    "selection_policy": policy_name,
                    "budget": budget,
                    "requested_count": len(proposed),
                    "evaluated_count": min(len(proposed), 1),
                }
            )
            proposed = proposed[:1]
        for raw in proposed:
            if isinstance(raw, dict):
                self._normalize_workspace_candidate_paths(
                    raw,
                    workspace_dir=workspace_dir,
                    workspace_generated_dir=workspace_generated_dir,
                )
                self._rewrite_workspace_source_paths_to_archive(
                    raw,
                    workspace_dir=workspace_dir,
                    archived_source_snapshot=call_dir / "source_snapshot",
                )
                raw.setdefault("source_family", self.config.progressive_target_system)
                raw.setdefault("budget", budget)
                raw.setdefault("reference_iterations", list(reference_iterations))
                raw.setdefault("source_snapshot_path", str(call_dir / "source_snapshot"))
        normalized_pending = json.dumps(
            {"candidates": proposed},
            indent=2,
            ensure_ascii=False,
        )
        self.pending_eval_path.write_text(normalized_pending, encoding="utf-8")
        (call_dir / "pending_eval.json").write_text(normalized_pending, encoding="utf-8")

        evaluated = self._evaluate_proposed(iteration, proposed, examples)
        best_ids = self._quality_frontier_ids(existing_candidates + evaluated)
        write_post_eval_artifacts(
            run_dir=self.run_dir,
            call_dir=call_dir,
            iteration=iteration,
            candidates=evaluated,
            frontier_ids=best_ids,
        )
        self.trace_harness.record_iteration(
            iteration=iteration,
            candidates=evaluated,
            patch_base=curaii_base_iter,
            budget=budget,
            selection_policy=policy_name,
            proposer_call_dir=str(call_dir),
        )
        self.run_store.record_eval(iteration, evaluated)
        if evaluated:
            self.run_store.commit_iteration(iteration)
        self._refresh_evidence_store(iteration)
        self._refresh_run_indexes(existing_candidates + evaluated)
        return evaluated

    def _build_progressive_workspace(
        self,
        *,
        iteration: int,
        budget: str,
        existing_candidates: list[CandidateResult],
        call_dir: Path,
        reference_iterations_override: tuple[int, ...] | None = None,
        bandit_policy: dict[str, Any] | None = None,
        base_iter: int | None = None,
    ) -> tuple[Path, tuple[int, ...]]:
        call_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = call_dir / "workspace"
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        workspace_generated_dir = workspace_dir / "generated"
        workspace_generated_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(workspace_generated_dir, root=workspace_generated_dir)

        if reference_iterations_override is not None:
            # Caller-supplied refs are authoritative — used by bandit and by
            # curaii when the policy explicitly wants the chosen base to
            # appear as a reference (low/medium budgets).
            reference_iterations = reference_iterations_override
        else:
            reference_iterations = self._reference_iterations_for_budget(
                budget,
                iteration=iteration,
                candidates=existing_candidates,
            )
            if base_iter is not None:
                # No override → strip base out of default refs.  The patch
                # base is already materialized inside `project_source/`, so
                # leaving it in refs would duplicate the same candidate as
                # both base and reference.
                reference_iterations = tuple(
                    item for item in reference_iterations if item != base_iter
                )
        assignment = {
            "iteration": iteration,
            "target_system": self.config.progressive_target_system,
            "budget": budget,
            "reference_iterations": list(reference_iterations),
            "generated_dir": str(workspace_generated_dir),
            "source_snapshot_dir": str(workspace_dir / "source_snapshot"),
            "pending_eval_path": str(workspace_dir / "pending_eval.json"),
            "base_iter": base_iter,
        }
        if bandit_policy:
            assignment["bandit_policy"] = bandit_policy
        for dest in (call_dir / "assignment.json", workspace_dir / "assignment.json"):
            dest.write_text(json.dumps(assignment, indent=2, ensure_ascii=False), encoding="utf-8")

        if self._summaries_in_workspace_enabled():
            self._copy_workspace_summaries(workspace_dir / "summaries")
        self._copy_workspace_traces(workspace_dir / "traces")
        self._copy_reference_iterations(
            workspace_dir / "reference_iterations",
            reference_iterations=reference_iterations,
        )
        self._build_source_snapshot_workspace(
            iteration=iteration,
            source_family=self.config.progressive_target_system,
            call_dir=call_dir,
            target_system=self.config.progressive_target_system,
            snapshot_root=workspace_dir / "source_snapshot",
            generated_dir=workspace_generated_dir,
            base_iter=base_iter,
        )
        self._write_workspace_manifest(
            workspace_dir,
            call_dir=call_dir,
            assignment=assignment,
        )
        self._write_access_policy(
            workspace_dir,
            source_snapshot_dir=workspace_dir / "source_snapshot",
            generated_dir=workspace_generated_dir,
            pending_eval_path=workspace_dir / "pending_eval.json",
            bandit_policy=bandit_policy,
        )
        self._copy_workspace_state(workspace_dir / "state.md")
        self._prepare_workspace_evidence_store(iteration)
        self._deploy_mcp_server_assets(workspace_dir)
        self._write_proposer_agent_config(workspace_dir)
        self._deploy_proposer_skill(workspace_dir)
        return workspace_dir, reference_iterations

    def _uses_codex_proposer(self) -> bool:
        return self.config.proposer_agent.strip().lower() == "codex"

    def _codex_mcp_servers(self, workspace_dir: Path) -> dict[str, dict[str, Any]]:
        """Build the per-invocation MCP server spec for Codex.

        Returns ``{}`` when the evidence-tools surface is disabled (the
        ablation that hides the trace-harness section also disables the
        evidence-store tools). Mirrors ``_write_proposer_agent_config``
        on the Claude path, but emits a Python-dict spec that the runner
        translates into ``-c mcp_servers.evidence-tools.*`` overrides.

        Codex spawns MCP server subprocesses with the codex session's
        cwd (the workspace) as their working directory. EVIDENCE_DB and
        PYTHONPATH come out of ``_evidence_mcp_server_env`` as relative
        paths anchored at the project root, so we resolve them to
        absolute here — otherwise the MCP server would try to open
        ``<workspace>/runs/<run>/evidence_store.db`` and fail.
        """

        if not self.config.proposer_show_trace_harness_section:
            return {}
        evidence_env = dict(self._evidence_mcp_server_env(workspace_dir))
        for key in ("EVIDENCE_DB", "PYTHONPATH"):
            value = evidence_env.get(key)
            if value:
                evidence_env[key] = str(Path(value).resolve(strict=False))
        return {
            "evidence-tools": {
                "command": self._mcp_python_command(),
                "args": ["-m", "optimizer1.evidence_store_mcp_server"],
                "env": evidence_env,
            },
        }

    def _uses_claude_subagent_proposer(self) -> bool:
        """Whether the proposer runs through the Claude Code CLI.

        For a Claude proposer the self-contained benchmark skill is
        delivered via ``--append-system-prompt``; for a Codex proposer
        it is written into ``<workspace>/AGENTS.md``.
        """

        return self.config.proposer_agent.strip().lower() == "claude"

    def _proposer_skill_mode(self) -> str:
        """Return the evidence-workflow mode for the proposer skill."""

        if not self.config.organized:
            return "default"
        if self.config.organized_include_summaries:
            return "organized-summaries"
        return "organized"

    def _proposer_skill_key(self) -> str:
        """Return the benchmark skill key for this run."""

        from optimizer1.prompts import benchmark_skill_name

        return benchmark_skill_name(
            benchmark_name=self._benchmark_prompt_name(),
            target_system=self.config.progressive_target_system,
        )

    def _resolve_proposer_skill(self) -> str:
        """Return the resolved per-benchmark proposer skill text.

        The skill is the proposer's full self-contained contract for
        this benchmark, with the active evidence-workflow mode block
        kept. It is delivered through the system-prompt channel.
        """

        from optimizer1.prompts import load_proposer_skill

        return load_proposer_skill(
            self._proposer_skill_key(), self._proposer_skill_mode()
        )

    def _deploy_proposer_skill(self, workspace_dir: Path) -> None:
        """Deploy the resolved per-benchmark proposer skill into the workspace.

        The skill is the proposer's full self-contained contract — role,
        objective, search space, workflow, quality gate, and
        ``pending_eval.json`` conventions — for this benchmark.

          - Codex proposer: the skill is written to ``<workspace>/AGENTS.md``,
            which Codex auto-loads at session start.
          - Claude proposer: the skill is delivered at invocation time via
            ``--append-system-prompt``; an audit copy is written to
            ``<workspace>/PROPOSER_SKILL.md``.

        No-op when the proposer agent is neither Codex nor Claude.
        """

        if self._uses_codex_proposer():
            self._deploy_codex_agents_md(workspace_dir, self._resolve_proposer_skill())
            return
        if not self._uses_claude_subagent_proposer():
            return
        (workspace_dir / "PROPOSER_SKILL.md").write_text(
            self._resolve_proposer_skill(), encoding="utf-8"
        )

    def _deploy_codex_agents_md(self, workspace_dir: Path, skill_text: str) -> None:
        """Write the Codex-facing ``<workspace>/AGENTS.md``.

        Codex auto-loads ``AGENTS.md`` at the session cwd and prepends it
        to its system prompt. We write the resolved per-benchmark proposer
        skill — the same contract the Claude path delivers via
        ``--append-system-prompt``.
        """

        body = (
            "# Codex proposer — auto-loaded contract\n\n"
            "Codex loads this file at session start. It is the proposer's "
            "self-contained skill for this benchmark: role, objective, "
            "search space, workflow, quality gate, and pending_eval.json "
            "conventions. Treat it as binding for this session.\n\n"
            "---\n\n"
            f"{skill_text.rstrip()}\n"
        )
        (workspace_dir / "AGENTS.md").write_text(body, encoding="utf-8")

    def _write_proposer_agent_config(self, workspace_dir: Path) -> None:
        """Register proposer-facing MCP servers in Claude settings.

        Writes ``<workspace>/.claude/settings.local.json`` with an
        ``mcpServers`` entry. Skipped when the trace-harness section is
        suppressed; in that ablation the proposer gets no historical
        query tools.

        No-op when the Codex proposer is selected: Codex has no
        per-workspace MCP config file; the evidence-tools server is
        injected at exec time via ``-c mcp_servers.evidence-tools.*``
        flags from :meth:`_codex_mcp_servers`.
        """

        if self._uses_codex_proposer():
            return
        if not self.config.proposer_show_trace_harness_section:
            return
        evidence_env = self._evidence_mcp_server_env(workspace_dir)
        command = self._mcp_python_command()
        self._write_claude_settings(
            workspace_dir,
            servers={
                "evidence-tools": {
                    "command": command,
                    "args": ["-m", "optimizer1.evidence_store_mcp_server"],
                    "env": evidence_env,
                },
            },
        )

    def _mcp_server_env(self, workspace_dir: Path) -> dict[str, str]:
        env: dict[str, str] = {
            "TRACE_DB": str(self._workspace_visible_path(workspace_dir, "traces/index.db")),
            "PYTHONPATH": str(self._mcp_pythonpath(workspace_dir)),
        }
        # trace_similar lazily embeds historical diffs and the
        # proposer's query; it needs an OpenAI-compatible endpoint.
        # Forward whatever the parent process has.
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "DIFF_EMBEDDING_MODEL"):
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def _runstore_mcp_server_env(self, workspace_dir: Path) -> dict[str, str]:
        return {
            "RUNSTORE_DB": str(self._workspace_visible_path(workspace_dir, "runstore.db")),
            "PYTHONPATH": str(self._mcp_pythonpath(workspace_dir)),
        }

    def _evidence_mcp_server_env(self, workspace_dir: Path) -> dict[str, str]:
        evidence_db = (
            Path("/evidence/evidence_store.db")
            if self.config.proposer_sandbox.strip().lower() == "docker"
            else self.evidence_store.db_path
        )
        return {
            "EVIDENCE_DB": str(evidence_db),
            "PYTHONPATH": str(self._mcp_pythonpath(workspace_dir)),
        }

    def _mcp_python_command(self) -> str:
        if self.config.proposer_sandbox.strip().lower() == "docker":
            return "python"
        return sys.executable

    def _mcp_pythonpath(self, workspace_dir: Path) -> Path:
        return self._workspace_visible_path(workspace_dir, ".optimizer1_mcp_src")

    def _workspace_visible_path(self, workspace_dir: Path, rel: str) -> Path:
        if self.config.proposer_sandbox.strip().lower() == "docker":
            return Path(self.config.proposer_docker_workspace or "/workspace") / rel
        return workspace_dir / rel

    def _write_claude_settings(
        self,
        workspace_dir: Path,
        *,
        servers: dict[str, dict[str, Any]],
    ) -> None:
        """Write the workspace MCP server configuration.

        Current Claude Code loads project-scoped MCP servers from
        ``<workspace>/.mcp.json``.  Keep writing the older
        ``.claude/settings.local.json`` location as a compatibility copy for
        older CLIs, but the runner now passes ``--mcp-config .mcp.json``
        explicitly when the file exists.
        """

        settings_dir = workspace_dir / ".claude"
        settings_dir.mkdir(parents=True, exist_ok=True)
        config = {"mcpServers": servers}
        (workspace_dir / ".mcp.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (settings_dir / "settings.local.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _copy_workspace_traces(self, dest: Path) -> None:
        """Mirror run-level `traces/` into the proposer workspace.

        The proposer reads these files via the workspace mount; run-level
        paths are not directly accessible inside the sandbox, so we copy.
        """

        src = self.trace_harness.root
        if not src.exists():
            return
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            src,
            dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    def _copy_workspace_run_store(self, dest: Path) -> None:
        src = self.run_store.db_path
        if src.exists():
            shutil.copy2(src, dest)

    def _prepare_workspace_evidence_store(self, iteration: int) -> None:
        self._refresh_evidence_store(iteration)

    def _deploy_mcp_server_assets(self, workspace_dir: Path) -> None:
        src_pkg = self.project_root / "src" / "optimizer1"
        dest_pkg = workspace_dir / ".optimizer1_mcp_src" / "optimizer1"
        if dest_pkg.exists():
            shutil.rmtree(dest_pkg)
        shutil.copytree(
            src_pkg,
            dest_pkg,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    def _write_state_md(
        self,
        *,
        iteration: int,
        as_of_iteration: int,
        base_iteration: int | None,
    ) -> None:
        state_md = self.run_store.render_state_md(
            iteration=iteration,
            as_of_iteration=as_of_iteration,
            benchmark=self.workspace_spec.benchmark,
            base_iteration=base_iteration,
        )
        self.run_store.record_state_snapshot(iteration, state_md)
        (self.run_dir / "state.md").write_text(state_md, encoding="utf-8")

    def _copy_workspace_state(self, dest: Path) -> None:
        if not self.config.organized:
            return
        src = self.run_dir / "state.md"
        if src.exists():
            shutil.copy2(src, dest)

    def _refresh_evidence_store(self, iteration: int) -> None:
        """Best-effort unified evidence-store refresh.

        The legacy run artifacts remain the source of truth during rollout, so
        an evidence import issue should be visible but must not burn an
        optimization iteration.
        """

        try:
            self.evidence_store.refresh(iteration=iteration)
        except Exception as exc:  # noqa: BLE001 - diagnostic sidecar only
            self.run_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "iteration": int(iteration),
                "error": str(exc),
            }
            with (self.run_dir / "evidence_store_errors.jsonl").open(
                "a",
                encoding="utf-8",
            ) as fh:
                fh.write(json.dumps(row, ensure_ascii=False))
                fh.write("\n")

    def _copy_workspace_summaries(self, summaries_dir: Path) -> None:
        summaries_dir.mkdir(parents=True, exist_ok=True)
        summary_files = (
            (self.summary_path, "evolution_summary.jsonl", ""),
            (self.frontier_path, "best_candidates.json", "[]\n"),
            (self.candidate_score_table_path, "candidate_score_table.json", "[]\n"),
            (
                self.retrieval_diagnostics_summary_path,
                "retrieval_diagnostics_summary.json",
                "[]\n",
            ),
            (self.iteration_index_path, "iteration_index.json", "[]\n"),
            (self.diff_summary_path, "diff_summary.jsonl", ""),
        )
        for src, name, default_text in summary_files:
            dest = summaries_dir / name
            if src.exists():
                shutil.copy2(src, dest)
            else:
                dest.write_text(default_text, encoding="utf-8")

    def _summaries_in_workspace_enabled(self) -> bool:
        if not self.config.summaries_in_workspace:
            return False
        if self.config.organized and not self.config.organized_include_summaries:
            return False
        return True

    def _copy_reference_iterations(
        self,
        reference_dir: Path,
        *,
        reference_iterations: tuple[int, ...],
    ) -> None:
        reference_dir.mkdir(parents=True, exist_ok=True)
        for item in reference_iterations:
            src = self._iteration_dir(item)
            if not src.exists():
                continue
            self._copy_iteration_bundle(
                src,
                reference_dir / f"iter_{item:03d}",
            )

    def _write_workspace_manifest(
        self,
        workspace_dir: Path,
        *,
        call_dir: Path,
        assignment: dict[str, Any],
    ) -> None:
        manifest = {
            "workspace_dir": str(workspace_dir),
            "call_dir": str(call_dir),
            "assignment": assignment,
            "summaries_dir": str(workspace_dir / "summaries"),
            "reference_iterations_dir": str(workspace_dir / "reference_iterations"),
            "source_snapshot_dir": str(workspace_dir / "source_snapshot"),
            "generated_dir": str(workspace_dir / "generated"),
            "pending_eval_path": str(workspace_dir / "pending_eval.json"),
        }
        for dest in (
            workspace_dir / "workspace_manifest.json",
            call_dir / "workspace_manifest.json",
        ):
            dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_access_policy(
        self,
        workspace_dir: Path,
        *,
        source_snapshot_dir: Path,
        generated_dir: Path,
        pending_eval_path: Path,
        bandit_policy: dict[str, Any] | None = None,
    ) -> None:
        policy = {
            "read_roots": [str(workspace_dir)],
            "write_roots": [
                str(source_snapshot_dir / "candidate"),
                str(generated_dir),
            ],
            "write_files": [str(pending_eval_path)],
            "forbidden_roots": [
                str(self.project_root),
                str(self.run_dir),
                str(self.project_root / "references" / "vendor"),
                str(self.run_dir / "candidate_results"),
            ],
            "notes": [
                "The proposer workspace is self-contained.",
                (
                    "Do not read global runs, repo-root source, references/vendor, "
                    f"{self._raw_data_policy_name()}, or scoring helpers."
                ),
            ],
        }
        if bandit_policy:
            policy.update(
                {
                    "hot_paths": list(bandit_policy.get("hot_files", ())),
                    "warm_paths": list(bandit_policy.get("warm_files", ())),
                    "cold_paths": list(bandit_policy.get("cold_files", ())),
                    "read_budget_lines_by_path": dict(
                        bandit_policy.get("read_budget_lines_by_path", {})
                    ),
                }
            )
        for dest in (
            workspace_dir / "access_policy.json",
            workspace_dir.parent / "access_policy.json",
        ):
            dest.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")

    def _archive_workspace_outputs(
        self,
        *,
        workspace_dir: Path,
        call_dir: Path,
        result: Any,
    ) -> None:
        self._sync_workspace_outputs(workspace_dir=workspace_dir, call_dir=call_dir)

        source_snapshot = workspace_dir / "source_snapshot"
        archived_snapshot = call_dir / "source_snapshot"
        if source_snapshot.exists():
            if archived_snapshot.exists():
                shutil.rmtree(archived_snapshot)
            shutil.copytree(
                source_snapshot,
                archived_snapshot,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        for name in ("workspace_manifest.json", "access_policy.json"):
            self._copy_if_exists(workspace_dir / name, call_dir / name)

        agent_dir = call_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        tool_access = getattr(result, "tool_access", None)
        if isinstance(tool_access, dict):
            (agent_dir / "tool_access.json").write_text(
                json.dumps(tool_access, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, dict):
            (agent_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        self._write_source_snapshot_diff(call_dir)
        write_diff_digest(call_dir=call_dir)
        self._append_diff_summary(call_dir)

    def _write_source_snapshot_diff(self, call_dir: Path) -> None:
        original = call_dir / "source_snapshot" / "candidate" / "original_project_source"
        updated = call_dir / "source_snapshot" / "candidate" / "project_source"
        if not original.exists() or not updated.exists():
            (call_dir / "diff.patch").write_text(
                "Source snapshot diff unavailable.\n",
                encoding="utf-8",
            )
            return
        cmd = ["git", "diff", "--no-index", "--", str(original), str(updated)]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                text=True,
                capture_output=True,
                timeout=30,
            )
            text = completed.stdout or completed.stderr or ""
        except Exception as exc:  # noqa: BLE001 - best-effort artifact
            text = f"Failed to capture source snapshot diff: {exc}\n"
        (call_dir / "diff.patch").write_text(text, encoding="utf-8")

    def _append_diff_summary(self, call_dir: Path) -> None:
        iteration = _iteration_from_dir_name(call_dir.name) or 0
        diff_path = call_dir / "diff.patch"
        text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
        stats = diff_stats(text)
        self.run_store.record_diff(iteration, text)
        self._refresh_evidence_store(iteration)
        row = {
            "iteration": iteration,
            "iteration_dir": str(call_dir),
            "diff_path": str(diff_path),
            "diff_digest_path": str(call_dir / "diff_digest.md"),
            "files_changed": stats["files_changed"],
            "insertions": stats["insertions"],
            "deletions": stats["deletions"],
        }
        rows: list[dict[str, Any]] = []
        if self.diff_summary_path.exists():
            for raw in self.diff_summary_path.read_text(encoding="utf-8").splitlines():
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and int(item.get("iteration") or -1) != iteration:
                    rows.append(item)
        rows.append(row)
        self.diff_summary_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
            encoding="utf-8",
        )

    def _load_examples(self) -> list[LocomoExample]:
        return self._load_examples_for_split(self.config.split, self.config.limit)

    def _load_examples_for_split(self, split: str, limit: int = 0) -> list[LocomoExample]:
        if not default_data_path().exists():
            prepare_locomo()
        examples = load_locomo_examples()
        selected = select_split(examples, split=split)
        if limit:
            selected = selected[:limit]
        return selected

    def _run_seed_frontier(self) -> dict[str, Any]:
        return run_initial_frontier(
            split=self.config.split,
            limit=self.config.limit,
            out_dir=self.run_dir,
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout_s=self.config.eval_timeout_s,
            dry_run=self.config.dry_run,
            max_context_chars=self.config.max_context_chars,
            max_eval_workers=self.config.max_eval_workers,
            pareto_quality_threshold=self.config.pareto_quality_threshold,
            scaffolds=self.config.scaffolds,
            scaffold_extra=self.config.scaffold_extra,
        )

    def _run_test_frontier(self, candidates: list[CandidateResult]) -> dict[str, Any]:
        full_frontier = self._quality_frontier(candidates)
        candidate_limit = max(0, int(self.config.test_frontier_candidate_limit or 0))
        frontier = full_frontier[:candidate_limit] if candidate_limit else full_frontier
        test_dir = self.run_dir / "test_frontier"
        specs_dir = test_dir / "candidate_specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        examples = self._load_examples_for_split(self.config.test_split, self.config.test_limit)
        runner = self._make_evaluation_runner(examples, out_dir=test_dir)

        rows: list[dict[str, Any]] = []
        test_results: list[CandidateResult] = []
        failures: list[dict[str, Any]] = []
        for candidate in frontier:
            spec = self._candidate_test_spec(candidate)
            spec_path = specs_dir / f"{spec['candidate_id']}.json"
            spec_path.write_text(
                json.dumps(spec, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            try:
                scaffold = load_candidate_scaffold(spec, project_root=self.project_root)
                config = ScaffoldConfig(
                    top_k=int(spec.get("top_k", 8)),
                    window=int(spec.get("window", 1)),
                    extra=dict(spec.get("extra") or {}),
                )
                result = runner.evaluate_scaffold(
                    scaffold=scaffold,
                    scaffold_name=str(spec.get("name") or candidate.scaffold_name),
                    config=config,
                    candidate_id=str(spec["candidate_id"]),
                )
            except Exception as exc:  # noqa: BLE001 - keep testing the rest of the frontier
                failure = {
                    "original_candidate_id": candidate.candidate_id,
                    "test_candidate_id": spec["candidate_id"],
                    "candidate_spec_path": str(spec_path),
                    "error": str(exc),
                }
                failures.append(failure)
                rows.append(
                    {
                        "original_candidate": candidate.to_dict(),
                        "candidate_spec_path": str(spec_path),
                        "error": str(exc),
                    }
                )
                self._append_event(
                    {
                        "event": "test_frontier_candidate_failed",
                        **failure,
                    }
                )
                continue

            test_results.append(result)
            rows.append(
                {
                    "original_candidate": candidate.to_dict(),
                    "candidate_spec_path": str(spec_path),
                    "test_candidate": result.to_dict(),
                }
            )

        results_path = test_dir / "test_results.json"
        results_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
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
            "candidate_spec_dir": str(specs_dir),
            "failures": failures,
        }
        summary_path = self.run_dir / "test_frontier_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_path)
        return summary

    def _candidate_test_spec(self, candidate: CandidateResult) -> dict[str, Any]:
        config = candidate.config if isinstance(candidate.config, dict) else {}
        extra = self._candidate_extra(candidate)
        top_k, _ = _single_top_k(config.get("top_k", 8))
        spec: dict[str, Any] = {
            "name": candidate.scaffold_name,
            "candidate_id": self._test_candidate_id(candidate.candidate_id),
            "original_candidate_id": candidate.candidate_id,
            "top_k": top_k,
            "window": int(config.get("window", 1)),
            "extra": extra,
        }
        for key in (
            "build_tag",
            "candidate_root",
            "class",
            "factory",
            "generated_dir",
            "module",
            "module_path",
            "project_source_path",
            "source_project_path",
            "memomemo_source_path",
        ):
            if key in extra:
                spec[key] = extra[key]

        has_source_project = any(
            spec.get(key)
            for key in ("source_project_path", "project_source_path", "memomemo_source_path")
        )
        has_dynamic_module = bool(spec.get("module") or spec.get("module_path"))
        if has_source_project:
            spec["scaffold_name"] = self._source_scaffold_name(candidate)
        elif not has_dynamic_module:
            spec["scaffold_name"] = str(extra.get("scaffold_name") or candidate.scaffold_name)
        return spec

    def _source_scaffold_name(self, candidate: CandidateResult) -> str:
        extra = self._candidate_extra(candidate)
        explicit = str(extra.get("scaffold_name") or "").strip()
        if explicit:
            return explicit
        return {
            "memgpt": "memgpt_source",
        }.get(self._infer_source_family(candidate), candidate.scaffold_name)

    def _test_candidate_id(self, candidate_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id).strip("._-")
        return f"test_{safe or 'candidate'}"

    def _benchmark_prompt_name(self) -> str:
        return "LOCOMO conversational-memory QA"

    def _raw_data_policy_name(self) -> str:
        return "raw LOCOMO data"

    def _run_proposer_agent(
        self,
        prompt: str,
        *,
        log_dir: Path,
        name: str,
        cwd: Path | None = None,
    ) -> Any:
        agent = self.config.proposer_agent.strip().lower()
        kwargs: dict[str, Any] = dict(
            cwd=cwd or self.project_root,
            log_dir=log_dir,
            name=name,
            timeout_s=self.config.propose_timeout_s,
            sandbox=self._proposer_sandbox_config(),
            model=self.config.claude_model,
            effort=self.config.claude_effort,
            claude_base_url=self.config.claude_base_url,
            claude_auth_token=self.config.claude_auth_token,
            claude_native_auth=self.config.claude_native_auth,
            codex_model=self.config.codex_model,
            codex_reasoning_effort=self.config.codex_reasoning_effort,
            codex_home=self.config.codex_home or None,
        )
        # The proposer's self-contained benchmark skill is delivered
        # through the system-prompt channel via --append-system-prompt,
        # so the role / identity / contract text never enters the user
        # message. (Codex receives the same skill from <workspace>/AGENTS.md,
        # written by _deploy_proposer_skill.)
        if self._uses_claude_subagent_proposer() and name == "proposer":
            kwargs["claude_append_system_prompt"] = self._resolve_proposer_skill()
        # Codex has no per-workspace MCP config flag like Claude's
        # --mcp-config: we inject the evidence-tools server via
        # -c mcp_servers.evidence-tools.* on every exec instead.
        if self._uses_codex_proposer() and cwd is not None:
            kwargs["codex_mcp_servers"] = self._codex_mcp_servers(cwd)

        retries = 0
        while True:
            # Re-sync the docker proposer's staged Claude OAuth credentials
            # with the host's before every attempt. Crucial after a long
            # rate-limit wait: the per-run STAGE_HOME copy made at launch
            # time would otherwise have expired (→ 401 cascade) while we
            # slept out the 429 window.
            self._refresh_native_auth_credentials()
            result = run_code_agent_prompt(prompt, agent=agent, **kwargs)
            if not (
                self.config.wait_on_rate_limit
                and getattr(result, "rate_limited", False)
                and retries < self.config.rate_limit_max_retries
            ):
                return result
            retries += 1
            self._wait_for_rate_limit_reset(
                result, retry=retries, name=name, log_dir=log_dir
            )

    def _wait_for_rate_limit_reset(
        self, result: Any, *, retry: int, name: str, log_dir: Path
    ) -> None:
        """Sleep until an Anthropic usage-limit window resets, logging a
        ``proposer_rate_limited`` event and periodic progress.

        Called from :meth:`_run_proposer_agent` before retrying the same
        invocation, so a 429 does not consume the iteration."""

        now = time.time()
        resets_at = getattr(result, "rate_limit_resets_at", None)
        if isinstance(resets_at, (int, float)) and resets_at > now:
            wait_s = (resets_at - now) + self.config.rate_limit_buffer_s
        else:
            wait_s = float(self.config.rate_limit_default_wait_s)
        wait_s = max(1.0, min(wait_s, float(self.config.rate_limit_max_wait_s)))
        resume_at = now + wait_s
        self._append_event(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "event": "proposer_rate_limited",
                "name": name,
                "log_dir": str(log_dir),
                "retry": retry,
                "max_retries": self.config.rate_limit_max_retries,
                "resets_at": resets_at,
                "wait_s": round(wait_s, 1),
                "resume_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(resume_at)
                ),
            }
        )
        poll = max(1, int(self.config.rate_limit_poll_log_s))
        print(
            f"[{self.config.run_id}] {name}: usage limit hit "
            f"(retry {retry}/{self.config.rate_limit_max_retries}); "
            f"waiting ~{wait_s / 60:.1f} min, resuming at "
            f"{time.strftime('%H:%M:%S', time.localtime(resume_at))}",
            flush=True,
        )
        while True:
            remaining = resume_at - time.time()
            if remaining <= 0:
                break
            chunk = min(float(poll), remaining)
            time.sleep(chunk)
            remaining = resume_at - time.time()
            if remaining > 0:
                print(
                    f"[{self.config.run_id}] {name}: still rate-limited, "
                    f"~{remaining / 60:.1f} min left",
                    flush=True,
                )

    def _native_auth_stage_home(self) -> Path | None:
        """Locate the host directory bind-mounted at the docker proposer's
        ``$HOME`` under ``--claude-native-auth``.

        That directory holds the per-run copy of ``~/.claude.json`` +
        ``~/.claude/.credentials.json`` (see the launch script's
        ``prepare_proposer_home``); identifying it lets us keep the staged
        OAuth credentials in sync with the host's. Returns ``None`` when
        native auth is off, the proposer is not sandboxed in docker, or no
        mount targets the container ``$HOME``."""

        if not self.config.claude_native_auth:
            return None
        if self.config.proposer_sandbox.strip().lower() != "docker":
            return None
        home = (self.config.proposer_docker_home or "").rstrip("/")
        if not home:
            return None
        found: Path | None = None
        for spec in self.config.proposer_docker_mount:
            parts = str(spec).split(":")
            if len(parts) < 2 or not parts[0]:
                continue
            if parts[1].rstrip("/") == home:
                found = Path(parts[0])  # last mount wins, matching docker
        return found

    def _refresh_native_auth_credentials(self) -> None:
        """Copy the host's current ``~/.claude/.credentials.json`` into the
        docker proposer's staged ``$HOME`` so a token the host CLI has since
        rotated/refreshed is the one the container sees.

        No-op unless ``--claude-native-auth`` + a docker proposer with a
        mount at the container ``$HOME``. Best-effort: a transient read/copy
        failure is logged and swallowed (the proposer call then fails/retries
        exactly as it would have without this sync)."""

        stage = self._native_auth_stage_home()
        if stage is None:
            return
        host_creds = Path.home() / ".claude" / ".credentials.json"
        if not host_creds.is_file():
            return
        dst = stage / ".claude" / ".credentials.json"
        try:
            if dst.is_file() and dst.read_bytes() == host_creds.read_bytes():
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(host_creds, dst)
            dst.chmod(0o600)
        except OSError as exc:
            self._append_event(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "event": "native_auth_refresh_failed",
                    "stage_home": str(stage),
                    "error": repr(exc),
                }
            )

    def _proposer_sandbox_config(self) -> ProposerSandboxConfig | None:
        kind = self.config.proposer_sandbox.strip().lower()
        if kind == "none":
            return None
        if kind != "docker":
            raise ValueError(f"unsupported proposer sandbox: {self.config.proposer_sandbox!r}")
        docker_env = _dedupe_tuple(
            DEFAULT_DOCKER_ENV_VARS + self.config.proposer_docker_env
        )
        docker_mount = tuple(self.config.proposer_docker_mount)
        evidence_db = self.evidence_store.db_path.resolve(strict=False)
        if evidence_db.exists():
            docker_mount = (
                *docker_mount,
                f"{evidence_db}:/evidence/evidence_store.db:ro",
            )
        docker_mount = _dedupe_tuple(docker_mount)
        return ProposerSandboxConfig(
            kind="docker",
            docker_image=self._effective_proposer_docker_image(),
            docker_workspace=self.config.proposer_docker_workspace or "/workspace",
            docker_env_vars=docker_env,
            docker_mounts=docker_mount,
            docker_user=self.config.proposer_docker_user,
            docker_home=self.config.proposer_docker_home,
        )

    def _effective_proposer_docker_image(self) -> str:
        configured = self.config.proposer_docker_image.strip()
        if configured:
            return configured
        return DEFAULT_PROPOSER_DOCKER_IMAGE

    def _evaluate_proposed(
        self,
        iteration: int,
        proposed: list[dict[str, Any]],
        examples: list[LocomoExample],
    ) -> list[CandidateResult]:
        runner = self._make_evaluation_runner(examples)
        results: list[CandidateResult] = []
        for raw in proposed:
            if isinstance(raw, dict):
                raw = dict(raw)
                raw.setdefault("candidate_root", str(self.generated_dir))
            violations = self._candidate_code_policy_violations(raw)
            if violations:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_policy_rejected",
                        "candidate": raw,
                        "violations": violations,
                    }
                )
                continue
            try:
                scaffold = load_candidate_scaffold(raw, project_root=self.project_root)
            except Exception as exc:  # noqa: BLE001 - log and continue
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_import_failed",
                        "candidate": raw,
                        "error": str(exc),
                    }
                )
                continue

            top_k, top_k_adjusted = _single_top_k(raw.get("top_k", 8))
            if top_k_adjusted:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_top_k_adjusted",
                        "candidate": raw,
                        "evaluated_top_k": top_k,
                    }
                )
            extra = dict(raw.get("extra") or {})
            for key in (
                "build_tag",
                "candidate_root",
                "class",
                "cost_level",
                "factory",
                "generated_dir",
                "module",
                "module_path",
                "project_source_path",
                "scaffold_name",
                "source_base_dir",
                "source_family",
                "source_path",
                "source_project_path",
                "upstream_source_path",
                "memgpt_source_path",
                "optimization_target",
            ):
                if key in raw and key not in extra:
                    extra[key] = raw[key]
            for key, value in self._candidate_extra_defaults().items():
                extra.setdefault(key, value)
            config = ScaffoldConfig(
                top_k=top_k,
                window=int(raw.get("window", 1)),
                extra=extra,
            )
            candidate_name = str(raw.get("name") or scaffold.name)
            candidate_id = f"iter{iteration:03d}_{candidate_name}_top{top_k}"
            try:
                result = runner.evaluate_scaffold(
                    scaffold=scaffold,
                    scaffold_name=candidate_name,
                    config=config,
                    candidate_id=candidate_id,
                )
            except Exception as exc:  # noqa: BLE001 - log and continue
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_eval_failed",
                        "candidate": raw,
                        "candidate_id": candidate_id,
                        "error": str(exc),
                    }
                )
                continue
            results.append(result)
            self._append_summary(iteration=iteration, candidate=result, proposal=raw)
        return results

    def _make_evaluation_runner(
        self,
        examples: list[LocomoExample],
        *,
        out_dir: Path | None = None,
    ) -> EvaluationRunner:
        return EvaluationRunner(
            examples=examples,
            out_dir=out_dir or self.run_dir,
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout_s=self.config.eval_timeout_s,
            dry_run=self.config.dry_run,
            max_context_chars=self.config.max_context_chars,
            max_eval_workers=self.config.max_eval_workers,
        )

    def _candidate_extra_defaults(self) -> dict[str, object]:
        return {}

    def _candidate_code_policy_violations(self, candidate: Any) -> list[dict[str, str]]:
        if not isinstance(candidate, dict):
            return []
        paths = self._candidate_policy_scan_paths(candidate)
        violations: list[dict[str, str]] = []
        forbidden = {
            "candidate_results": "runtime code must not read previous candidate results",
            "data/locomo": "runtime code must not read raw LOCOMO data",
            "data\\locomo": "runtime code must not read raw LOCOMO data",
            "locomo10.json": "runtime code must not read raw LOCOMO data",
            "data/longmemeval": "runtime code must not read raw LongMemEval data",
            "data\\longmemeval": "runtime code must not read raw LongMemEval data",
            "longmemeval_s_cleaned.json": "runtime code must not read raw LongMemEval data",
            "longmemeval_m_cleaned.json": "runtime code must not read raw LongMemEval data",
            "longmemeval_oracle.json": "runtime code must not read raw LongMemEval data",
            "score_prediction": "runtime code must not call OptiHarness scoring helpers",
            "optimizer1.metrics": "runtime code must not import OptiHarness scoring helpers",
            "load_locomo_examples": "runtime code must not load the full LOCOMO dataset",
            "load_longmemeval_examples": "runtime code must not load the full LongMemEval dataset",
        }
        source_project = self._candidate_source_project_root(candidate)
        original_source_project = (
            self._candidate_original_source_project_root(source_project)
            if source_project is not None
            else None
        )
        for path in paths:
            text = self._candidate_policy_scan_text(
                path,
                source_project=source_project,
                original_source_project=original_source_project,
            )
            if text is None:
                continue
            lower = text.lower()
            for marker, reason in forbidden.items():
                if marker.lower() in lower:
                    violations.append(
                        {
                            "path": str(path),
                            "marker": marker,
                            "reason": reason,
                        }
                    )
        return violations

    def _candidate_policy_scan_text(
        self,
        path: Path,
        *,
        source_project: Path | None,
        original_source_project: Path | None,
    ) -> str | None:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        if source_project is None or original_source_project is None:
            return text
        try:
            rel = path.resolve(strict=False).relative_to(source_project.resolve(strict=False))
        except ValueError:
            return text
        original_path = original_source_project / rel
        try:
            original_text = original_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return text
        if text == original_text:
            return ""
        return _added_policy_lines(original_text, text)

    def _proposer_access_violations(
        self,
        result: Any,
        *,
        workspace_dir: Path,
    ) -> list[dict[str, str]]:
        tool_access = getattr(result, "tool_access", None)
        if not isinstance(tool_access, dict):
            return []
        if self._proposer_allows_container_internal_access():
            return []

        allowed_roots = [workspace_dir]
        violations: list[dict[str, str]] = []

        for raw_path in sorted((tool_access.get("files_read") or {}).keys()):
            path = self._normalize_agent_access_path(raw_path, base_dir=workspace_dir)
            if not self._path_is_under_any(path, allowed_roots):
                violations.append(
                    {
                        "operation": "read",
                        "path": str(path),
                        "reason": "proposer reads must stay inside the scoped workspace",
                    }
                )

        for raw_path in sorted((tool_access.get("files_written") or {}).keys()):
            path = self._normalize_agent_access_path(raw_path, base_dir=workspace_dir)
            if not self._path_is_under_any(path, allowed_roots):
                violations.append(
                    {
                        "operation": "write",
                        "path": str(path),
                        "reason": "proposer writes must stay inside the scoped workspace",
                    }
                )
        return violations

    def _proposer_allows_container_internal_access(self) -> bool:
        return self.config.proposer_sandbox.strip().lower() == "docker"

    def _access_retry_note(
        self,
        *,
        violations: list[dict[str, str]],
        workspace_dir: Path,
    ) -> str:
        lines = [
            "## Retry Required: Filesystem Boundary Violation",
            "",
            "Your previous attempt read or wrote files outside the proposer workspace.",
            f"Allowed workspace root: `{workspace_dir.resolve(strict=False)}`",
            "",
            "Do not use absolute paths or `..` paths that leave this directory.",
            "Use only the files copied into the current working directory for this proposer call.",
            "Recreate exactly one candidate from scratch and write a fresh `pending_eval.json`.",
            "",
            "Violations from the previous attempt:",
        ]
        for item in violations:
            operation = item.get("operation", "access")
            path = item.get("path", "")
            reason = item.get("reason", "")
            lines.append(f"- {operation}: `{path}` ({reason})")
        return "\n".join(lines)

    def _normalize_agent_access_path(
        self,
        raw_path: str,
        *,
        base_dir: Path | None = None,
    ) -> Path:
        path = Path(str(raw_path)).expanduser()
        if path.is_absolute() and base_dir is not None:
            path = self._map_container_workspace_path(path, workspace_dir=base_dir)
        if not path.is_absolute():
            path = (base_dir or self.project_root) / path
        return path.resolve(strict=False)

    def _path_is_under_any(self, path: Path, roots: list[Path]) -> bool:
        normalized = path.resolve(strict=False)
        for root in roots:
            root_path = root.resolve(strict=False)
            if normalized == root_path or root_path in normalized.parents:
                return True
        return False

    def _path_matches_any(self, path: Path, files: list[Path]) -> bool:
        normalized = path.resolve(strict=False)
        return any(normalized == item.resolve(strict=False) for item in files)

    def _candidate_policy_scan_paths(self, candidate: dict[str, Any]) -> list[Path]:
        out: list[Path] = []
        module_path = str(candidate.get("module_path") or "").strip()
        if module_path:
            path = Path(module_path).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            out.append(path)

        candidate_root = candidate.get("candidate_root") or candidate.get("generated_dir")
        root: Path | None = None
        if candidate_root:
            root = Path(str(candidate_root)).expanduser()
            if not root.is_absolute():
                root = self.project_root / root
        module_name = str(candidate.get("module") or "").strip()
        if root is not None and root.exists():
            if module_name:
                rel = Path(*module_name.split(".")).with_suffix(".py")
                module_file = root / rel
                if module_file.exists():
                    out.append(module_file)
            # Historical candidates may import earlier run-local parent modules.
            # Scan that workspace so contaminated parent modules remain visible
            # to the policy check.
            out.extend(sorted(root.glob("*.py")))

        source_project = self._candidate_source_project_root(candidate)
        if source_project is not None:
            package_dir = source_project / "src" / "optimizer1"
            if package_dir.exists():
                out.extend(sorted(package_dir.rglob("*.py")))
        return sorted(set(out))

    def _candidate_source_project_root(self, candidate: dict[str, Any]) -> Path | None:
        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        for key in ("source_project_path", "project_source_path", "memomemo_source_path"):
            value = candidate.get(key) or extra.get(key)
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            if (path / "src").exists():
                return path
            if path.name == "src":
                return path.parent
            return path
        return None

    def _candidate_original_source_project_root(self, source_project: Path) -> Path | None:
        candidate_dir = source_project.parent if source_project.name == "project_source" else None
        if candidate_dir is not None:
            original = candidate_dir / "original_project_source"
            if (original / "src").exists():
                return original

        for parent in source_project.parents:
            manifest_path = parent / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            original_value = manifest.get("original_project_source")
            if not original_value:
                continue
            original = Path(str(original_value)).expanduser()
            if not original.is_absolute():
                original = manifest_path.parent / original
            if (original / "src").exists():
                return original
        return None

    def _load_existing_candidates(self) -> list[CandidateResult]:
        out: list[CandidateResult] = []
        for path in sorted((self.run_dir / "candidate_results").glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                out.append(CandidateResult.from_dict(payload["candidate"]))
            except Exception:
                continue
        return out

    def _resume_start_iteration(self, candidates: list[CandidateResult]) -> int:
        """First iteration to (re)run on ``--resume``.

        Iterations that already produced a stored candidate are treated as
        complete (this includes iterations whose proposer ran but whose
        candidate was rejected by the selection policy — those budget rounds
        were spent). We resume at ``max(completed) + 1``; if nothing past
        the seed is stored we start at 1, and if every iteration through
        ``config.iterations`` is already done the train loop is a no-op and
        only the test-frontier step (if enabled) runs.
        """

        completed = {it for it in self._candidate_iterations(candidates) if it >= 1}
        return (max(completed) + 1) if completed else 1

    def _clean_stale_iteration_artifacts(self, start_iteration: int) -> None:
        """Remove per-iteration dirs and dangling jsonl rows for iterations
        ``>= start_iteration`` before a ``--resume`` rerun.

        Crashed/partial ``proposer_calls/iter_NNN`` dirs and the
        ``proposer_failed`` rows they left in ``evolution_summary.jsonl`` /
        ``diff_summary.jsonl`` would otherwise shadow the fresh run."""

        calls_dir = self.run_dir / "proposer_calls"
        if calls_dir.exists():
            for child in calls_dir.iterdir():
                iteration = _iteration_from_dir_name(child.name)
                if iteration is not None and iteration >= start_iteration and child.is_dir():
                    shutil.rmtree(child)

        for path in (self.summary_path, self.diff_summary_path):
            if not path.exists():
                continue
            kept: list[str] = []
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    iteration = row.get("iteration")
                except (json.JSONDecodeError, AttributeError):
                    kept.append(line)
                    continue
                if isinstance(iteration, int) and iteration >= start_iteration:
                    continue
                kept.append(line)
            path.write_text(
                ("\n".join(kept) + "\n") if kept else "", encoding="utf-8"
            )

    def _rederive_progressive_state(
        self, candidates: list[CandidateResult], *, start_iteration: int
    ) -> None:
        """Rebuild ``progressive_state.json`` from the surviving candidates so
        ``--resume`` is not penalised by crashed iterations that bumped
        ``stagnation_count`` / cycled the budget without producing a row.

        Mirrors ``_update_progressive_state``'s pareto-style advance rule
        (best passrate must strictly increase) replayed over the stored
        iterations in order."""

        best_per_iter: dict[int, CandidateResult] = {}
        for candidate in candidates:
            iteration = _candidate_iteration(candidate.candidate_id)
            if iteration is None or iteration < 1:
                continue
            current = best_per_iter.get(iteration)
            if current is None or _candidate_score(candidate) > _candidate_score(current):
                best_per_iter[iteration] = candidate

        running_best_passrate = self._seed_passrate(candidates)
        last_improved = 0
        for iteration in sorted(best_per_iter):
            if best_per_iter[iteration].passrate > running_best_passrate:
                running_best_passrate = best_per_iter[iteration].passrate
                last_improved = iteration
        stagnation = max(0, (start_iteration - 1) - last_improved)

        overall_best = max(candidates, key=_candidate_score) if candidates else None
        frontier_ids = self._quality_frontier_ids(candidates)
        prior = self._load_progressive_state()
        budget = str(prior.get("current_budget") or "high")
        state = {
            "current_budget": budget,
            "next_budget": str(prior.get("next_budget") or budget),
            "stagnation_count": stagnation,
            "best_passrate": overall_best.passrate if overall_best is not None else 0.0,
            "best_average_score": (
                overall_best.average_score if overall_best is not None else 0.0
            ),
            "best_candidate_id": (
                overall_best.candidate_id if overall_best is not None else None
            ),
            "frontier_candidate_ids": sorted(frontier_ids),
            "last_improved_iteration": last_improved,
        }
        self.progressive_state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _iteration_dir(self, iteration: int) -> Path:
        return self.run_dir / "proposer_calls" / f"iter_{iteration:03d}"

    def _workspace_dir(self, iteration: int) -> Path:
        return self._iteration_dir(iteration) / "workspace"

    def _progressive_budget_for_iteration(self, iteration: int) -> str:
        if iteration <= self.config.progressive_initial_low_iterations:
            return "low"
        state = self._load_progressive_state()
        budget = str(state.get("next_budget") or state.get("current_budget") or "low")
        if budget not in {"low", "medium", "high"}:
            return "low"
        return budget

    def _load_progressive_state(self) -> dict[str, Any]:
        if not self.progressive_state_path.exists():
            return {
                "current_budget": "low",
                "next_budget": "low",
                "stagnation_count": 0,
                "best_passrate": 0.0,
                "best_candidate_id": None,
                "last_improved_iteration": 0,
            }
        try:
            payload = json.loads(self.progressive_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _update_progressive_state(
        self,
        *,
        iteration: int,
        budget: str,
        previous_best_passrate: float,
        previous_frontier_ids: set[str] | None = None,
        candidates: list[CandidateResult],
        evaluated: list[CandidateResult],
    ) -> bool:
        """Update ``progressive_state.json`` and return whether the
        iteration advanced (``improved=True`` resets stagnation_count;
        else stagnation_count increments).

        Advance signal:
        - ``previous_frontier_ids`` provided → frontier-id diff
          (curai / curaii / progressive use this — joining top-K
          frontier counts as advance).
        - ``previous_frontier_ids`` is ``None`` → strict passrate gate
          (pareto uses this — best passrate must strictly increase).
        """

        best = max(candidates, key=_candidate_score) if candidates else None
        best_passrate = best.passrate if best is not None else 0.0
        frontier_ids = self._quality_frontier_ids(candidates)
        if previous_frontier_ids is None:
            improved = bool(evaluated and best_passrate > previous_best_passrate)
        else:
            evaluated_ids = {item.candidate_id for item in evaluated}
            improved = bool(evaluated_ids & (frontier_ids - previous_frontier_ids))
        prior = self._load_progressive_state()
        stagnation = 0 if improved else int(prior.get("stagnation_count") or 0) + 1
        if iteration < self.config.progressive_initial_low_iterations:
            next_budget = "low"
        elif improved:
            next_budget = "low"
        elif budget == "low":
            next_budget = "medium"
        elif budget == "medium":
            next_budget = "high"
        else:
            next_budget = "high"
        state = {
            "current_budget": budget,
            "next_budget": next_budget,
            "stagnation_count": stagnation,
            "best_passrate": best_passrate,
            "best_average_score": best.average_score if best is not None else 0.0,
            "best_candidate_id": best.candidate_id if best is not None else None,
            "frontier_candidate_ids": sorted(frontier_ids),
            "last_improved_iteration": (
                iteration if improved else int(prior.get("last_improved_iteration") or 0)
            ),
        }
        self.progressive_state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return improved

    def _load_bandit_state(self) -> dict[str, Any]:
        if not self.bandit_state_path.exists():
            return {
                "total_iters": 0,
                "success_iters": 0,
                "global_reward_sum": 0.0,
                "files": {},
                "last_policy": {},
            }
        try:
            payload = json.loads(self.bandit_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "total_iters": 0,
                "success_iters": 0,
                "global_reward_sum": 0.0,
                "files": {},
                "last_policy": {},
            }
        return payload if isinstance(payload, dict) else {}

    def _bandit_policy_for_workspace(
        self,
        *,
        iteration: int,
        candidates: list[CandidateResult],
        force_budget: str | None = None,
    ) -> dict[str, Any]:
        state = self._load_bandit_state()
        files = state.get("files") if isinstance(state.get("files"), dict) else {}
        required = self._bandit_required_paths_set()
        scored = sorted(
            (
                (str(path), float(info.get("policy_score")))
                for path, info in files.items()
                if isinstance(info, dict)
                and info.get("policy_score") is not None
                and str(path) not in required
            ),
            key=lambda item: (-item[1], item[0]),
        )
        core_files = self._bandit_core_files()
        hot = _dedupe_list(core_files + [path for path, _ in scored[:8]])
        warm = _dedupe_list([path for path, _ in scored[8:20]])

        last_improvement = self._bandit_last_improvement_iteration(candidates) or 0
        stagnation = max(0, iteration - last_improvement - 1)
        stagnated = stagnation >= self.config.bandit_stagnation_threshold

        if force_budget == "low":
            budget = "low"
            reference_iterations: tuple[int, ...] = self._bandit_reference_iterations(
                iteration=iteration,
                candidates=candidates,
                state=state,
                budget=budget,
            )
        elif force_budget == "medium":
            budget = "medium"
            reference_iterations = self._bandit_reference_iterations(
                iteration=iteration,
                candidates=candidates,
                state=state,
                budget=budget,
            )
        elif force_budget == "high":
            budget = "high"
            reference_iterations = self._bandit_reference_iterations(
                iteration=iteration,
                candidates=candidates,
                state=state,
                budget=budget,
            )
        elif iteration <= 1 or not files:
            budget = "low"
            reference_iterations = ()
        elif stagnated:
            budget = "high"
            reference_iterations = self._bandit_reference_iterations(
                iteration=iteration,
                candidates=candidates,
                state=state,
                budget=budget,
            )
        else:
            budget = "medium"
            reference_iterations = self._bandit_reference_iterations(
                iteration=iteration,
                candidates=candidates,
                state=state,
                budget=budget,
            )

        best_iter_count = 1 if budget == "low" else 3
        best_iters = [
            it for it in self._best_iterations(candidates, k=best_iter_count)
            if it in set(reference_iterations)
        ]
        read_budget = {
            path: (800 if path in hot else 300)
            for path in _dedupe_list(hot + warm)
        }
        policy = {
            "budget": budget,
            "reference_iterations": list(reference_iterations),
            "hot_files": hot,
            "warm_files": warm,
            "cold_files": [],
            "best_iterations": best_iters,
            "read_budget_lines_by_path": read_budget,
            "policy_score_snapshot": {path: score for path, score in scored[:20]},
        }
        state["last_policy"] = policy
        self.bandit_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.bandit_state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return policy

    def _bandit_required_paths_set(self) -> set[str]:
        required: set[str] = set(self._bandit_core_files())
        # Cumulative summaries the proposer should consult freely; sizing handles
        # whether re-reading is worthwhile, not the bandit ranking.
        for rel in (
            "summaries/evolution_summary.jsonl",
            "summaries/best_candidates.json",
            "summaries/iteration_index.json",
            "pending_eval.json",
        ):
            required.add(rel)
        return required

    def _bandit_reference_iterations(
        self,
        *,
        iteration: int,
        candidates: list[CandidateResult],
        state: dict[str, Any],
        budget: str,
    ) -> tuple[int, ...]:
        # bandit v4: align ref-iter selection with progressive's best-k structure,
        # then prepend hot_iters reverse-resolved from the previous iteration's
        # hot/warm file paths. cap=3 for low, cap=5 for medium; high keeps the
        # full-history behaviour.
        available = {
            item
            for item in self._candidate_iterations(candidates)
            if 0 < item < iteration and self._iteration_dir(item).exists()
        }
        if budget == "high":
            return tuple(sorted(available))

        last_policy = (
            state.get("last_policy")
            if isinstance(state.get("last_policy"), dict)
            else {}
        ) or {}
        hot_iters = self._iters_from_policy_paths(
            list(last_policy.get("hot_files") or [])
            + list(last_policy.get("warm_files") or [])
        )

        base: list[int] = []
        base.extend(self._best_iterations(candidates, k=1 if budget == "low" else 3))

        cap = 3 if budget == "low" else 5
        out: list[int] = []
        seen: set[int] = set()
        for item in list(hot_iters) + base:
            if item in available and item not in seen:
                out.append(item)
                seen.add(item)
            if len(out) >= cap:
                break
        return tuple(out)

    def _recent_reference_iterations(self, available: set[int]) -> tuple[int, ...]:
        return tuple(sorted(available)[-3:])

    def _random_reference_iterations(
        self,
        available: set[int],
        *,
        iteration: int,
    ) -> tuple[int, ...]:
        ordered = sorted(available)
        if len(ordered) <= 3:
            return tuple(ordered)
        seed_material = f"{self.config.run_id}:{iteration}:random".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        return tuple(sorted(random.Random(seed).sample(ordered, k=3)))

    def _best_reference_iterations(
        self,
        available: set[int],
        candidates: list[CandidateResult],
    ) -> tuple[int, ...]:
        out: list[int] = []
        seen: set[int] = set()
        for candidate in sorted(candidates, key=_candidate_best_rank):
            iteration = _candidate_iteration(candidate.candidate_id)
            if iteration is None or iteration not in available or iteration in seen:
                continue
            seen.add(iteration)
            out.append(iteration)
            if len(out) >= 3:
                break
        return tuple(sorted(out))

    def _update_bandit_state(
        self,
        *,
        iteration: int,
        previous_best_passrate: float,
        previous_best_quality: float | None = None,
        evaluated: list[CandidateResult],
        call_dir: Path,
    ) -> None:
        state = self._load_bandit_state()
        files = state.setdefault("files", {})
        if not isinstance(files, dict):
            files = {}
            state["files"] = files

        tool_access = self._load_json_file(call_dir / "agent" / "tool_access.json")
        if not isinstance(tool_access, dict):
            tool_access = self._latest_proposer_tool_access(iteration)
        read_paths = self._bandit_read_paths(tool_access)
        written_paths = self._bandit_written_paths(tool_access)
        changed_paths = sorted({
            self._bandit_normalize_access_path(path)
            for path in self._changed_paths_from_diff(call_dir / "diff.patch")
        })
        best_eval_passrate = max((item.passrate for item in evaluated), default=0.0)
        previous_quality = previous_best_passrate
        reward_value = best_eval_passrate
        history = list(state.get("passrate_history") or [])
        prev_quality = [float(p) for p in history if p is not None]
        recent_quality = prev_quality[-self.config.bandit_reward_window :]
        clip = self.config.bandit_reward_clip
        if not evaluated:
            reward = -clip * 0.25
        elif len(recent_quality) < 2:
            # Pre-window iterations: use scaled improvement vs prior best as a
            # rough surrogate, then transition to z-score once history fills.
            raw = reward_value - previous_quality
            reward = max(-clip, min(clip, raw * 10.0))
        else:
            mu = sum(recent_quality) / len(recent_quality)
            var = sum((p - mu) ** 2 for p in recent_quality) / len(recent_quality)
            sigma = max(self.config.bandit_reward_sigma_floor, var**0.5)
            reward = max(-clip, min(clip, (reward_value - mu) / sigma))
        success = reward > 0.0

        history.append(reward_value if evaluated else None)
        state.pop("quality_history", None)
        state["passrate_history"] = history
        state["total_iters"] = int(state.get("total_iters") or 0) + 1
        if success:
            state["success_iters"] = int(state.get("success_iters") or 0) + 1
        state["global_reward_sum"] = float(state.get("global_reward_sum") or 0.0) + reward

        for path, stats in read_paths.items():
            row = files.setdefault(path, self._empty_bandit_file_state())
            row["read_iters"] = int(row.get("read_iters") or 0) + 1
            row["read_calls"] = int(row.get("read_calls") or 0) + int(stats.get("reads") or 0)
            row["read_lines"] = int(row.get("read_lines") or 0) + int(stats.get("lines") or 0)
            row["reward_sum"] = float(row.get("reward_sum") or 0.0) + reward
            if success:
                row["success_iters"] = int(row.get("success_iters") or 0) + 1
        for path in written_paths:
            row = files.setdefault(path, self._empty_bandit_file_state())
            row["write_iters"] = int(row.get("write_iters") or 0) + 1
        for path in changed_paths:
            row = files.setdefault(path, self._empty_bandit_file_state())
            row["changed_iters"] = int(row.get("changed_iters") or 0) + 1

        self._recompute_bandit_scores(state)
        self.bandit_state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _recompute_bandit_scores(self, state: dict[str, Any]) -> None:
        files = state.get("files") if isinstance(state.get("files"), dict) else {}
        required = self._bandit_required_paths_set()
        total_iters = max(1, int(state.get("total_iters") or 0))

        # Global stats are computed only from "scored" reads — reads of files
        # that the bandit actually ranks. Required (always-include) files and
        # phantom entries (reads=0, only seen via diff/write) are excluded so
        # the prior on per-file utility reflects the discretionary read pool.
        scored_read_iters = 0
        scored_success_iters = 0
        scored_reward_sum = 0.0
        for path, info in files.items():
            if not isinstance(info, dict):
                continue
            read_iters = int(info.get("read_iters") or 0)
            if read_iters == 0 or path in required:
                continue
            scored_read_iters += read_iters
            scored_success_iters += int(info.get("success_iters") or 0)
            scored_reward_sum += float(info.get("reward_sum") or 0.0)

        p_global = (
            scored_success_iters / scored_read_iters if scored_read_iters > 0 else 0.0
        )
        mean_reward_global = (
            scored_reward_sum / scored_read_iters if scored_read_iters > 0 else 0.0
        )
        alpha = self.config.bandit_prior_alpha
        for path, info in files.items():
            if not isinstance(info, dict):
                continue
            read_iters = int(info.get("read_iters") or 0)
            if read_iters == 0 or path in required:
                # Required files are always included via _bandit_core_files();
                # phantom entries (changed/written but never read) carry no
                # signal worth ranking against discretionary reads.
                info["utility"] = 0.0
                info["policy_score"] = None
                info.setdefault("cooldown_until", 0)
                continue
            p_file = (
                float(info.get("success_iters") or 0) + alpha * p_global
            ) / (read_iters + alpha)
            mean_reward = (
                float(info.get("reward_sum") or 0.0)
                + alpha * self.config.bandit_prior_weight * mean_reward_global
            ) / (read_iters + alpha)
            avg_lines = float(info.get("read_lines") or 0) / max(1, read_iters)
            cost = self.config.bandit_cost_lambda * math.log1p(
                avg_lines / max(1, self.config.bandit_line_scale)
            )
            bonus = self.config.bandit_exploration_c * math.sqrt(
                math.log(total_iters + 1) / (read_iters + 1)
            )
            binary_utility = p_file - p_global
            reward_utility = mean_reward - mean_reward_global
            score = 0.7 * binary_utility + 0.3 * reward_utility - cost + bonus
            info["utility"] = 0.7 * binary_utility + 0.3 * reward_utility
            info["policy_score"] = score
            info.setdefault("cooldown_until", 0)

    def _empty_bandit_file_state(self) -> dict[str, Any]:
        return {
            "read_iters": 0,
            "success_iters": 0,
            "reward_sum": 0.0,
            "read_calls": 0,
            "read_lines": 0,
            "write_iters": 0,
            "changed_iters": 0,
            "utility": 0.0,
            "policy_score": 0.0,
            "cooldown_until": 0,
        }

    def _bandit_core_files(self) -> list[str]:
        if not self.config.bandit_min_core_files:
            return []
        source_files = set(self.workspace_spec.source_files)
        target_files = [self.workspace_spec.primary_source_file]
        scaffold_path = self._source_scaffold_path(self.config.progressive_target_system)
        if scaffold_path is not None:
            rel = f"scaffolds/{scaffold_path.name}"
            if rel in source_files:
                target_files.append(rel)
        target_files.extend(
            rel
            for rel in ("scaffolds/base.py", "model.py", "schemas.py")
            if rel in source_files
        )
        project_paths = [
            f"source_snapshot/candidate/project_source/src/optimizer1/{rel}"
            for rel in _dedupe_list(target_files)
        ]
        return _dedupe_list(
            project_paths
            + [
                "summaries/candidate_score_table.json",
                "summaries/retrieval_diagnostics_summary.json",
                "summaries/diff_summary.jsonl",
                "summaries/evolution_summary.jsonl",
                "summaries/best_candidates.json",
                "summaries/iteration_index.json",
                "pending_eval.json",
            ]
        )

    def _bandit_read_paths(self, tool_access: dict[str, Any]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        raw = tool_access.get("files_read") if isinstance(tool_access, dict) else {}
        if not isinstance(raw, dict):
            return out
        for path, stats in raw.items():
            normalized = self._bandit_normalize_access_path(str(path))
            if not self._bandit_is_trackable_path(normalized):
                continue
            info = stats if isinstance(stats, dict) else {}
            out[normalized] = {
                "reads": max(1, int(info.get("reads") or 1)),
                "lines": int(info.get("lines") or 0),
            }
        return out

    def _bandit_written_paths(self, tool_access: dict[str, Any]) -> list[str]:
        raw = tool_access.get("files_written") if isinstance(tool_access, dict) else {}
        if not isinstance(raw, dict):
            return []
        return sorted(
            {
                normalized
                for path in raw
                if self._bandit_is_trackable_path(
                    normalized := self._bandit_normalize_access_path(str(path))
                )
            }
        )

    def _bandit_normalize_access_path(self, raw_path: str) -> str:
        text = raw_path.replace("\\", "/")
        marker = "/workspace/"
        if marker in text:
            text = text.split(marker, 1)[1]
        for marker in ("/source_snapshot/", "/summaries/", "/reference_iterations/", "/generated/"):
            if marker in text:
                return marker.strip("/") + "/" + text.split(marker, 1)[1].lstrip("/")
        return text.lstrip("./")

    def _bandit_is_trackable_path(self, path: str) -> bool:
        if not path or "\n" in path or "\t" in path:
            return False
        if any(token in path for token in ("|", "[]", "(", ")", "@")):
            return False
        return (
            path == "pending_eval.json"
            or path in {"assignment.json", "workspace_manifest.json", "access_policy.json"}
            or path.startswith("summaries/")
            or path.startswith("reference_iterations/")
            or path.startswith("source_snapshot/")
            or path.startswith("generated/")
        )

    def _changed_paths_from_diff(self, diff_path: Path) -> list[str]:
        if not diff_path.exists():
            return []
        out: list[str] = []
        for line in diff_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("diff --git "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                out.append(parts[3].removeprefix("b/"))
        return sorted(set(out))

    def _load_json_file(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _latest_proposer_tool_access(self, iteration: int) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {}
        for raw in reversed(self.summary_path.read_text(encoding="utf-8").splitlines()):
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(item, dict)
                and item.get("event") == "proposer_result"
                and int(item.get("iteration") or 0) == iteration
            ):
                return {
                    "files_read": item.get("files_read", {}),
                    "files_written": item.get("files_written", {}),
                }
        return {}

    def _bandit_last_improvement_iteration(
        self,
        candidates: list[CandidateResult],
    ) -> int | None:
        best = 0.0
        last: int | None = None
        for candidate in sorted(
            candidates,
            key=lambda item: ((_candidate_iteration(item.candidate_id) or 0), item.candidate_id),
        ):
            iteration = _candidate_iteration(candidate.candidate_id) or 0
            if iteration <= 0:
                continue
            if candidate.passrate > best:
                best = candidate.passrate
                last = iteration
        return last

    _BANDIT_REF_ITER_RE = re.compile(r"reference_iterations/iter_(\d+)/")

    def _iters_from_policy_paths(self, paths: list[str]) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for path in paths:
            match = self._BANDIT_REF_ITER_RE.search(str(path))
            if not match:
                continue
            n = int(match.group(1))
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
        return out

    def _seed_passrate(self, candidates: list[CandidateResult]) -> float:
        """Return the seed (iter-0) candidate's passrate, or 0.0 if none.

        Seed candidates are those without an ``iterNNN_`` prefix in
        ``candidate_id``.  When the user supplies ``--baseline-dir`` the
        baseline candidates are also counted as seeds.  The maximum is
        used so repeated MemGPT seed variants collapse to
        a single baseline floor.
        """

        seeds = [
            item
            for item in candidates
            if _candidate_iteration(item.candidate_id) is None
        ]
        return max((item.passrate for item in seeds), default=0.0)

    def _curaii_select_for_budget(
        self,
        existing_candidates: list[CandidateResult],
        *,
        iteration: int,
        budget: str,
        baseline_passrate: float,
    ) -> tuple[int | None, tuple[int, ...] | None]:
        """Budget-coupled curaii selection with a baseline-gated parent pool.

        Returns ``(base_iter, refs_override)``:

        * ``base_iter`` is the iteration whose archived candidate source
          replaces the clean baseline in ``project_source/``.  ``None``
          signals the empty-pool fallback — this iteration runs as if
          policy were ``curai`` (clean baseline, default refs).
        * ``refs_override`` is an explicit reference iteration tuple.
          ``None`` means "fall through to ``_reference_iterations_for_budget``".

        Eligibility: iter strictly less than the current iter, the
        candidate's archived ``project_source/`` directory is on disk,
        AND ``passrate > baseline_passrate`` (strict gate so tied parents
        — the locomo failure mode — are excluded).

        Budget mapping:

        * ``low``    — base = best1 in the pool; refs = ``(base_iter,)``.
        * ``medium`` — base = uniform random ∈ top-3 of pool; refs = the
          (≤3) pool itself (sorted by iteration), so the proposer sees
          the chosen base and its closest siblings.
        * ``high``   — base = uniform random ∈ top-3 of pool; refs =
          ``None`` (fall through to "all available").
        """

        eligible: list[tuple[int, CandidateResult]] = []
        for item in existing_candidates:
            base_iter = _candidate_iteration(item.candidate_id)
            if base_iter is None or base_iter <= 0 or base_iter >= iteration:
                continue
            if item.passrate <= baseline_passrate:
                continue
            base_source = (
                self._iteration_dir(base_iter)
                / "source_snapshot"
                / "candidate"
                / "project_source"
            )
            if not base_source.exists():
                continue
            eligible.append((base_iter, item))

        if not eligible:
            return None, None

        eligible.sort(key=lambda pair: _candidate_score(pair[1]), reverse=True)

        if budget == "low":
            best_iter = eligible[0][0]
            return best_iter, (best_iter,)

        pool = eligible[: min(3, len(eligible))]
        chosen = random.choice(pool)[0]

        if budget == "medium":
            refs = tuple(sorted(it for it, _ in pool))
            return chosen, refs

        # high — refs fall through to "all available" via the standard
        # _reference_iterations_for_budget path.
        return chosen, None

    def _pareto_select_base(
        self,
        existing_candidates: list[CandidateResult],
        *,
        iteration: int,
        baseline_passrate: float,
    ) -> int | None:
        """Sample a patch base for the ``pareto`` selection policy.

        Returns the iteration index of a candidate drawn uniformly at
        random from the current passrate × token_consuming Pareto
        frontier, restricted to candidates whose archived
        ``project_source/`` is on disk and whose passrate strictly
        exceeds the seed baseline (so a tied/worse parent is never
        chosen). Returns ``None`` when no eligible frontier candidate
        exists; the caller then falls back to the clean snapshot, i.e.
        the same behavior as the ``default`` policy for that iter.

        Reference iterations always fall through to "all available"
        (``refs_override=None``) — pareto's only divergence from
        ``default`` is the patch base, not the iter context budget.
        """

        frontier = self._quality_frontier(existing_candidates)
        if not frontier:
            return None
        eligible: list[int] = []
        for item in frontier:
            base_iter = _candidate_iteration(item.candidate_id)
            if base_iter is None or base_iter <= 0 or base_iter >= iteration:
                continue
            if item.passrate <= baseline_passrate:
                continue
            base_source = (
                self._iteration_dir(base_iter)
                / "source_snapshot"
                / "candidate"
                / "project_source"
            )
            if not base_source.exists():
                continue
            eligible.append(base_iter)
        if not eligible:
            return None
        return random.choice(eligible)

    def _state_snapshot_base_iteration(
        self,
        existing_candidates: list[CandidateResult],
        *,
        iteration: int,
    ) -> int | None:
        """Choose the comparison base rendered in organized ``state.md``.

        Default-policy proposer iterations still edit a clean source snapshot;
        this base is only the state/evidence anchor. Prefer the current quality
        frontier's strongest evaluated iteration, and fall back to the seed
        baseline (iteration 0) when no positive iteration exists yet.
        """

        if not existing_candidates:
            return None
        candidates = self._quality_frontier(existing_candidates) or existing_candidates
        sorted_candidates = sorted(candidates, key=_candidate_score, reverse=True)
        has_seed = False
        for candidate in sorted_candidates:
            candidate_iter = _candidate_iteration(candidate.candidate_id)
            if candidate_iter is None:
                has_seed = True
                continue
            if 0 <= candidate_iter < iteration:
                return candidate_iter
        if has_seed:
            return 0
        return None

    def _reference_iterations_for_budget(
        self,
        budget: str,
        *,
        iteration: int,
        candidates: list[CandidateResult],
    ) -> tuple[int, ...]:
        available = {
            item
            for item in self._candidate_iterations(candidates)
            if 0 < item < iteration and self._iteration_dir(item).exists()
        }
        if self.config.selection_policy == "recent":
            return self._recent_reference_iterations(available)
        if self.config.selection_policy == "random":
            return self._random_reference_iterations(available, iteration=iteration)
        if self.config.selection_policy == "best":
            return self._best_reference_iterations(available, candidates)
        if budget == "high":
            return tuple(sorted(available))

        if budget == "medium":
            best_k = self.config.progressive_medium_best_count
        else:
            best_k = self.config.progressive_low_best_count
        selected = self._best_iterations(candidates, k=best_k)
        out: list[int] = []
        seen: set[int] = set()
        for item in selected:
            if item not in available or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return tuple(out)

    def _best_iterations(self, candidates: list[CandidateResult], *, k: int) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for candidate in sorted(candidates, key=_candidate_best_rank):
            iteration = _candidate_iteration(candidate.candidate_id)
            if iteration is None or iteration <= 0 or iteration in seen:
                continue
            seen.add(iteration)
            out.append(iteration)
            if len(out) >= k:
                break
        return out

    def _candidate_iterations(self, candidates: list[CandidateResult]) -> set[int]:
        out: set[int] = set()
        for candidate in candidates:
            iteration = _candidate_iteration(candidate.candidate_id)
            if iteration is not None:
                out.add(iteration)
        return out

    def _best_passrate(self, candidates: list[CandidateResult]) -> float:
        return max((item.passrate for item in candidates), default=0.0)

    def _refresh_run_indexes(self, candidates: list[CandidateResult]) -> None:
        self._write_candidate_score_table_from_candidates(candidates)
        if not self.retrieval_diagnostics_summary_path.exists():
            self.retrieval_diagnostics_summary_path.write_text("[]\n", encoding="utf-8")
        self._write_iteration_index(candidates)
        if not self.diff_summary_path.exists():
            self.diff_summary_path.write_text("", encoding="utf-8")

    def _write_candidate_score_table_from_candidates(
        self,
        candidates: list[CandidateResult],
    ) -> None:
        rows = []
        frontier_ids = self._quality_frontier_ids(candidates)
        best_passrate_ids = self._best_passrate_ids(candidates)
        for candidate in sorted(
            candidates,
            key=lambda item: ((_candidate_iteration(item.candidate_id) or 0), item.candidate_id),
        ):
            extra = self._candidate_extra(candidate)
            iteration = _candidate_iteration(candidate.candidate_id) or 0
            rows.append(
                {
                    "iteration": iteration,
                    "candidate_id": candidate.candidate_id,
                    "scaffold_name": candidate.scaffold_name,
                    "passrate": candidate.passrate,
                    "average_score": candidate.average_score,
                    "token_consuming": candidate.token_consuming,
                    "source_family": extra.get("source_family"),
                    "build_tag": extra.get("build_tag"),
                    "result_path": candidate.result_path,
                    "iteration_dir": str(self._iteration_dir(iteration)),
                    "is_best_passrate": candidate.candidate_id in best_passrate_ids,
                    "is_quality_frontier": candidate.candidate_id in frontier_ids,
                    # Audit-only flag from swebench eval-gate integrity check.
                    # Always present so dashboards can filter on it; non-SWE
                    # runs leave it False because their evaluators don't set
                    # this key on candidate.config.
                    "reward_hack_attempt": bool(
                        (candidate.config or {}).get("reward_hack_attempt", False)
                    ),
                }
            )
        self.candidate_score_table_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_iteration_index(self, candidates: list[CandidateResult]) -> None:
        by_iteration: dict[int, dict[str, Any]] = {}
        for candidate in candidates:
            iteration = _candidate_iteration(candidate.candidate_id) or 0
            call_dir = self._iteration_dir(iteration)
            row = by_iteration.setdefault(
                iteration,
                {
                    "iteration": iteration,
                    "iteration_dir": str(call_dir),
                    "candidate_ids": [],
                    "candidate_result_paths": [],
                    "compact_result_path": str(call_dir / "eval" / "candidate_result.compact.json"),
                    "retrieval_diagnostics_path": str(
                        call_dir / "eval" / "retrieval_diagnostics.json"
                    ),
                    "diff_path": str(call_dir / "diff.patch"),
                    "diff_digest_path": str(call_dir / "diff_digest.md"),
                    "source_snapshot_dir": str(call_dir / "source_snapshot"),
                    "generated_dir": str(call_dir / "generated"),
                },
            )
            row["candidate_ids"].append(candidate.candidate_id)
            row["candidate_result_paths"].append(candidate.result_path)

        for call_dir in sorted((self.run_dir / "proposer_calls").glob("iter_*")):
            iteration = _iteration_from_dir_name(call_dir.name)
            if iteration is None:
                continue
            by_iteration.setdefault(
                iteration,
                {
                    "iteration": iteration,
                    "iteration_dir": str(call_dir),
                    "candidate_ids": [],
                    "candidate_result_paths": [],
                    "compact_result_path": str(call_dir / "eval" / "candidate_result.compact.json"),
                    "retrieval_diagnostics_path": str(
                        call_dir / "eval" / "retrieval_diagnostics.json"
                    ),
                    "diff_path": str(call_dir / "diff.patch"),
                    "diff_digest_path": str(call_dir / "diff_digest.md"),
                    "source_snapshot_dir": str(call_dir / "source_snapshot"),
                    "generated_dir": str(call_dir / "generated"),
                },
            )

        rows = [by_iteration[key] for key in sorted(by_iteration)]
        self.iteration_index_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _optimization_direction_lines(self, target_system: str | None = None) -> tuple[str, ...]:
        """Return prompt-only optimization directions for proposers."""

        lines: list[str] = []
        target = target_system or self.config.progressive_target_system
        for cell in get_target_cells(target):
            focus = ", ".join(cell.focus_functions) if cell.focus_functions else "all functions"
            lines.append(
                f"{cell.name}: {cell.description} Focus areas: {focus}. "
                f"Guidance: {cell.prompt_guidance}"
            )
        if not lines:
            lines.append(
                "global: improve memory construction, retrieval, evidence selection, "
                "and answering."
            )
        return tuple(lines)

    def _candidate_extra(self, candidate: CandidateResult) -> dict[str, Any]:
        extra = candidate.config.get("extra") if isinstance(candidate.config, dict) else None
        if isinstance(extra, dict):
            return dict(extra)
        return {}

    def _infer_source_family(self, candidate: CandidateResult) -> str:
        extra = candidate.config.get("extra") if isinstance(candidate.config, dict) else None
        if isinstance(extra, dict):
            source_family = str(extra.get("source_family") or "").lower()
            if source_family == "memgpt":
                return source_family
        text = f"{candidate.candidate_id} {candidate.scaffold_name}".lower()
        if "memgpt" in text or "letta" in text:
            return "memgpt"
        return "memgpt"

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
        generated_dir = generated_dir or self.generated_dir
        snapshot_root = snapshot_root or (
            generated_dir / "source_snapshots" / f"iter_{iteration:03d}"
        )
        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(snapshot_root, root=snapshot_root)

        scaffold_source = self._source_scaffold_path(source_family)
        source_files = [path for path in (scaffold_source,) if path is not None and path.exists()]

        candidate_dir = snapshot_root / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(candidate_dir, root=snapshot_root)
        for path in source_files:
            self._copy_if_exists(path, candidate_dir / path.name)
        self._copy_project_source_context(candidate_dir)
        # Always seed `original_project_source/` directly from the baseline
        # so the on-disk diff captures cumulative evolution from baseline
        # regardless of what `project_source/` is replaced with below.
        original_project_source = candidate_dir / "original_project_source"
        if original_project_source.exists():
            shutil.rmtree(original_project_source)
        self._copy_baseline_project_source_into(original_project_source)
        project_source = candidate_dir / "project_source"
        # If a curaii-style parent base was supplied, replace the freshly
        # baseline-seeded `project_source/` with that parent's archived
        # source so the proposer edits on top of a previously evaluated
        # candidate rather than restarting from baseline.
        if base_iter is not None:
            parent_source = (
                self._iteration_dir(base_iter)
                / "source_snapshot"
                / "candidate"
                / "project_source"
            )
            if parent_source.exists():
                if project_source.exists():
                    shutil.rmtree(project_source)
                shutil.copytree(
                    parent_source,
                    project_source,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
        self._copy_upstream_source_context(source_family, candidate_dir)
        self._copy_if_exists(
            self.project_root / "src" / "optimizer1" / "scaffolds" / "base.py",
            candidate_dir / "base.py",
        )
        self._copy_if_exists(
            self.project_root / "src" / "optimizer1" / "schemas.py",
            candidate_dir / "schemas.py",
        )
        if base_iter is not None:
            base_anchor_lines = [
                f"Patch base: iter_{base_iter:03d} (curaii lineage anchor).",
                "`project_source/` is initialized from that iteration's archived",
                "candidate source. `original_project_source/` remains the clean",
                "baseline so `diff.patch` records cumulative evolution from baseline.",
            ]
        else:
            base_anchor_lines = [
                "Historical iterations are diagnostic references only; do not treat",
                "their source snapshots as editable parents.",
            ]
        (candidate_dir / "SNAPSHOT.md").write_text(
            "\n".join(
                [
                    "# Source Snapshot Candidate",
                    "",
                    f"Iteration: {iteration}",
                    f"Source family: {source_family}",
                    f"Target system: {target_system or source_family}",
                    "",
                    "This directory is a writable candidate-specific clean source snapshot.",
                    "It also contains benchmark-scoped project source under",
                    "`project_source/src/optimizer1` and relevant upstream source",
                    "under `upstream_source` for inspection.",
                    *base_anchor_lines,
                    "Existing source-backed base memories are read-only. You may edit",
                    "copied build/database-construction paths such as add/build/schema/",
                    "extraction/evolution/embedding or persistence layout, but source",
                    "edits that alter persisted memories must use a fresh source_base_dir",
                    "and build_tag in pending_eval.json.",
                    "Modify files here for the mechanism under test, then expose the",
                    "edited built-in source scaffold in `pending_eval.json` with",
                    "`scaffold_name` and `extra.source_project_path`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        manifest = {
            "iteration": iteration,
            "source_family": source_family,
            "target_system": target_system or source_family,
            "benchmark": self.workspace_spec.benchmark,
            "candidate_dir": str(candidate_dir),
            "project_source": str(project_source),
            "original_project_source": str(original_project_source),
            "primary_source_file": self.workspace_spec.primary_source_file,
            "project_source_files": list(self.workspace_spec.source_files),
            "source_files": [str(path) for path in source_files],
            "base_iter": base_iter,
        }
        (snapshot_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (call_dir / "source_snapshot_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return snapshot_root

    def _ensure_package_dirs(self, path: Path, *, root: Path | None = None) -> None:
        generated_root = root or self.generated_dir
        current = generated_root
        while True:
            current.mkdir(parents=True, exist_ok=True)
            init = current / "__init__.py"
            if not init.exists():
                init.write_text('"""Generated source snapshot package."""\n', encoding="utf-8")
            if current == path:
                break
            try:
                rel = path.relative_to(current)
            except ValueError:
                break
            parts = rel.parts
            if not parts:
                break
            current = current / parts[0]

    def _copy_iteration_bundle(self, src: Path, dest: Path) -> None:
        if not src.exists():
            return
        if dest.exists():
            shutil.rmtree(dest)
        ignore = shutil.ignore_patterns(
            "workspace",
            "context",
            "claude_session",
            "__pycache__",
            "*.pyc",
        )
        shutil.copytree(src, dest, ignore=ignore)

    def _copy_if_exists(self, src: Path, dest: Path) -> None:
        if src.exists() and src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    def _sync_workspace_outputs(
        self,
        *,
        workspace_dir: Path,
        call_dir: Path,
    ) -> None:
        workspace_generated_dir = workspace_dir / "generated"
        if workspace_generated_dir.exists():
            self.generated_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_package_dirs(self.generated_dir)
            call_generated_dir = call_dir / "generated"
            if call_generated_dir.exists():
                shutil.rmtree(call_generated_dir)
            call_generated_dir.mkdir(parents=True, exist_ok=True)
            for src in sorted(workspace_generated_dir.rglob("*")):
                if not src.is_file():
                    continue
                if "__pycache__" in src.parts or src.suffix == ".pyc":
                    continue
                rel = src.relative_to(workspace_generated_dir)
                dest = self.generated_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                call_dest = call_generated_dir / rel
                call_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, call_dest)

        workspace_pending = workspace_dir / "pending_eval.json"
        if workspace_pending.exists():
            self._copy_if_exists(workspace_pending, self.pending_eval_path)
            self._copy_if_exists(workspace_pending, call_dir / "pending_eval.raw.json")

    def _normalize_workspace_candidate_paths(
        self,
        candidate: dict[str, Any],
        *,
        workspace_dir: Path,
        workspace_generated_dir: Path,
    ) -> None:
        candidate["candidate_root"] = str(self.generated_dir)
        if candidate.get("generated_dir"):
            candidate["generated_dir"] = str(self.generated_dir)
        self._normalize_workspace_path_fields(candidate, workspace_dir, workspace_generated_dir)
        extra = candidate.get("extra")
        if isinstance(extra, dict):
            self._normalize_workspace_path_fields(extra, workspace_dir, workspace_generated_dir)

    def _normalize_workspace_path_fields(
        self,
        payload: dict[str, Any],
        workspace_dir: Path,
        workspace_generated_dir: Path,
    ) -> None:
        for key in (
            "module_path",
            "source_path",
            "source_project_path",
            "project_source_path",
            "memomemo_source_path",
            "upstream_source_path",
            "memgpt_source_path",
            "mini_swe_agent_source_path",
            "source_base_dir",
            "base_memory_dir",
        ):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            payload[key] = str(
                self._resolve_workspace_path(
                    value,
                    workspace_dir=workspace_dir,
                    workspace_generated_dir=workspace_generated_dir,
                )
            )

    def _rewrite_workspace_source_paths_to_archive(
        self,
        candidate: dict[str, Any],
        *,
        workspace_dir: Path,
        archived_source_snapshot: Path,
    ) -> None:
        self._rewrite_workspace_source_path_fields(
            candidate,
            workspace_dir=workspace_dir,
            archived_source_snapshot=archived_source_snapshot,
        )
        extra = candidate.get("extra")
        if isinstance(extra, dict):
            self._rewrite_workspace_source_path_fields(
                extra,
                workspace_dir=workspace_dir,
                archived_source_snapshot=archived_source_snapshot,
            )

    def _rewrite_workspace_source_path_fields(
        self,
        payload: dict[str, Any],
        *,
        workspace_dir: Path,
        archived_source_snapshot: Path,
    ) -> None:
        workspace_source = (workspace_dir / "source_snapshot").resolve(strict=False)
        archived_source = archived_source_snapshot.resolve(strict=False)
        for key in (
            "source_snapshot_path",
            "source_project_path",
            "project_source_path",
            "memomemo_source_path",
            "upstream_source_path",
            "memgpt_source_path",
            "mini_swe_agent_source_path",
        ):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = workspace_dir / path
            path = self._map_container_workspace_path(path, workspace_dir=workspace_dir)
            resolved = path.resolve(strict=False)
            if resolved == workspace_source or workspace_source in resolved.parents:
                rel = resolved.relative_to(workspace_source)
                payload[key] = str((archived_source / rel).resolve(strict=False))

    def _resolve_workspace_path(
        self,
        value: str,
        *,
        workspace_dir: Path,
        workspace_generated_dir: Path,
    ) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            path = self._map_container_workspace_path(path, workspace_dir=workspace_dir)
        if not path.is_absolute():
            path = workspace_dir / path
        resolved = path.resolve(strict=False)
        workspace_generated = workspace_generated_dir.resolve(strict=False)
        if resolved == workspace_generated or workspace_generated in resolved.parents:
            rel = resolved.relative_to(workspace_generated)
            return (self.generated_dir / rel).resolve(strict=False)
        return resolved

    def _map_container_workspace_path(self, path: Path, *, workspace_dir: Path) -> Path:
        if self.config.proposer_sandbox.strip().lower() != "docker":
            return path
        container_root = Path(self.config.proposer_docker_workspace or "/workspace")
        try:
            rel = path.relative_to(container_root)
        except ValueError:
            return path
        return workspace_dir / rel

    def _copy_tree_if_exists(
        self,
        src: Path,
        dest: Path,
        *,
        ignore_names: tuple[str, ...] = (),
    ) -> None:
        if not src.exists() or not src.is_dir():
            return
        if dest.exists():
            shutil.rmtree(dest)
        ignore_patterns = (
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "*.pyc",
            *ignore_names,
        )
        shutil.copytree(
            src,
            dest,
            ignore=shutil.ignore_patterns(*ignore_patterns),
        )

    def _copy_baseline_project_source_into(self, target_root: Path) -> list[str]:
        """Copy the baseline project source into ``target_root/src/optimizer1``.

        Returns the list of files copied (used by the caller to write a
        manifest, when applicable).
        """

        return list(
            copy_benchmark_project_source(
                project_root=self.project_root,
                dest_pkg=target_root / "src" / "optimizer1",
                spec=self.workspace_spec,
            )
        )

    def _copy_project_source_context(self, dest_dir: Path) -> None:
        copied = self._copy_baseline_project_source_into(dest_dir / "project_source")
        (dest_dir / "project_source_manifest.json").write_text(
            json.dumps(
                {
                    "benchmark": self.workspace_spec.benchmark,
                    "primary_source_file": self.workspace_spec.primary_source_file,
                    "source_files": list(copied),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _source_scaffold_path(self, source_family: str) -> Path | None:
        mapping = {
            "memgpt": "memgpt_scaffold.py",
        }
        name = mapping.get(source_family)
        if not name:
            return None
        return self.project_root / "src" / "optimizer1" / "scaffolds" / name

    def _copy_upstream_source_context(self, source_family: str, dest_dir: Path) -> None:
        upstream_dir = dest_dir / "upstream_source"
        if source_family == "memgpt":
            self._copy_tree_if_exists(
                self.project_root / "references" / "vendor" / "MemGPT",
                upstream_dir / "MemGPT",
            )

    def _best_passrate_ids(self, candidates: list[CandidateResult]) -> set[str]:
        if not candidates:
            return set()
        best = max(item.passrate for item in candidates)
        return {item.candidate_id for item in candidates if item.passrate == best}

    def _quality_frontier(self, candidates: list[CandidateResult]) -> list[CandidateResult]:
        if not candidates:
            return []
        by_id = {item.candidate_id: item for item in candidates}
        points = [
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
            for item in candidates
        ]
        return [by_id[point.candidate_id] for point in pareto_frontier(points)]

    def _quality_frontier_ids(self, candidates: list[CandidateResult]) -> set[str]:
        return {item.candidate_id for item in self._quality_frontier(candidates)}

    def _best_quality_value(self, candidates: list[CandidateResult]) -> float:
        return max((_candidate_quality_value(item) for item in candidates), default=0.0)

    def _save_best_candidates(self, candidates: list[CandidateResult]) -> None:
        self.frontier_path.parent.mkdir(parents=True, exist_ok=True)
        frontier = self._quality_frontier(candidates)
        payload = [item.to_dict() for item in frontier]
        self.frontier_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.run_store.update_frontier(
            as_of_iteration=max(
                (_candidate_iteration(item.candidate_id) or 0 for item in candidates),
                default=0,
            ),
            candidates=candidates,
            frontier=frontier,
        )
        self._sync_pareto_frontier_index(candidates, frontier)

    def _sync_pareto_frontier_index(
        self,
        candidates: list[CandidateResult],
        frontier: list[CandidateResult],
    ) -> None:
        """Mirror the current pareto frontier into iteration_meta.

        Builds a {iteration → on_pareto_frontier} map across every iter
        we have at least one candidate for, then bulk-updates the
        ``iteration_meta`` table so MCP queries see a consistent
        snapshot. No-op when the trace index does not yet exist.
        """

        if not candidates:
            return
        if not self.trace_harness.indexer.db_path.exists():
            return
        frontier_iters = {
            _candidate_iteration(item.candidate_id)
            for item in frontier
        }
        frontier_iters.discard(None)
        flags: dict[int, bool] = {}
        for candidate in candidates:
            iteration = _candidate_iteration(candidate.candidate_id)
            if iteration is None:
                continue
            flags[iteration] = iteration in frontier_iters
        if flags:
            self.trace_harness.indexer.refresh_pareto_frontier(flags)

    def _append_summary(
        self,
        *,
        iteration: int,
        candidate: CandidateResult,
        proposal: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iteration": iteration,
            "candidate": candidate.to_dict(),
            "proposal": proposal or {},
            "self_best": [candidate.to_dict()],
        }
        self.run_store.record_candidates(
            iteration,
            [candidate],
            proposals_by_candidate={candidate.candidate_id: proposal or {}},
        )
        self._refresh_evidence_store(iteration)
        self._append_event(row)

    def _append_proposer_result_event(
        self,
        *,
        iteration: int,
        result: Any,
        selection_policy: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        tool_access_raw = getattr(result, "tool_access", {}) or {}
        tool_access = tool_access_raw if isinstance(tool_access_raw, dict) else {}
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iteration": iteration,
            "event": "proposer_result",
            "selection_policy": selection_policy,
            "proposer_agent": self.config.proposer_agent,
            "returncode": getattr(result, "returncode", None),
            "timed_out": bool(getattr(result, "timed_out", False)),
            "proposer_metrics": getattr(result, "metrics", {}) or {},
            "usage": getattr(result, "usage", None),
            "files_read": tool_access.get("files_read", {}),
            "files_written": tool_access.get("files_written", {}),
            "grep_requests": tool_access.get("grep_requests", []),
            "tool_counts": tool_access.get("tool_counts", {}),
            "evidence_usage": tool_access.get("evidence_usage", {}),
        }
        if extra:
            row.update(extra)
        self.run_store.record_proposer_call(
            iteration,
            result=result,
            selection_policy=selection_policy,
            proposer_agent=self.config.proposer_agent,
            extra=extra or {},
        )
        self._refresh_evidence_store(iteration)
        self._append_event(row)

    def _aggregate_proposer_metrics(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {}

        totals = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_reported_tokens": 0,
            "estimated_cost_usd": 0.0,
            "duration_s": 0.0,
            "tool_calls": 0,
            "read_file_calls": 0,
            "read_lines": 0,
            "write_file_calls": 0,
            "written_lines": 0,
            "runstore_tool_calls": 0,
            "runstore_trace_tool_calls": 0,
            "runstore_mod_tool_calls": 0,
            "raw_trace_file_reads": 0,
            "raw_reference_file_reads": 0,
            "raw_summary_file_reads": 0,
            "raw_evidence_file_reads": 0,
            "evidence_usage_events": 0,
        }
        tool_counts: dict[str, int] = {}
        unique_files_read: set[str] = set()

        for line in self.summary_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "proposer_result":
                continue

            metrics = row.get("proposer_metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            totals["calls"] += 1
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "total_reported_tokens",
                "tool_calls",
                "read_file_calls",
                "read_lines",
                "write_file_calls",
                "written_lines",
            ):
                totals[key] += _int_metric(metrics.get(key))
            for key in ("estimated_cost_usd", "duration_s"):
                totals[key] += _float_metric(metrics.get(key))

            evidence_usage = row.get("evidence_usage") or metrics.get("evidence_usage") or {}
            if isinstance(evidence_usage, dict):
                for key in (
                    "runstore_tool_calls",
                    "runstore_trace_tool_calls",
                    "runstore_mod_tool_calls",
                    "raw_trace_file_reads",
                    "raw_reference_file_reads",
                    "raw_summary_file_reads",
                    "raw_evidence_file_reads",
                    "evidence_usage_events",
                ):
                    totals[key] += _int_metric(evidence_usage.get(key))

            row_tool_counts = row.get("tool_counts") or metrics.get("tool_counts") or {}
            if isinstance(row_tool_counts, dict):
                for name, count in row_tool_counts.items():
                    tool_counts[str(name)] = tool_counts.get(str(name), 0) + _int_metric(
                        count
                    )

            files_read = row.get("files_read") or {}
            if isinstance(files_read, dict):
                unique_files_read.update(str(path) for path in files_read)

        totals["estimated_cost_usd"] = round(totals["estimated_cost_usd"], 6)
        totals["duration_s"] = round(totals["duration_s"], 3)
        totals["unique_files_read"] = len(unique_files_read)
        totals["tool_counts"] = dict(sorted(tool_counts.items()))
        events = totals["evidence_usage_events"]
        totals["evidence_usage_rate"] = (
            round(totals["runstore_tool_calls"] / events, 4) if events else 0.0
        )
        return totals

    def _append_event(self, row: dict[str, Any]) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


LocomoOptimizerConfig = OptimizerConfig
MemoOptimizer = LocomoOptimizer


def _candidate_score(item: CandidateResult) -> tuple[float, int, str]:
    return (item.passrate, item.average_score, -item.token_consuming, item.candidate_id)


def _candidate_quality_value(item: CandidateResult) -> float:
    return item.passrate


def _candidate_best_rank(item: CandidateResult) -> tuple[float, float, int, str]:
    return (-item.passrate, -item.average_score, item.token_consuming, item.candidate_id)


def _candidate_iteration(candidate_id: str) -> int | None:
    if not candidate_id.startswith("iter"):
        return None
    digits = []
    for char in candidate_id[4:]:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return None
    return int("".join(digits))


def _iteration_from_dir_name(name: str) -> int | None:
    if not name.startswith("iter_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _added_policy_lines(original: str, updated: str) -> str:
    original_lines = original.splitlines()
    updated_lines = updated.splitlines()
    prefix = 0
    limit = min(len(original_lines), len(updated_lines))
    while prefix < limit and original_lines[prefix] == updated_lines[prefix]:
        prefix += 1

    suffix = 0
    original_remaining = len(original_lines) - prefix
    updated_remaining = len(updated_lines) - prefix
    while (
        suffix < original_remaining
        and suffix < updated_remaining
        and original_lines[len(original_lines) - 1 - suffix]
        == updated_lines[len(updated_lines) - 1 - suffix]
    ):
        suffix += 1

    end = len(updated_lines) - suffix if suffix else len(updated_lines)
    return "\n".join(updated_lines[prefix:end])


def _single_top_k(raw: Any) -> tuple[int, bool]:
    if isinstance(raw, int):
        return raw, False
    if isinstance(raw, list) and raw:
        return int(raw[0]), len(raw) != 1
    return int(raw or 8), raw != 8


def _dedupe_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _dedupe_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)
