"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from optimizer1.benchmark_tasks import TASK_CHOICES, normalize_task_name, task_spec
from optimizer1.baseline import (
    DEFAULT_BASELINE_REPEATS,
    DEFAULT_BASELINE_SPLITS,
    run_baseline_suite,
)
from optimizer1.evaluation import run_initial_frontier
from optimizer1.locomo import prepare_locomo
from optimizer1.locomo_optimizer import LocomoOptimizer, LocomoOptimizerConfig
from optimizer1.longmemeval import (
    DEFAULT_LONGMEMEVAL_SCAFFOLDS,
    DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL,
    DEFAULT_LONGMEMEVAL_JUDGE_MODEL,
    prepare_longmemeval,
    run_longmemeval_frontier,
)
from optimizer1.longmemeval_optimizer import (
    LongMemEvalOptimizer,
    LongMemEvalOptimizerConfig,
)
from optimizer1.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from optimizer1.claude_runner import DEFAULT_OPENCODE_MODEL
from optimizer1.swebench import DEFAULT_MINI_SWE_AGENT_SOURCE_PATH
from optimizer1.swebench_optimizer import SwebenchOptimizer, SwebenchOptimizerConfig
from optimizer1.scaffolds import (
    DEFAULT_BASELINE_SCAFFOLDS,
    DEFAULT_EVOLUTION_SEED_SCAFFOLDS,
    DEFAULT_SCAFFOLD_TOP_KS,
    available_scaffolds,
)


def main() -> int:
    _load_project_env()
    parser = argparse.ArgumentParser(prog="optiharness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    locomo = subparsers.add_parser("locomo")
    locomo_sub = locomo.add_subparsers(dest="locomo_command", required=True)
    prepare = locomo_sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, default=None)
    prepare.add_argument("--dest", type=Path, default=None)
    prepare.add_argument("--allow-download", action="store_true")
    prepare.add_argument("--warmup-size", type=int, default=0)
    prepare.add_argument("--train-size", type=int, default=80)
    prepare.add_argument(
        "--train-sample-id",
        default="auto",
        help="Sample id to draw the train split from, or 'auto' for the largest available sample.",
    )
    prepare.add_argument("--seed", type=int, default=13)

    longmemeval = subparsers.add_parser("longmemeval")
    longmemeval_sub = longmemeval.add_subparsers(
        dest="longmemeval_command",
        required=True,
    )
    lme_prepare = longmemeval_sub.add_parser("prepare")
    lme_prepare.add_argument("--variant", choices=("s", "m", "oracle"), default="s")
    lme_prepare.add_argument("--source", type=Path, default=None)
    lme_prepare.add_argument("--dest", type=Path, default=None)
    lme_prepare.add_argument("--allow-download", action="store_true")
    lme_prepare.add_argument("--warmup-size", type=int, default=0)
    lme_prepare.add_argument("--train-size", type=int, default=100)
    lme_prepare.add_argument("--seed", type=int, default=13)

    lme_benchmark = longmemeval_sub.add_parser("benchmark")
    lme_benchmark.add_argument("--variant", choices=("s", "m", "oracle"), default="s")
    lme_benchmark.add_argument("--data-path", type=Path, default=None)
    lme_benchmark.add_argument("--split-path", type=Path, default=None)
    lme_benchmark.add_argument("--split", choices=("warmup", "train", "test"), default="train")
    lme_benchmark.add_argument("--limit", type=int, default=0)
    lme_benchmark.add_argument(
        "--out",
        type=Path,
        default=Path("runs/longmemeval_memory_scaffold_run"),
    )
    lme_benchmark.add_argument("--model", default=DEFAULT_MODEL)
    lme_benchmark.add_argument("--base-url", default=DEFAULT_BASE_URL)
    lme_benchmark.add_argument("--api-key", default="EMPTY")
    lme_benchmark.add_argument("--timeout-s", type=int, default=300)
    lme_benchmark.add_argument("--dry-run", action="store_true")
    lme_benchmark.add_argument("--max-context-chars", type=int, default=6000)
    lme_benchmark.add_argument("--eval-workers", type=int, default=1)
    lme_benchmark.add_argument("--judge-model", default=DEFAULT_LONGMEMEVAL_JUDGE_MODEL)
    lme_benchmark.add_argument("--judge-base-url", default=DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL)
    lme_benchmark.add_argument("--judge-api-key", default=None)
    lme_benchmark.add_argument("--judge-timeout-s", type=int, default=300)
    lme_benchmark.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Use local token/F1 scoring instead of LongMemEval's LLM-as-judge scorer.",
    )
    lme_benchmark.add_argument("--pareto-quality-threshold", type=float, default=0.125)
    lme_benchmark.add_argument("--force", action="store_true")
    lme_benchmark.add_argument(
        "--scaffolds",
        default=",".join(DEFAULT_LONGMEMEVAL_SCAFFOLDS),
        help=f"Comma-separated scaffolds. Default: {', '.join(DEFAULT_LONGMEMEVAL_SCAFFOLDS)}.",
    )
    lme_benchmark.add_argument("--top-k", default=None)
    lme_benchmark.add_argument("--question-types", default="")
    lme_benchmark.add_argument("--scaffold-extra-json", default=None)

    evolve = subparsers.add_parser("evolve")
    evolve.add_argument("--split", choices=("warmup", "train", "test"), default="train")
    evolve.add_argument("--limit", type=int, default=0)
    evolve.add_argument("--out", type=Path, default=Path("runs/locomo_memory_scaffold_run"))
    evolve.add_argument("--model", default=DEFAULT_MODEL)
    evolve.add_argument("--base-url", default=DEFAULT_BASE_URL)
    evolve.add_argument("--api-key", default="EMPTY")
    evolve.add_argument("--timeout-s", type=int, default=300)
    evolve.add_argument("--dry-run", action="store_true")
    evolve.add_argument("--max-context-chars", type=int, default=6000)
    evolve.add_argument("--eval-workers", type=int, default=1)
    evolve.add_argument(
        "--pareto-quality-threshold",
        type=float,
        default=0.125,
        help="Passrate gap above which lower-quality cheap candidates are excluded from Pareto.",
    )
    evolve.add_argument(
        "--force",
        action="store_true",
        help="Rerun existing candidate result files instead of reusing them.",
    )
    evolve.add_argument(
        "--scaffolds",
        default=",".join(DEFAULT_EVOLUTION_SEED_SCAFFOLDS),
        help=f"Comma-separated scaffolds. Available: {', '.join(available_scaffolds())}",
    )
    evolve.add_argument(
        "--top-k",
        default=None,
        help=(
            "Comma-separated top-k variants. Omit to use fixed scaffold defaults: "
            f"{_format_scaffold_top_k_defaults(DEFAULT_EVOLUTION_SEED_SCAFFOLDS)}."
        ),
    )
    evolve.add_argument(
        "--scaffold-extra-json",
        default=None,
        help="JSON object of per-scaffold extra config, or @path to a JSON file.",
    )

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument(
        "--splits",
        default=",".join(DEFAULT_BASELINE_SPLITS),
        help="Comma-separated splits to evaluate.",
    )
    baseline.add_argument("--repeats", type=int, default=DEFAULT_BASELINE_REPEATS)
    baseline.add_argument("--limit", type=int, default=0)
    baseline.add_argument("--out", type=Path, default=Path("runs/baselines"))
    baseline.add_argument("--model", default=DEFAULT_MODEL)
    baseline.add_argument("--base-url", default=DEFAULT_BASE_URL)
    baseline.add_argument("--api-key", default="EMPTY")
    baseline.add_argument("--timeout-s", type=int, default=300)
    baseline.add_argument("--dry-run", action="store_true")
    baseline.add_argument("--max-context-chars", type=int, default=6000)
    baseline.add_argument("--eval-workers", type=int, default=1)
    baseline.add_argument(
        "--scaffolds",
        default=",".join(DEFAULT_BASELINE_SCAFFOLDS),
        help=f"Comma-separated scaffolds. Available: {', '.join(available_scaffolds())}",
    )
    baseline.add_argument(
        "--top-k",
        default=None,
        help=(
            "Comma-separated top-k variants. Omit to use fixed scaffold defaults: "
            f"{_format_scaffold_top_k_defaults(DEFAULT_BASELINE_SCAFFOLDS)}."
        ),
    )
    baseline.add_argument(
        "--scaffold-extra-json",
        default=None,
        help="JSON object of per-scaffold extra config, or @path to a JSON file.",
    )
    baseline.add_argument(
        "--force",
        action="store_true",
        help="Rerun existing baseline repeat directories instead of reusing them.",
    )

    optimize = subparsers.add_parser("optimize")
    optimize.add_argument("--run-id", default=None)
    optimize.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default=None,
        help="Benchmark task to optimize. Backward-compatible aliases are accepted.",
    )
    task_flags = optimize.add_mutually_exclusive_group()
    task_flags.add_argument(
        "--locomo",
        dest="task_locomo",
        action="store_true",
        help="Optimize the LOCOMO memory benchmark.",
    )
    task_flags.add_argument(
        "--longmemeval",
        dest="task_longmemeval",
        action="store_true",
        help="Optimize the LongMemEval memory benchmark.",
    )
    task_flags.add_argument(
        "--swebench",
        dest="task_swebench",
        action="store_true",
        help="Optimize the SWE-bench coding-agent benchmark.",
    )
    optimize.add_argument("--iterations", type=int, default=20)
    optimize.add_argument("--split", choices=("warmup", "train", "test"), default="train")
    optimize.add_argument("--limit", type=int, default=0)
    optimize.add_argument("--out", type=Path, default=None)
    optimize.add_argument("--model", default=DEFAULT_MODEL)
    optimize.add_argument("--base-url", default=DEFAULT_BASE_URL)
    optimize.add_argument("--api-key", default="EMPTY")
    optimize.add_argument("--eval-timeout-s", type=int, default=300)
    optimize.add_argument(
        "--proposer-agent",
        choices=("opencode", "claude"),
        default="opencode",
        help=(
            "Code agent used to generate candidates: 'opencode' (default, "
            "any OpenAI-compatible provider) or 'claude' (Claude Code, "
            "routable via ANTHROPIC_BASE_URL to DeepSeek/Kimi/etc.)."
        ),
    )
    optimize.add_argument(
        "--opencode-model",
        default=DEFAULT_OPENCODE_MODEL,
        help=(
            "OpenCode model override in 'provider/model' form. "
            f"Defaults to {DEFAULT_OPENCODE_MODEL!r}."
        ),
    )
    optimize.add_argument(
        "--claude-model",
        default="deepseek-v4-pro[1m]",
        help=(
            "Model name for Claude Code proposer (sets ANTHROPIC_MODEL). "
            "Default 'deepseek-v4-pro[1m]' targets DeepSeek's anthropic-compat "
            "endpoint with the 1M context tier."
        ),
    )
    optimize.add_argument(
        "--claude-base-url",
        default="https://api.deepseek.com/anthropic",
        help="ANTHROPIC_BASE_URL for Claude Code proposer.",
    )
    optimize.add_argument(
        "--claude-auth-token",
        default=None,
        help=(
            "ANTHROPIC_AUTH_TOKEN for Claude Code proposer. If unset, falls "
            "back to DEEPSEEK_API_KEY env var, then ANTHROPIC_AUTH_TOKEN, "
            "then ANTHROPIC_API_KEY."
        ),
    )
    optimize.add_argument("--propose-timeout-s", type=int, default=2400)
    optimize.add_argument("--dry-run", action="store_true")
    optimize.add_argument("--max-context-chars", type=int, default=6000)
    optimize.add_argument("--eval-workers", type=int, default=1)
    optimize.add_argument(
        "--scaffolds",
        default=None,
        help=f"Comma-separated seed scaffolds. Available: {', '.join(available_scaffolds())}",
    )
    optimize.add_argument(
        "--scaffold-extra-json",
        default=None,
        help="JSON object of per-scaffold seed extra config, or @path to a JSON file.",
    )
    optimize.add_argument(
        "--selection-policy",
        choices=(
            "default",
            "progressive",
            "bandit",
            "random",
            "recent",
            "best",
            "curai",
            "curaii",
        ),
        default="default",
        help=(
            "Use fixed-high scoped context (default), adaptive progressive "
            "context loading, online bandit context selection, one of the "
            "fixed 3-iteration file-access baselines (random / recent / best), "
            "curai (= progressive plus a Stagnation Forensics prompt block "
            "when the schedule escalates to high budget), or curaii "
            "(= curai with the patch base resampled each iter from the top-3 "
            "previous candidates by passrate so successful mechanisms can "
            "compound rather than restarting from baseline every iter)."
        ),
    )
    optimize.add_argument(
        "--bandit-reward-window",
        type=int,
        default=8,
        help=(
            "Sliding-window length for bandit z-score reward. Default 8; "
            "the published v3 LoCoMo runs use 16."
        ),
    )
    optimize.add_argument(
        "--include-optimization-direction",
        action="store_true",
        help=(
            "Inject the Optimization Focus mechanism direction list into the "
            "proposer prompt. Off by default for every selection policy; "
            "set this flag to opt in regardless of policy."
        ),
    )
    optimize.add_argument(
        "--force-budget",
        choices=("", "low", "medium", "high"),
        default="",
        help=(
            "Override the per-iteration context budget for every iteration. "
            "Empty (default) keeps each policy's own decision: progressive "
            "uses its state machine, bandit uses its stagnation heuristic, "
            "default uses fixed-high. Setting low/medium/high pins the "
            "budget across the run for any policy."
        ),
    )
    optimize.add_argument(
        "--progressive-low-best-count",
        type=int,
        default=1,
        help=(
            "Number of best reference iterations surfaced to the proposer "
            "during low-budget iterations (selection_policy=progressive). "
            "Default 1."
        ),
    )
    optimize.add_argument(
        "--progressive-medium-best-count",
        type=int,
        default=3,
        help=(
            "Number of best reference iterations surfaced to the proposer "
            "during medium-budget iterations (selection_policy=progressive). "
            "Default 3."
        ),
    )
    optimize.add_argument(
        "--proposer-sandbox",
        choices=("none", "docker"),
        default="none",
        help="Run proposer code agents directly or inside a Docker filesystem sandbox.",
    )
    optimize.add_argument(
        "--proposer-docker-image",
        default="",
        help="Docker image used when --proposer-sandbox=docker.",
    )
    optimize.add_argument(
        "--proposer-docker-workspace",
        default="/workspace",
        help="Container path for the mounted proposer workspace.",
    )
    optimize.add_argument(
        "--proposer-docker-env",
        action="append",
        default=[],
        help="Extra environment variable name to pass into the proposer container.",
    )
    optimize.add_argument(
        "--proposer-docker-mount",
        action="append",
        default=[],
        help="Extra Docker volume mount spec, for example /host/path:/container/path:ro.",
    )
    optimize.add_argument(
        "--proposer-docker-user",
        default="",
        help="Optional Docker user spec for proposer containers, for example 1000:1000.",
    )
    optimize.add_argument(
        "--proposer-docker-home",
        default="",
        help="Optional HOME value inside the proposer container.",
    )
    optimize.add_argument(
        "--pareto-quality-threshold",
        type=float,
        default=0.125,
        help="Passrate gap above which lower-quality cheap candidates are excluded from Pareto.",
    )
    optimize.add_argument(
        "--skip-scaffold-eval",
        action="store_true",
        help="Resume from existing candidate_results instead of rerunning built-in scaffolds.",
    )
    optimize.add_argument(
        "--force",
        action="store_true",
        help="Rerun task-specific baseline/seed evaluations instead of reusing cached rows.",
    )
    optimize.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Load precomputed baseline candidates from this directory.",
    )
    optimize.add_argument(
        "--no-test-frontier",
        action="store_true",
        help="Do not evaluate the final train Pareto frontier on the test split.",
    )
    optimize.add_argument(
        "--test-frontier-limit",
        type=int,
        default=0,
        help="Optional limit for automatic final test-frontier evaluation.",
    )
    optimize.add_argument(
        "--trace-baseline",
        type=Path,
        default=None,
        help=(
            "Optional path to another run's traces/ directory to use as "
            "the diff baseline. If omitted, the current run is treated "
            "as a baseline (no diff is computed)."
        ),
    )
    optimize.add_argument(
        "--proposer-no-trace-harness-section",
        action="store_true",
        help=(
            "Skip the entire trace-harness file-path section in the "
            "proposer prompt. The harness still records traces/spans, "
            "diagnostic markdown, and the SQLite index server-side, and "
            "MCP tools remain registered on the proposer container, but "
            "the prompt no longer points at the on-disk paths. Use as a "
            "control arm for trace-harness exposure ablations."
        ),
    )
    optimize.add_argument(
        "--diff-embedding-model",
        default=None,
        help=(
            "Embedding model name (e.g. text-embedding-3-small). When set, "
            "each iteration's diff.patch is embedded after recording and "
            "stored in traces/index.db for the trace_similar MCP tool. "
            "If unset, embedding generation is skipped."
        ),
    )
    optimize.add_argument(
        "--diff-embedding-base-url",
        default=None,
        help="OpenAI-compatible base URL for the embedding model (optional).",
    )
    optimize.add_argument(
        "--diff-embedding-api-key",
        default=None,
        help="API key for the embedding model (optional).",
    )
    optimize.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Run a diagnoser subagent in the proposer container before "
            "each proposer step. The diagnoser explores traces and the "
            "source snapshot and writes a structured failure-mode report "
            "(diagnoser_report.json); the proposer reads it as hypothesis "
            "input. Reports are cached per (base_iter, reference_iters) "
            "within the run so identical lineage states do not re-diagnose."
        ),
    )
    optimize.add_argument("--longmemeval-variant", choices=("s", "m", "oracle"), default="s")
    optimize.add_argument("--longmemeval-data-path", type=Path, default=None)
    optimize.add_argument("--longmemeval-split-path", type=Path, default=None)
    optimize.add_argument("--longmemeval-question-types", default="")
    optimize.add_argument("--longmemeval-judge-model", default=DEFAULT_LONGMEMEVAL_JUDGE_MODEL)
    optimize.add_argument("--longmemeval-judge-base-url", default=DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL)
    optimize.add_argument("--longmemeval-judge-api-key", default=None)
    optimize.add_argument("--longmemeval-judge-timeout-s", type=int, default=300)
    optimize.add_argument("--longmemeval-no-llm-judge", action="store_true")
    optimize.add_argument("--swebench-data-path", type=Path, default=None)
    optimize.add_argument(
        "--mini-swe-agent-source-path",
        type=Path,
        default=DEFAULT_MINI_SWE_AGENT_SOURCE_PATH,
        help="Path to a local mini-SWE-agent source checkout.",
    )
    optimize.add_argument(
        "--mini-swe-agent-command",
        default="",
        help=(
            "Command template that runs one task and writes {patch_path}. "
            "Placeholders include {source_path}, {task_dir}, {instance_path}, "
            "{patch_path}, {instance_id}, {repo}, and {base_commit}."
        ),
    )
    optimize.add_argument(
        "--mini-swe-agent-eval-command",
        default="",
        help="Optional command template that evaluates {patch_path}; exit code 0 means pass.",
    )
    args = parser.parse_args()
    if args.command == "locomo" and args.locomo_command == "prepare":
        payload = prepare_locomo(
            dest=args.dest,
            source=args.source,
            allow_download=args.allow_download,
            warmup_size=args.warmup_size,
            train_size=args.train_size,
            train_sample_id=args.train_sample_id,
            seed=args.seed,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "longmemeval" and args.longmemeval_command == "prepare":
        payload = prepare_longmemeval(
            variant=args.variant,
            dest=args.dest,
            source=args.source,
            allow_download=args.allow_download,
            warmup_size=args.warmup_size,
            train_size=args.train_size,
            seed=args.seed,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "longmemeval" and args.longmemeval_command == "benchmark":
        selected_scaffolds = _csv(args.scaffolds)
        top_k = None if args.top_k is None else [int(item) for item in _csv(args.top_k)]
        scaffold_extra = _scaffold_extra(args.scaffold_extra_json)
        payload = run_longmemeval_frontier(
            split=args.split,
            limit=args.limit,
            out_dir=args.out,
            variant=args.variant,
            data_path=args.data_path,
            split_path=args.split_path,
            question_types=tuple(_csv(args.question_types)),
            scaffolds=selected_scaffolds,
            top_k_variants=top_k,
            scaffold_extra=scaffold_extra,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            max_context_chars=args.max_context_chars,
            max_eval_workers=args.eval_workers,
            force=args.force,
            pareto_quality_threshold=args.pareto_quality_threshold,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            judge_api_key=args.judge_api_key,
            judge_timeout_s=args.judge_timeout_s,
            use_llm_judge=not args.no_llm_judge,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "evolve":
        selected_scaffolds = _csv(args.scaffolds)
        top_k = None if args.top_k is None else [int(item) for item in _csv(args.top_k)]
        scaffold_extra = _scaffold_extra(args.scaffold_extra_json)
        payload = run_initial_frontier(
            split=args.split,
            limit=args.limit,
            out_dir=args.out,
            scaffolds=selected_scaffolds,
            top_k_variants=top_k,
            scaffold_extra=scaffold_extra,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            max_context_chars=args.max_context_chars,
            max_eval_workers=args.eval_workers,
            force=args.force,
            pareto_quality_threshold=args.pareto_quality_threshold,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "baseline":
        selected_splits = _csv(args.splits)
        selected_scaffolds = _csv(args.scaffolds)
        top_k = None if args.top_k is None else [int(item) for item in _csv(args.top_k)]
        scaffold_extra = _scaffold_extra(args.scaffold_extra_json)
        payload = run_baseline_suite(
            out_dir=args.out,
            splits=selected_splits,
            repeats=args.repeats,
            limit=args.limit,
            scaffolds=selected_scaffolds,
            top_k_variants=top_k,
            scaffold_extra=scaffold_extra,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            max_context_chars=args.max_context_chars,
            max_eval_workers=args.eval_workers,
            force=args.force,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "optimize":
        try:
            selected_task = _optimize_task(args)
        except ValueError as exc:
            parser.error(str(exc))
        selected_spec = task_spec(selected_task)
        run_id = args.run_id or selected_spec.default_run_id
        out_dir = args.out or Path("runs") / run_id
        if selected_task == "longmemeval":
            selected_scaffolds = (
                _csv(args.scaffolds)
                if args.scaffolds
                else list(DEFAULT_LONGMEMEVAL_SCAFFOLDS)
            )
            scaffold_extra = _scaffold_extra(args.scaffold_extra_json)
            optimizer = LongMemEvalOptimizer(
                LongMemEvalOptimizerConfig(
                    run_id=run_id,
                    out_dir=out_dir,
                    iterations=args.iterations,
                    split=args.split,
                    limit=args.limit,
                    dataset_variant=args.longmemeval_variant,
                    data_path=args.longmemeval_data_path,
                    split_path=args.longmemeval_split_path,
                    question_types=tuple(_csv(args.longmemeval_question_types)),
                    judge_model=args.longmemeval_judge_model,
                    judge_base_url=args.longmemeval_judge_base_url,
                    judge_api_key=args.longmemeval_judge_api_key,
                    judge_timeout_s=args.longmemeval_judge_timeout_s,
                    use_llm_judge=not args.longmemeval_no_llm_judge,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    eval_timeout_s=args.eval_timeout_s,
                    proposer_agent=args.proposer_agent,
                    opencode_model=args.opencode_model,
                    claude_model=args.claude_model,
                    claude_base_url=args.claude_base_url,
                    claude_auth_token=args.claude_auth_token,
                    propose_timeout_s=args.propose_timeout_s,
                    dry_run=args.dry_run,
                    max_context_chars=args.max_context_chars,
                    max_eval_workers=args.eval_workers,
                    skip_scaffold_eval=args.skip_scaffold_eval,
                    baseline_dir=args.baseline_dir,
                    scaffolds=tuple(selected_scaffolds),
                    scaffold_extra=scaffold_extra,
                    selection_policy=args.selection_policy,
                    include_optimization_direction=args.include_optimization_direction,
                    force_budget=args.force_budget,
                    progressive_low_best_count=args.progressive_low_best_count,
                    progressive_medium_best_count=args.progressive_medium_best_count,
                    bandit_reward_window=args.bandit_reward_window,
                    pareto_quality_threshold=args.pareto_quality_threshold,
                    proposer_sandbox=args.proposer_sandbox,
                    proposer_docker_image=args.proposer_docker_image,
                    proposer_docker_workspace=args.proposer_docker_workspace,
                    proposer_docker_env=tuple(_csv_many(args.proposer_docker_env)),
                    proposer_docker_mount=tuple(args.proposer_docker_mount or ()),
                    proposer_docker_user=args.proposer_docker_user,
                    proposer_docker_home=args.proposer_docker_home,
                    test_frontier=not args.no_test_frontier,
                    test_limit=args.test_frontier_limit,
                    trace_baseline_path=args.trace_baseline,
                    proposer_show_trace_harness_section=not args.proposer_no_trace_harness_section,
                    diff_embedding_model=args.diff_embedding_model,
                    diff_embedding_base_url=args.diff_embedding_base_url,
                    diff_embedding_api_key=args.diff_embedding_api_key,
                    diagnose=args.diagnose,
                )
            )
            payload = optimizer.run()
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if selected_task == "swebench":
            optimizer = SwebenchOptimizer(
                SwebenchOptimizerConfig(
                    run_id=run_id,
                    out_dir=out_dir,
                    iterations=args.iterations,
                    split=args.split,
                    limit=args.limit,
                    data_path=args.swebench_data_path,
                    mini_swe_agent_source_path=args.mini_swe_agent_source_path,
                    mini_swe_agent_command=args.mini_swe_agent_command,
                    mini_swe_agent_eval_command=args.mini_swe_agent_eval_command,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    eval_timeout_s=args.eval_timeout_s,
                    proposer_agent=args.proposer_agent,
                    opencode_model=args.opencode_model,
                    claude_model=args.claude_model,
                    claude_base_url=args.claude_base_url,
                    claude_auth_token=args.claude_auth_token,
                    propose_timeout_s=args.propose_timeout_s,
                    dry_run=args.dry_run,
                    max_context_chars=args.max_context_chars,
                    max_eval_workers=args.eval_workers,
                    skip_scaffold_eval=args.skip_scaffold_eval,
                    baseline_dir=args.baseline_dir,
                    selection_policy=args.selection_policy,
                    include_optimization_direction=args.include_optimization_direction,
                    force_budget=args.force_budget,
                    progressive_low_best_count=args.progressive_low_best_count,
                    progressive_medium_best_count=args.progressive_medium_best_count,
                    bandit_reward_window=args.bandit_reward_window,
                    pareto_quality_threshold=args.pareto_quality_threshold,
                    proposer_sandbox=args.proposer_sandbox,
                    proposer_docker_image=args.proposer_docker_image,
                    proposer_docker_workspace=args.proposer_docker_workspace,
                    proposer_docker_env=tuple(_csv_many(args.proposer_docker_env)),
                    proposer_docker_mount=tuple(args.proposer_docker_mount or ()),
                    proposer_docker_user=args.proposer_docker_user,
                    proposer_docker_home=args.proposer_docker_home,
                    force=args.force,
                    trace_baseline_path=args.trace_baseline,
                    proposer_show_trace_harness_section=not args.proposer_no_trace_harness_section,
                    diff_embedding_model=args.diff_embedding_model,
                    diff_embedding_base_url=args.diff_embedding_base_url,
                    diff_embedding_api_key=args.diff_embedding_api_key,
                    diagnose=args.diagnose,
                )
            )
            payload = optimizer.run()
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        selected_scaffolds = (
            _csv(args.scaffolds)
            if args.scaffolds
            else list(DEFAULT_EVOLUTION_SEED_SCAFFOLDS)
        )
        scaffold_extra = _scaffold_extra(args.scaffold_extra_json)
        optimizer = LocomoOptimizer(
            LocomoOptimizerConfig(
                run_id=run_id,
                out_dir=out_dir,
                iterations=args.iterations,
                split=args.split,
                limit=args.limit,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                eval_timeout_s=args.eval_timeout_s,
                proposer_agent=args.proposer_agent,
                opencode_model=args.opencode_model,
                claude_model=args.claude_model,
                claude_base_url=args.claude_base_url,
                claude_auth_token=args.claude_auth_token,
                propose_timeout_s=args.propose_timeout_s,
                dry_run=args.dry_run,
                max_context_chars=args.max_context_chars,
                max_eval_workers=args.eval_workers,
                skip_scaffold_eval=args.skip_scaffold_eval,
                baseline_dir=args.baseline_dir,
                scaffolds=tuple(selected_scaffolds),
                scaffold_extra=scaffold_extra,
                selection_policy=args.selection_policy,
                include_optimization_direction=args.include_optimization_direction,
                force_budget=args.force_budget,
                progressive_low_best_count=args.progressive_low_best_count,
                progressive_medium_best_count=args.progressive_medium_best_count,
                bandit_reward_window=args.bandit_reward_window,
                pareto_quality_threshold=args.pareto_quality_threshold,
                proposer_sandbox=args.proposer_sandbox,
                proposer_docker_image=args.proposer_docker_image,
                proposer_docker_workspace=args.proposer_docker_workspace,
                proposer_docker_env=tuple(_csv_many(args.proposer_docker_env)),
                proposer_docker_mount=tuple(args.proposer_docker_mount or ()),
                proposer_docker_user=args.proposer_docker_user,
                proposer_docker_home=args.proposer_docker_home,
                test_frontier=not args.no_test_frontier,
                test_limit=args.test_frontier_limit,
                trace_baseline_path=args.trace_baseline,
                proposer_show_trace_harness_section=not args.proposer_no_trace_harness_section,
                diff_embedding_model=args.diff_embedding_model,
                diff_embedding_base_url=args.diff_embedding_base_url,
                diff_embedding_api_key=args.diff_embedding_api_key,
                diagnose=args.diagnose,
            )
        )
        payload = optimizer.run()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    return 1


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_many(values: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(_csv(value))
    return out


def _load_project_env() -> None:
    """Load simple KEY=VALUE entries from the project .env without overriding env."""

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _scaffold_extra(value: str | None) -> dict[str, dict[str, object]]:
    if not value:
        return {}
    text = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--scaffold-extra-json must be a JSON object")
    out: dict[str, dict[str, object]] = {}
    for name, extra in payload.items():
        if not isinstance(extra, dict):
            raise ValueError(f"extra config for {name!r} must be a JSON object")
        out[str(name)] = dict(extra)
    return out


def _optimize_task(args: argparse.Namespace) -> str:
    flag_tasks = [
        task
        for task, enabled in (
            ("locomo", args.task_locomo),
            ("longmemeval", args.task_longmemeval),
            ("swebench", args.task_swebench),
        )
        if enabled
    ]
    if args.task and flag_tasks:
        raise ValueError("Use either --task or one benchmark flag, not both.")
    if flag_tasks:
        return flag_tasks[0]
    return normalize_task_name(args.task)

def _format_scaffold_top_k_defaults(scaffolds: tuple[str, ...]) -> str:
    return ", ".join(
        f"{scaffold}=top{DEFAULT_SCAFFOLD_TOP_KS[scaffold]}"
        for scaffold in scaffolds
    )


if __name__ == "__main__":
    raise SystemExit(main())
