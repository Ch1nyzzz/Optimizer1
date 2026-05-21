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
from optimizer1.swebench import DEFAULT_MINI_SWE_AGENT_SOURCE_PATH
from optimizer1.swebench_optimizer import SwebenchOptimizer, SwebenchOptimizerConfig
from optimizer1.graph_colouring import (
    DEFAULT_COMPILE_TIMEOUT_S as DEFAULT_GRAPH_COLOURING_COMPILE_TIMEOUT_S,
    DEFAULT_GRAPH_COLOURING_SOURCE_PATH,
    DEFAULT_TARGET_ALGORITHM as DEFAULT_GRAPH_COLOURING_TARGET_ALGORITHM,
    DEFAULT_TEST_INSTANCES as DEFAULT_GRAPH_COLOURING_TEST_INSTANCES,
    DEFAULT_TIME_BUDGET_S as DEFAULT_GRAPH_COLOURING_TIME_BUDGET_S,
    DEFAULT_TRAIN_INSTANCES as DEFAULT_GRAPH_COLOURING_TRAIN_INSTANCES,
)
from optimizer1.graph_colouring_optimizer import (
    GraphColouringOptimizer,
    GraphColouringOptimizerConfig,
)
from optimizer1.terminus import (
    DEFAULT_TERMINUS_AGENT_TIMEOUT_MULTIPLIER,
    DEFAULT_TERMINUS_CONCURRENCY,
    DEFAULT_TERMINUS_DATASET,
    DEFAULT_TERMINUS_ENV,
    DEFAULT_TERMINUS_HARBOR_COMMAND,
    DEFAULT_TERMINUS_JOB_TIMEOUT_S,
    DEFAULT_TERMINUS_REASONING_EFFORT,
    DEFAULT_TERMINUS_SOLVER_API_KEY_ENV,
    DEFAULT_TERMINUS_SOLVER_BASE_URL,
    DEFAULT_TERMINUS_SOLVER_MODEL,
    DEFAULT_TERMINUS_SOURCE_PATH,
    DEFAULT_TERMINUS_TASKS_PATH,
    DEFAULT_TERMINUS_TRIALS,
)
from optimizer1.terminus_optimizer import TerminusOptimizer, TerminusOptimizerConfig
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
    task_flags.add_argument(
        "--graph-colouring",
        dest="task_graph_colouring",
        action="store_true",
        help="Optimize the graph-colouring DIMACS C++ heuristic benchmark.",
    )
    task_flags.add_argument(
        "--terminus",
        dest="task_terminus",
        action="store_true",
        help="Optimize the Terminal-Bench 2.0 agent scaffold (Terminus-KIRA, Harbor-backed).",
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
        choices=("claude", "codex"),
        default="claude",
        help=(
            "Code agent used to generate candidates. 'claude' runs "
            "Claude Code (routable via ANTHROPIC_BASE_URL to "
            "DeepSeek/Kimi/etc.); 'codex' runs the Codex CLI against "
            "the user's ~/.codex ChatGPT login."
        ),
    )
    optimize.add_argument(
        "--claude-model",
        default=None,
        help=(
            "Model name for Claude Code proposer (sets ANTHROPIC_MODEL). "
            "Defaults to 'deepseek-v4-pro[1m]' (DeepSeek anthropic-compat "
            "endpoint, 1M context tier). With --claude-native-auth the "
            "default becomes empty so claude CLI picks its own model."
        ),
    )
    optimize.add_argument(
        "--claude-effort",
        choices=("", "low", "medium", "high", "xhigh", "max"),
        default="",
        help=(
            "Claude Code --effort level for proposer calls. Empty leaves the "
            "CLI default unchanged; use 'max' for maximum thinking effort."
        ),
    )
    optimize.add_argument(
        "--claude-base-url",
        default=None,
        help=(
            "ANTHROPIC_BASE_URL for Claude Code proposer. Defaults to the "
            "DeepSeek anthropic-compat endpoint. Ignored when "
            "--claude-native-auth is set."
        ),
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
    optimize.add_argument(
        "--claude-native-auth",
        action="store_true",
        help=(
            "Use the Claude Code subscription OAuth login from "
            "~/.claude/.credentials.json. When set, ANTHROPIC_BASE_URL, "
            "ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, and ANTHROPIC_MODEL "
            "are stripped from the proposer's environment so the underlying "
            "claude CLI uses its own credentials. --claude-base-url / "
            "--claude-auth-token are ignored, and --claude-model defaults "
            "to empty (no --model flag passed) unless explicitly provided."
        ),
    )
    optimize.add_argument(
        "--codex-model",
        default="gpt-5.5",
        help=(
            "Model name for Codex proposer (passed as `codex exec -m`). "
            "Default 'gpt-5.5'. Only used when --proposer-agent codex."
        ),
    )
    optimize.add_argument(
        "--codex-reasoning-effort",
        choices=("", "minimal", "low", "medium", "high", "xhigh"),
        default="high",
        help=(
            "Codex --c model_reasoning_effort. Default 'high'. Empty "
            "leaves the choice to the Codex CLI's own config.toml."
        ),
    )
    optimize.add_argument(
        "--codex-home",
        default="",
        help=(
            "Override $CODEX_HOME for the Codex proposer. Empty (default) "
            "uses ~/.codex. The directory must contain auth.json with a "
            "valid ChatGPT login (run `codex login` once on the host)."
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
            "pareto",
        ),
        default="default",
        help=(
            "Use fixed-high scoped context (default), adaptive progressive "
            "context loading, online bandit context selection, one of the "
            "fixed 3-iteration file-access baselines (random / recent / best), "
            "curai (= progressive plus a Stagnation Forensics prompt block "
            "when the schedule escalates to high budget), curaii "
            "(= curai with the patch base resampled each iter from the top-3 "
            "previous candidates by passrate so successful mechanisms can "
            "compound rather than restarting from baseline every iter), or "
            "pareto (= default's fixed-high context every iter, but the "
            "patch base is resampled uniformly from the current "
            "passrate x token_consuming Pareto frontier — same iter budget "
            "as default, the only divergence is the base)."
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
        "--resume",
        action="store_true",
        help=(
            "Continue an existing --run-id: reload candidate_results/ (implies "
            "--skip-scaffold-eval's load path, no seed re-eval), resume the "
            "iteration loop at max(completed iteration)+1, wipe stale per-iter "
            "dirs/rows for the iterations being rerun, and rebuild "
            "progressive_state.json from the surviving rows."
        ),
    )
    optimize.add_argument(
        "--no-wait-on-rate-limit",
        dest="wait_on_rate_limit",
        action="store_false",
        help=(
            "Disable the default behaviour of pausing the iteration loop until "
            "an Anthropic usage-limit window resets and retrying the same "
            "proposer call; instead a 429 fails the iteration immediately."
        ),
    )
    optimize.set_defaults(wait_on_rate_limit=True)
    optimize.add_argument(
        "--rate-limit-buffer-s",
        type=int,
        default=120,
        help="Seconds to wait past the reported reset time before retrying (default 120).",
    )
    optimize.add_argument(
        "--rate-limit-default-wait-s",
        type=int,
        default=1800,
        help="Fallback wait when the transcript carries no reset timestamp (default 1800).",
    )
    optimize.add_argument(
        "--rate-limit-max-wait-s",
        type=int,
        default=6 * 3600,
        help="Hard cap on a single rate-limit wait (default 21600 = 6h).",
    )
    optimize.add_argument(
        "--rate-limit-max-retries",
        type=int,
        default=8,
        help="Maximum consecutive rate-limit waits per proposer call before failing it (default 8).",
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
        "--test-frontier-candidate-limit",
        type=int,
        default=0,
        help=(
            "Optional number of train-frontier candidates to evaluate on the "
            "test split. Default 0 evaluates the whole frontier."
        ),
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
        "--no-summary",
        action="store_true",
        help=(
            "Withhold the cumulative cross-session summary from the proposer: "
            "the workspace `summaries/` directory (evolution_summary, "
            "best_candidates, candidate_score_table, etc.) is not created and "
            "the prompt's summary section is replaced with a note pointing at "
            "the raw `reference_iterations/iter_NNN/` bundles instead. The "
            "no-summary probe (tests what the cumulative digest actually buys)."
        ),
    )
    optimize.add_argument(
        "--organized",
        action="store_true",
        help=(
            "Use the organized optimizer interface: generate state.md from "
            "RunStore, register RunStore tools, and withhold cumulative "
            "summaries from the proposer workspace."
        ),
    )
    optimize.add_argument(
        "--organized-include-summaries",
        action="store_true",
        help=(
            "With --organized, also copy the cumulative summaries/ directory "
            "into the proposer workspace. This is an ablation of organized "
            "state.md + tools with summaries still available."
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
    optimize.add_argument(
        "--graph-colouring-source-path",
        type=Path,
        default=DEFAULT_GRAPH_COLOURING_SOURCE_PATH,
        help=(
            "Path to a local checkout of Rmehta-sudo/graph-colouring. Defaults "
            "to references/vendor/graph-colouring (run "
            "scripts/fetch_reference_repos.sh to clone)."
        ),
    )
    optimize.add_argument(
        "--graph-colouring-target-algorithm",
        default=DEFAULT_GRAPH_COLOURING_TARGET_ALGORITHM,
        help=(
            "Algorithm name passed to build/benchmark_runner --algorithm <NAME>. "
            "Default 'evolved' is the slot the proposer mutates; switch to "
            "dsatur/tabu_search/simulated_annealing/etc. only for diagnostic "
            "runs against unmodified upstream."
        ),
    )
    optimize.add_argument(
        "--graph-colouring-time-budget-s",
        type=int,
        default=DEFAULT_GRAPH_COLOURING_TIME_BUDGET_S,
        help="Per-instance runner timeout in seconds.",
    )
    optimize.add_argument(
        "--graph-colouring-compile-timeout-s",
        type=int,
        default=DEFAULT_GRAPH_COLOURING_COMPILE_TIMEOUT_S,
        help="Per-candidate `make all` timeout in seconds.",
    )
    optimize.add_argument(
        "--graph-colouring-train-instances",
        default="",
        help=(
            "Comma-separated DIMACS .col filenames to use as the training set. "
            "Empty = the curated default subset."
        ),
    )
    optimize.add_argument(
        "--graph-colouring-test-instances",
        default="",
        help=(
            "Comma-separated DIMACS .col filenames for the held-out test set. "
            "Empty = the curated default subset."
        ),
    )
    optimize.add_argument(
        "--terminus-source-path",
        type=Path,
        default=DEFAULT_TERMINUS_SOURCE_PATH,
        help=(
            "Path to the vendored terminal_bench_2 reference project (the agents/ "
            "directory the proposer evolves). Defaults to "
            "references/vendor/meta-harness/reference_examples/terminal_bench_2."
        ),
    )
    optimize.add_argument(
        "--terminus-tasks-path",
        type=Path,
        default=DEFAULT_TERMINUS_TASKS_PATH,
        help=(
            "JSON file listing the full Terminal-Bench 2.0 task ids, used to resolve "
            "the `test` split (all tasks minus the 30 hard tasks). The `train` split "
            "uses the 30 hard tasks regardless and needs no file."
        ),
    )
    optimize.add_argument(
        "--terminus-test-tasks",
        default="",
        help="Comma-separated TB2 task ids for the held-out test split (overrides --terminus-tasks-path).",
    )
    optimize.add_argument(
        "--terminus-dataset",
        default=DEFAULT_TERMINUS_DATASET,
        help="Harbor dataset name@version (default: terminal-bench@2.0).",
    )
    optimize.add_argument(
        "--terminus-env",
        default=DEFAULT_TERMINUS_ENV,
        help=(
            "Harbor environment type for rollouts (default: docker; e.g. "
            "runloop, daytona). 'daytona' runs each trial in a remote Daytona "
            "cloud sandbox and needs DAYTONA_API_KEY set."
        ),
    )
    optimize.add_argument(
        "--terminus-env-kwargs",
        action="append",
        default=[],
        help=(
            "Harbor environment kwarg as key=value, forwarded as `--ek`. "
            "Repeatable and comma-separated. Daytona keys: "
            "auto_stop_interval_mins, auto_delete_interval_mins, dind_image, "
            "dind_snapshot, network_block_all."
        ),
    )
    optimize.add_argument(
        "--terminus-harbor-command",
        default=DEFAULT_TERMINUS_HARBOR_COMMAND,
        help="Console command used to invoke Harbor (default: 'harbor'; it is a direct env dependency).",
    )
    optimize.add_argument(
        "--terminus-solver-model",
        default=DEFAULT_TERMINUS_SOLVER_MODEL,
        help="litellm model id for the Terminus agent's base model (default: openai/deepseek-v4-pro).",
    )
    optimize.add_argument(
        "--terminus-solver-base-url",
        default=DEFAULT_TERMINUS_SOLVER_BASE_URL,
        help="OpenAI-compatible base URL for the solver model (default: https://api.deepseek.com/v1).",
    )
    optimize.add_argument(
        "--terminus-solver-api-key-env",
        default=DEFAULT_TERMINUS_SOLVER_API_KEY_ENV,
        help="Environment variable holding the solver API key (default: DEEPSEEK_API_KEY).",
    )
    optimize.add_argument(
        "--terminus-reasoning-effort",
        default=DEFAULT_TERMINUS_REASONING_EFFORT,
        help="Reasoning effort passed to the solver model ('high' == max; '' to omit).",
    )
    optimize.add_argument(
        "--terminus-rollout-trials",
        type=int,
        default=DEFAULT_TERMINUS_TRIALS,
        help="Attempts per task during rollouts (default: 2, matching meta-harness).",
    )
    optimize.add_argument(
        "--terminus-rollout-concurrency",
        type=int,
        default=DEFAULT_TERMINUS_CONCURRENCY,
        help=(
            "Max concurrent harbor trials (default: 12 for a local docker box; "
            "raise for runloop; for daytona keep it at or below your account's "
            "concurrent-sandbox quota)."
        ),
    )
    optimize.add_argument(
        "--terminus-agent-timeout-multiplier",
        type=float,
        default=DEFAULT_TERMINUS_AGENT_TIMEOUT_MULTIPLIER,
        help=(
            "Multiplier on Harbor's per-task agent timeout (passed as "
            "--agent-timeout-multiplier). Default 2.0 because DeepSeek v4 Pro at "
            "high reasoning effort is slow; 0 keeps harbor's own default."
        ),
    )
    optimize.add_argument(
        "--terminus-job-timeout-s",
        type=int,
        default=DEFAULT_TERMINUS_JOB_TIMEOUT_S,
        help="Wall-clock timeout for one harbor run (all tasks x trials of a candidate).",
    )
    optimize.add_argument(
        "--terminus-no-secondary-baseline",
        action="store_true",
        help="Skip evaluating the secondary baseline (agents.baseline_terminus2) when seeding the frontier.",
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
        resolved_claude_model, resolved_claude_base_url = _resolve_claude_overrides(
            claude_model=args.claude_model,
            claude_base_url=args.claude_base_url,
            native_auth=args.claude_native_auth,
        )
        args.claude_model = resolved_claude_model
        args.claude_base_url = resolved_claude_base_url
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
                    claude_model=args.claude_model,
                    claude_effort=args.claude_effort,
                    claude_base_url=args.claude_base_url,
                    claude_auth_token=args.claude_auth_token,
                    claude_native_auth=args.claude_native_auth,
                    codex_model=args.codex_model,
                    codex_reasoning_effort=args.codex_reasoning_effort,
                    codex_home=args.codex_home,
                    propose_timeout_s=args.propose_timeout_s,
                    dry_run=args.dry_run,
                    max_context_chars=args.max_context_chars,
                    max_eval_workers=args.eval_workers,
                    skip_scaffold_eval=args.skip_scaffold_eval,
                    resume=args.resume,
                    wait_on_rate_limit=args.wait_on_rate_limit,
                    rate_limit_buffer_s=args.rate_limit_buffer_s,
                    rate_limit_default_wait_s=args.rate_limit_default_wait_s,
                    rate_limit_max_wait_s=args.rate_limit_max_wait_s,
                    rate_limit_max_retries=args.rate_limit_max_retries,
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
                    test_frontier_candidate_limit=args.test_frontier_candidate_limit,
                    trace_baseline_path=args.trace_baseline,
                    proposer_show_trace_harness_section=not args.proposer_no_trace_harness_section,
                    summaries_in_workspace=_summaries_in_workspace(args),
                    organized=args.organized,
                    organized_include_summaries=args.organized_include_summaries,
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
                    claude_model=args.claude_model,
                    claude_effort=args.claude_effort,
                    claude_base_url=args.claude_base_url,
                    claude_auth_token=args.claude_auth_token,
                    claude_native_auth=args.claude_native_auth,
                    codex_model=args.codex_model,
                    codex_reasoning_effort=args.codex_reasoning_effort,
                    codex_home=args.codex_home,
                    propose_timeout_s=args.propose_timeout_s,
                    dry_run=args.dry_run,
                    max_context_chars=args.max_context_chars,
                    max_eval_workers=args.eval_workers,
                    skip_scaffold_eval=args.skip_scaffold_eval,
                    resume=args.resume,
                    wait_on_rate_limit=args.wait_on_rate_limit,
                    rate_limit_buffer_s=args.rate_limit_buffer_s,
                    rate_limit_default_wait_s=args.rate_limit_default_wait_s,
                    rate_limit_max_wait_s=args.rate_limit_max_wait_s,
                    rate_limit_max_retries=args.rate_limit_max_retries,
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
                    test_frontier=not args.no_test_frontier,
                    test_limit=args.test_frontier_limit,
                    test_frontier_candidate_limit=args.test_frontier_candidate_limit,
                    trace_baseline_path=args.trace_baseline,
                    proposer_show_trace_harness_section=not args.proposer_no_trace_harness_section,
                    summaries_in_workspace=_summaries_in_workspace(args),
                    organized=args.organized,
                    organized_include_summaries=args.organized_include_summaries,
                )
            )
            payload = optimizer.run()
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if selected_task == "terminus":
            optimizer = TerminusOptimizer(
                TerminusOptimizerConfig(
                    run_id=run_id,
                    out_dir=out_dir,
                    iterations=args.iterations,
                    split=args.split,
                    limit=args.limit,
                    tasks_path=args.terminus_tasks_path,
                    test_tasks=tuple(_csv(args.terminus_test_tasks)),
                    terminus_source_path=args.terminus_source_path,
                    dataset=args.terminus_dataset,
                    harbor_environment=args.terminus_env,
                    env_kwargs=tuple(_csv_many(args.terminus_env_kwargs)),
                    harbor_command=args.terminus_harbor_command,
                    solver_model=args.terminus_solver_model,
                    solver_base_url=args.terminus_solver_base_url,
                    solver_api_key_env=args.terminus_solver_api_key_env,
                    reasoning_effort=args.terminus_reasoning_effort,
                    rollout_trials=args.terminus_rollout_trials,
                    rollout_concurrency=args.terminus_rollout_concurrency,
                    agent_timeout_multiplier=args.terminus_agent_timeout_multiplier,
                    job_timeout_s=args.terminus_job_timeout_s,
                    include_secondary_baseline=not args.terminus_no_secondary_baseline,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    eval_timeout_s=args.eval_timeout_s,
                    proposer_agent=args.proposer_agent,
                    claude_model=args.claude_model,
                    claude_effort=args.claude_effort,
                    claude_base_url=args.claude_base_url,
                    claude_auth_token=args.claude_auth_token,
                    claude_native_auth=args.claude_native_auth,
                    codex_model=args.codex_model,
                    codex_reasoning_effort=args.codex_reasoning_effort,
                    codex_home=args.codex_home,
                    propose_timeout_s=args.propose_timeout_s,
                    dry_run=args.dry_run,
                    max_context_chars=args.max_context_chars,
                    max_eval_workers=args.eval_workers,
                    skip_scaffold_eval=args.skip_scaffold_eval,
                    resume=args.resume,
                    wait_on_rate_limit=args.wait_on_rate_limit,
                    rate_limit_buffer_s=args.rate_limit_buffer_s,
                    rate_limit_default_wait_s=args.rate_limit_default_wait_s,
                    rate_limit_max_wait_s=args.rate_limit_max_wait_s,
                    rate_limit_max_retries=args.rate_limit_max_retries,
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
                    test_frontier=not args.no_test_frontier,
                    test_limit=args.test_frontier_limit,
                    test_frontier_candidate_limit=args.test_frontier_candidate_limit,
                    trace_baseline_path=args.trace_baseline,
                    proposer_show_trace_harness_section=not args.proposer_no_trace_harness_section,
                    summaries_in_workspace=_summaries_in_workspace(args),
                    organized=args.organized,
                    organized_include_summaries=args.organized_include_summaries,
                )
            )
            payload = optimizer.run()
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if selected_task == "graph_colouring":
            train_instances = tuple(_csv(args.graph_colouring_train_instances)) or (
                DEFAULT_GRAPH_COLOURING_TRAIN_INSTANCES
            )
            test_instances = tuple(_csv(args.graph_colouring_test_instances)) or (
                DEFAULT_GRAPH_COLOURING_TEST_INSTANCES
            )
            optimizer = GraphColouringOptimizer(
                GraphColouringOptimizerConfig(
                    run_id=run_id,
                    out_dir=out_dir,
                    iterations=args.iterations,
                    split=args.split,
                    limit=args.limit,
                    graph_colouring_source_path=args.graph_colouring_source_path,
                    target_algorithm=args.graph_colouring_target_algorithm,
                    time_budget_s=args.graph_colouring_time_budget_s,
                    compile_timeout_s=args.graph_colouring_compile_timeout_s,
                    train_instances=train_instances,
                    test_instances=test_instances,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    eval_timeout_s=args.eval_timeout_s,
                    proposer_agent=args.proposer_agent,
                    claude_model=args.claude_model,
                    claude_effort=args.claude_effort,
                    claude_base_url=args.claude_base_url,
                    claude_auth_token=args.claude_auth_token,
                    claude_native_auth=args.claude_native_auth,
                    codex_model=args.codex_model,
                    codex_reasoning_effort=args.codex_reasoning_effort,
                    codex_home=args.codex_home,
                    propose_timeout_s=args.propose_timeout_s,
                    dry_run=args.dry_run,
                    max_context_chars=args.max_context_chars,
                    max_eval_workers=args.eval_workers,
                    skip_scaffold_eval=args.skip_scaffold_eval,
                    resume=args.resume,
                    wait_on_rate_limit=args.wait_on_rate_limit,
                    rate_limit_buffer_s=args.rate_limit_buffer_s,
                    rate_limit_default_wait_s=args.rate_limit_default_wait_s,
                    rate_limit_max_wait_s=args.rate_limit_max_wait_s,
                    rate_limit_max_retries=args.rate_limit_max_retries,
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
                    summaries_in_workspace=_summaries_in_workspace(args),
                    organized=args.organized,
                    organized_include_summaries=args.organized_include_summaries,
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
                claude_model=args.claude_model,
                claude_effort=args.claude_effort,
                claude_base_url=args.claude_base_url,
                claude_auth_token=args.claude_auth_token,
                claude_native_auth=args.claude_native_auth,
                codex_model=args.codex_model,
                codex_reasoning_effort=args.codex_reasoning_effort,
                codex_home=args.codex_home,
                propose_timeout_s=args.propose_timeout_s,
                dry_run=args.dry_run,
                max_context_chars=args.max_context_chars,
                max_eval_workers=args.eval_workers,
                skip_scaffold_eval=args.skip_scaffold_eval,
                resume=args.resume,
                wait_on_rate_limit=args.wait_on_rate_limit,
                rate_limit_buffer_s=args.rate_limit_buffer_s,
                rate_limit_default_wait_s=args.rate_limit_default_wait_s,
                rate_limit_max_wait_s=args.rate_limit_max_wait_s,
                rate_limit_max_retries=args.rate_limit_max_retries,
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
                test_frontier_candidate_limit=args.test_frontier_candidate_limit,
                trace_baseline_path=args.trace_baseline,
                proposer_show_trace_harness_section=not args.proposer_no_trace_harness_section,
                summaries_in_workspace=_summaries_in_workspace(args),
                organized=args.organized,
                organized_include_summaries=args.organized_include_summaries,
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


def _summaries_in_workspace(args: argparse.Namespace) -> bool:
    if args.no_summary:
        return False
    if args.organized and not args.organized_include_summaries:
        return False
    return True


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


def _resolve_claude_overrides(
    *,
    claude_model: str | None,
    claude_base_url: str | None,
    native_auth: bool,
) -> tuple[str, str]:
    """Pick the right defaults for ``--claude-model`` / ``--claude-base-url``.

    Argparse stores ``None`` when the user did not pass either flag. We
    fall back to the DeepSeek anthropic-compat tier in legacy mode and
    to empty strings under ``--claude-native-auth`` so the OAuth login
    is not poisoned with a DeepSeek model name or base URL.
    """

    if native_auth:
        resolved_model = claude_model if claude_model is not None else ""
        resolved_base = claude_base_url if claude_base_url is not None else ""
        return resolved_model, resolved_base
    resolved_model = claude_model if claude_model is not None else "deepseek-v4-pro[1m]"
    resolved_base = (
        claude_base_url
        if claude_base_url is not None
        else "https://api.deepseek.com/anthropic"
    )
    return resolved_model, resolved_base


def _optimize_task(args: argparse.Namespace) -> str:
    flag_tasks = [
        task
        for task, enabled in (
            ("locomo", args.task_locomo),
            ("longmemeval", args.task_longmemeval),
            ("swebench", args.task_swebench),
            ("graph_colouring", args.task_graph_colouring),
            ("terminus", args.task_terminus),
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
