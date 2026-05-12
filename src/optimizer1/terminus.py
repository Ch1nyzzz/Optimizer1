"""Terminal-Bench 2.0 agent-scaffold evolution support (harbor-backed).

This module is the SWE-bench-mini analogue for Terminal-Bench 2.0. The
"scaffold" being evolved is an agent class — a single Python file under
``agents/<name>.py`` in the vendored ``terminal_bench_2`` reference
project that subclasses ``harbor.agents.terminus_2.terminus_2.Terminus2``
(seeded from ``baseline_kira.py``, i.e. Terminus-KIRA).

The evaluator runs Harbor in a child process (``harbor run ...``) — Harbor
is a direct environment dependency (see the ``terminus`` extra), so this is
a plain console-script invocation, not the ``uv run`` indirection used by
the upstream meta-harness scripts. Harbor handles its own per-trial
concurrency, so exactly one ``harbor run`` covers all tasks × trials of a
candidate. Per-trial rewards and token counts are read back from the job
directory's ``<task>__<n>/result.json`` files, mirroring the meta-harness
``parse_job_results`` / ``parse_trial_metrics`` logic.

The default solver model is DeepSeek v4 Pro (high reasoning effort) reached
through litellm's OpenAI-compatible provider; the rollout sandbox defaults
to Harbor's local ``docker`` environment.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from optimizer1.pareto import ParetoPoint, save_frontier
from optimizer1.schemas import CandidateResult, TaskResult


DEFAULT_TERMINUS_SOURCE_PATH = Path(
    "references/vendor/meta-harness/reference_examples/terminal_bench_2"
)
DEFAULT_TERMINUS_AGENT_NAME = "terminus_kira_source"
DEFAULT_TERMINUS_SOURCE_FAMILY = DEFAULT_TERMINUS_AGENT_NAME
DEFAULT_TERMINUS_DATASET = "terminal-bench@2.0"
DEFAULT_TERMINUS_ENV = "docker"
DEFAULT_TERMINUS_HARBOR_COMMAND = "harbor"
DEFAULT_TERMINUS_SOLVER_MODEL = "openai/deepseek-v4-pro"
DEFAULT_TERMINUS_SOLVER_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_TERMINUS_SOLVER_API_KEY_ENV = "DEEPSEEK_API_KEY"
# litellm's reasoning_effort token; the user-facing knob is "max effort",
# which maps to litellm's highest tier "high".
DEFAULT_TERMINUS_REASONING_EFFORT = "high"
DEFAULT_TERMINUS_SEED_AGENT_IMPORT_PATH = "agents.baseline_kira:AgentHarness"
DEFAULT_TERMINUS_SECONDARY_BASELINE_NAME = "terminus2_baseline"
DEFAULT_TERMINUS_SECONDARY_BASELINE_IMPORT_PATH = "agents.baseline_terminus2:AgentHarness"
DEFAULT_TERMINUS_TRIALS = 2
DEFAULT_TERMINUS_CONCURRENCY = 12
DEFAULT_TERMINUS_SMOKE_TASK = "extract-elf"
DEFAULT_TERMINUS_JOB_TIMEOUT_S = 6 * 3600
# DeepSeek v4 Pro at high ("max") reasoning effort is slow enough to trip
# Harbor's default per-task agent timeout even on the easy smoke task, so the
# default stretches it. 0 keeps harbor's own default.
DEFAULT_TERMINUS_AGENT_TIMEOUT_MULTIPLIER = 2.0

# The 30-task "hard" subset used as the cheaper development/train split, copied
# verbatim from meta-harness reference_examples/terminal_bench_2/scripts/run_eval.sh.
TERMINUS_HARD_TASKS: tuple[str, ...] = (
    "bn-fit-modify",
    "cancel-async-tasks",
    "circuit-fibsqrt",
    "configure-git-webserver",
    "dna-assembly",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "fix-code-vulnerability",
    "fix-ocaml-gc",
    "gpt2-codegolf",
    "install-windows-3.11",
    "llm-inference-batching-scheduler",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "model-extraction-relu-logits",
    "password-recovery",
    "path-tracing",
    "path-tracing-reverse",
    "polyglot-rust-c",
    "protein-assembly",
    "regex-chess",
    "sam-cell-seg",
    "sparql-university",
    "torch-pipeline-parallelism",
    "torch-tensor-parallelism",
    "train-fasttext",
    "video-processing",
    "write-compressor",
)

# Where `optimizer1 terminus prepare` writes the full TB2 task list so the
# `test` split (all tasks minus the hard subset) can be resolved offline.
DEFAULT_TERMINUS_TASKS_PATH = Path("data/terminus/tasks.json")


@dataclass(frozen=True)
class TerminusInstance:
    """One Terminal-Bench 2.0 task in the split being optimized."""

    task_id: str
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TerminusInstance":
        task_id = str(
            payload.get("task_id") or payload.get("id") or payload.get("name") or ""
        ).strip()
        if not task_id:
            raise ValueError("Terminal-Bench instance must include task_id/id/name")
        metadata = {
            str(key): value
            for key, value in payload.items()
            if key not in {"task_id", "id", "name", "split"}
        }
        return cls(
            task_id=task_id,
            split=str(payload.get("split") or "train"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["id"] = self.task_id
        return payload


@dataclass(frozen=True)
class TerminusTrialRun:
    """One agent rollout (task × attempt) parsed from a harbor job dir."""

    task_id: str
    attempt: int
    passed: bool
    score: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def load_terminus_instances(
    tasks_path: Path | None,
    *,
    split: str = "train",
    limit: int = 0,
    train_tasks: Sequence[str] | None = None,
    test_tasks: Sequence[str] | None = None,
    smoke_task: str = DEFAULT_TERMINUS_SMOKE_TASK,
) -> list[TerminusInstance]:
    """Resolve the Terminal-Bench 2.0 task list for a split.

    - ``train``  → the 30-task ``hard`` subset (overridable via ``train_tasks``)
    - ``test``   → all TB2 tasks minus the ``hard`` subset (needs either an
      explicit ``test_tasks`` list or a ``tasks_path`` JSON written by
      ``optimizer1 terminus prepare``)
    - ``warmup`` → the single smoke task
    """

    normalized = (split or "train").strip().lower()
    hard = list(train_tasks) if train_tasks else list(TERMINUS_HARD_TASKS)

    if normalized in {"warmup", "smoke"}:
        ids = [smoke_task]
    elif normalized in {"train", ""}:
        ids = list(hard)
    elif normalized in {"test", "eval", "holdout"}:
        if test_tasks:
            ids = list(test_tasks)
        else:
            all_tasks = _load_all_task_ids(tasks_path)
            if all_tasks is None:
                raise ValueError(
                    "Terminal-Bench `test` split needs the full TB2 task list. "
                    "Pass an explicit task list (CLI: --terminus-test-tasks) or "
                    "run `optimizer1 terminus prepare` to write "
                    f"{DEFAULT_TERMINUS_TASKS_PATH}. The 30 `hard` tasks excluded "
                    "from the test split are: " + ", ".join(TERMINUS_HARD_TASKS)
                )
            hard_set = set(hard)
            ids = [task for task in all_tasks if task not in hard_set]
    else:
        raise ValueError(f"unknown terminus split: {split!r}")

    if limit:
        ids = ids[:limit]
    return [TerminusInstance(task_id=task_id, split=normalized) for task_id in ids]


def _load_all_task_ids(tasks_path: Path | None) -> list[str] | None:
    path = tasks_path or DEFAULT_TERMINUS_TASKS_PATH
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        rows = payload.get("all") or payload.get("tasks") or payload.get("task_ids")
    else:
        rows = payload
    if not isinstance(rows, list):
        return None
    out: list[str] = []
    for row in rows:
        if isinstance(row, str):
            out.append(row)
        elif isinstance(row, Mapping):
            value = row.get("task_id") or row.get("id") or row.get("name")
            if value:
                out.append(str(value))
    return out or None


class TerminusHarborRunner:
    """Evaluate a Terminal-Bench 2.0 agent-scaffold candidate via Harbor."""

    def __init__(
        self,
        *,
        instances: list[TerminusInstance],
        out_dir: Path,
        source_path: Path = DEFAULT_TERMINUS_SOURCE_PATH,
        dataset: str = DEFAULT_TERMINUS_DATASET,
        env: str = DEFAULT_TERMINUS_ENV,
        harbor_command: str = DEFAULT_TERMINUS_HARBOR_COMMAND,
        solver_model: str = DEFAULT_TERMINUS_SOLVER_MODEL,
        solver_base_url: str = DEFAULT_TERMINUS_SOLVER_BASE_URL,
        solver_api_key_env: str = DEFAULT_TERMINUS_SOLVER_API_KEY_ENV,
        reasoning_effort: str = DEFAULT_TERMINUS_REASONING_EFFORT,
        trials: int = DEFAULT_TERMINUS_TRIALS,
        concurrency: int = DEFAULT_TERMINUS_CONCURRENCY,
        agent_timeout_multiplier: float = DEFAULT_TERMINUS_AGENT_TIMEOUT_MULTIPLIER,
        seed_agent_import_path: str = DEFAULT_TERMINUS_SEED_AGENT_IMPORT_PATH,
        jobs_dir: Path | None = None,
        timeout_s: int = DEFAULT_TERMINUS_JOB_TIMEOUT_S,
        dry_run: bool = False,
        force: bool = False,
    ) -> None:
        self.instances = instances
        # Harbor runs with cwd set to the vendored terminal_bench_2 project, so any
        # relative output path it is handed gets re-resolved against *that* dir, not
        # the optimizer's cwd. Resolve everything to absolute paths up front.
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.source_path = Path(source_path).expanduser().resolve()
        self.dataset = dataset
        self.env = env
        self.harbor_command = harbor_command
        self.solver_model = solver_model
        self.solver_base_url = solver_base_url
        self.solver_api_key_env = solver_api_key_env
        self.reasoning_effort = reasoning_effort
        self.trials = max(1, int(trials))
        self.concurrency = max(1, int(concurrency))
        self.agent_timeout_multiplier = float(agent_timeout_multiplier or 0.0)
        self.seed_agent_import_path = seed_agent_import_path
        self.jobs_dir = (
            Path(jobs_dir).expanduser().resolve()
            if jobs_dir is not None
            else self.out_dir / "terminus_jobs"
        )
        self.timeout_s = max(60, int(timeout_s))
        self.dry_run = dry_run
        self.force = force

    # ------------------------------------------------------------------ public

    def evaluate_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        agent_name: str = DEFAULT_TERMINUS_AGENT_NAME,
    ) -> CandidateResult:
        candidate_dir = self.out_dir / "candidate_results"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        result_path = candidate_dir / f"{candidate_id}.json"
        if not self.force:
            existing = _load_candidate_result(
                result_path,
                candidate_id=candidate_id,
                agent_name=agent_name,
                config=dict(candidate),
            )
            if existing is not None:
                return existing

        agent_import_path = self._candidate_agent_import_path(candidate)
        source_project_path = self._candidate_source_path(candidate)

        if self.dry_run:
            trial_runs = [
                TerminusTrialRun(
                    task_id=instance.task_id,
                    attempt=0,
                    passed=False,
                    score=0.0,
                    metadata={
                        "benchmark": "terminus",
                        "agent": agent_name,
                        "agent_import_path": agent_import_path,
                        "dry_run": True,
                    },
                )
                for instance in self.instances
            ]
        else:
            job_dir = self._run_harbor(
                agent_import_path=agent_import_path,
                source_project_path=source_project_path,
                job_name=candidate_id,
                agent_name=agent_name,
            )
            trial_runs = parse_terminus_job_dir(
                job_dir,
                agent_name=agent_name,
                agent_import_path=agent_import_path,
            )

        return self._finalize(
            trial_runs,
            candidate=candidate,
            candidate_id=candidate_id,
            agent_name=agent_name,
            result_path=result_path,
        )

    # ----------------------------------------------------------------- internal

    def _finalize(
        self,
        trial_runs: list[TerminusTrialRun],
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        agent_name: str,
        result_path: Path,
    ) -> CandidateResult:
        task_results: list[TaskResult] = []
        for run in trial_runs:
            label = run.task_id if run.attempt == 0 and not _has_multiple_attempts(
                trial_runs, run.task_id
            ) else f"{run.task_id}__{run.attempt}"
            task_results.append(
                TaskResult(
                    task_id=label,
                    question=run.task_id,
                    gold_answer="verifier reward > 0",
                    prediction=f"reward={run.score}",
                    score=run.score,
                    passed=run.passed,
                    prompt_tokens=run.prompt_tokens,
                    completion_tokens=run.completion_tokens,
                    retrieved=[],
                    metadata={
                        **run.metadata,
                        "terminus_task_id": run.task_id,
                        "attempt": run.attempt,
                        "cache_tokens": run.cache_tokens,
                    },
                )
            )

        count = len(task_results)
        passrate = (
            sum(1 for item in task_results if item.passed) / count if count else 0.0
        )
        average_score = (
            sum(item.score for item in task_results) / count if count else 0.0
        )
        prompt_tokens = sum(item.prompt_tokens for item in task_results)
        completion_tokens = sum(item.completion_tokens for item in task_results)
        token_consuming = prompt_tokens + completion_tokens
        result = CandidateResult(
            candidate_id=candidate_id,
            scaffold_name=agent_name,
            passrate=passrate,
            average_score=average_score,
            token_consuming=token_consuming,
            avg_token_consuming=(token_consuming / count if count else 0.0),
            avg_prompt_tokens=(prompt_tokens / count if count else 0.0),
            avg_completion_tokens=(completion_tokens / count if count else 0.0),
            count=count,
            config=dict(candidate),
            result_path=str(result_path),
        )
        payload = {
            "candidate": result.to_dict(),
            "tasks": [item.to_dict() for item in task_results],
            "score_breakdown": _score_breakdown(task_results),
        }
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    def _run_harbor(
        self,
        *,
        agent_import_path: str,
        source_project_path: Path,
        job_name: str,
        agent_name: str,
    ) -> Path:
        if not source_project_path.exists():
            raise FileNotFoundError(
                "Terminal-Bench source project does not exist. Vendor it under "
                f"{DEFAULT_TERMINUS_SOURCE_PATH} (or pass --terminus-source-path / "
                "source_project_path): " + str(source_project_path)
            )

        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        job_dir = self.jobs_dir / job_name
        if self.force and job_dir.exists():
            shutil.rmtree(job_dir)

        cmd = self._harbor_cmd(agent_import_path=agent_import_path, job_name=job_name)
        env = self._harbor_env()
        log_dir = self.out_dir / "terminus_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{job_name}.stdout.log"
        stderr_path = log_dir / f"{job_name}.stderr.log"
        started = time.time()
        returncode: int | None
        timed_out = False
        # Stream harbor's output straight to files so it can be `tail -f`'d while a
        # job (which can run for many minutes) is in flight.
        with open(stdout_path, "w", encoding="utf-8") as out_f, open(
            stderr_path, "w", encoding="utf-8"
        ) as err_f:
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=str(source_project_path),
                    env=env,
                    stdout=out_f,
                    stderr=err_f,
                    timeout=self.timeout_s,
                    check=False,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                err_f.write(f"\n[harbor run timed out after {self.timeout_s}s]\n")
                returncode = None
                timed_out = True
        duration_s = time.time() - started
        (log_dir / f"{job_name}.meta.json").write_text(
            json.dumps(
                {
                    "agent": agent_name,
                    "agent_import_path": agent_import_path,
                    "command": cmd,
                    "cwd": str(source_project_path),
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "duration_s": duration_s,
                    "job_dir": str(job_dir),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return job_dir

    def _harbor_cmd(self, *, agent_import_path: str, job_name: str) -> list[str]:
        cmd: list[str] = [
            *shlex.split(self.harbor_command),
            "run",
            "--agent-import-path",
            agent_import_path,
            "-d",
            self.dataset,
            "-m",
            self.solver_model,
            "-e",
            self.env,
            "-n",
            str(self.concurrency),
            "--n-attempts",
            str(self.trials),
            "--job-name",
            job_name,
            "--jobs-dir",
            str(self.jobs_dir),
            "-q",
            "-y",
        ]
        if self.reasoning_effort:
            cmd += ["--ak", f"reasoning_effort={self.reasoning_effort}"]
        if self.agent_timeout_multiplier and self.agent_timeout_multiplier > 0:
            cmd += ["--agent-timeout-multiplier", str(self.agent_timeout_multiplier)]
        for instance in self.instances:
            cmd += ["-i", instance.task_id]
        return cmd

    def _harbor_env(self) -> dict[str, str]:
        env = os.environ.copy()
        # litellm's OpenAI-compatible provider reads OPENAI_API_KEY / OPENAI_API_BASE;
        # point them at the configured solver (DeepSeek v4 Pro by default).
        api_key = os.environ.get(self.solver_api_key_env, "")
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        if self.solver_base_url:
            env["OPENAI_API_BASE"] = self.solver_base_url
            env["OPENAI_BASE_URL"] = self.solver_base_url
        env.pop("HARBOR_MODEL", None)
        return env

    def _candidate_agent_import_path(self, candidate: Mapping[str, Any]) -> str:
        extra = candidate.get("extra") if isinstance(candidate.get("extra"), Mapping) else {}
        for key in (
            "terminus_agent_import_path",
            "agent_import_path",
            "import_path",
        ):
            value = candidate.get(key) or extra.get(key)
            if value:
                value = str(value)
                # Normalize the class name like the meta-harness outer loop does.
                if ":" in value:
                    module, _ = value.rsplit(":", 1)
                    return f"{module}:AgentHarness"
                return value
        name = candidate.get("name") or candidate.get("agent_name")
        if name and str(name) not in {DEFAULT_TERMINUS_AGENT_NAME, DEFAULT_TERMINUS_SOURCE_FAMILY}:
            return f"agents.{name}:AgentHarness"
        return self.seed_agent_import_path

    def _candidate_source_path(self, candidate: Mapping[str, Any]) -> Path:
        extra = candidate.get("extra") if isinstance(candidate.get("extra"), Mapping) else {}
        for key in (
            "source_project_path",
            "terminus_source_path",
            "upstream_source_path",
        ):
            value = candidate.get(key) or extra.get(key)
            if value:
                # Becomes harbor's cwd, so it must be absolute.
                return Path(str(value)).expanduser().resolve()
        return self.source_path


def parse_terminus_job_dir(
    job_dir: Path,
    *,
    agent_name: str = DEFAULT_TERMINUS_AGENT_NAME,
    agent_import_path: str = DEFAULT_TERMINUS_SEED_AGENT_IMPORT_PATH,
) -> list[TerminusTrialRun]:
    """Parse per-trial rewards + token metrics from a harbor job directory.

    Harbor lays out a job dir as ``<job>/<task>__<hash>/result.json`` (one such
    directory per attempt; the suffix after ``__`` is a random id, not an
    attempt index) plus a job-level ``<job>/result.json`` that is ignored here
    in favour of the per-trial files (the meta-harness loop does the same — the
    job-level file can be a stale snapshot). A missing, unreadable, or
    rewardless per-trial ``result.json`` counts as ``reward=0``, matching
    harbor's own ``total_passes / total_trials`` metric.
    """

    runs: list[TerminusTrialRun] = []
    if not job_dir.exists() or not job_dir.is_dir():
        return runs

    attempt_counter: dict[str, int] = {}
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        task = trial_dir.name.rsplit("__", 1)[0]
        attempt = attempt_counter.get(task, 0)
        attempt_counter[task] = attempt + 1

        result_file = trial_dir / "result.json"
        reward = 0.0
        agent_metrics: dict[str, Any] = {}
        verifier_returncode: Any = None
        agent_returncode: Any = None
        agent_exception: Any = None
        if result_file.exists():
            try:
                payload = json.loads(result_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            verifier = payload.get("verifier_result") or {}
            rewards = verifier.get("rewards") or {}
            raw_reward = rewards.get("reward")
            if raw_reward is not None:
                try:
                    reward = float(raw_reward)
                except (TypeError, ValueError):
                    reward = 0.0
            agent_result = payload.get("agent_result") or {}
            md = agent_result.get("metadata") or {}
            agent_metrics = {
                "n_input_tokens": agent_result.get("n_input_tokens"),
                "n_output_tokens": agent_result.get("n_output_tokens"),
                "n_cache_tokens": agent_result.get("n_cache_tokens"),
                "cost_usd": agent_result.get("cost_usd"),
                "n_episodes": md.get("n_episodes"),
                "n_api_calls": len(md.get("api_request_times_msec", []) or []),
            }
            verifier_returncode = verifier.get("returncode")
            agent_returncode = agent_result.get("returncode")
            agent_exception = agent_result.get("exception") or md.get("exception")
        runs.append(
            TerminusTrialRun(
                task_id=task,
                attempt=attempt,
                passed=reward > 0,
                score=reward,
                prompt_tokens=int(agent_metrics.get("n_input_tokens") or 0),
                completion_tokens=int(agent_metrics.get("n_output_tokens") or 0),
                cache_tokens=int(agent_metrics.get("n_cache_tokens") or 0),
                metadata={
                    "benchmark": "terminus",
                    "agent": agent_name,
                    "agent_import_path": agent_import_path,
                    "trial_dir": str(trial_dir),
                    "cost_usd": agent_metrics.get("cost_usd"),
                    "n_episodes": agent_metrics.get("n_episodes"),
                    "n_api_calls": agent_metrics.get("n_api_calls"),
                    "agent_returncode": agent_returncode,
                    "verifier_returncode": verifier_returncode,
                    "agent_exception": agent_exception,
                },
            )
        )
    return runs


def run_terminus_frontier(
    *,
    out_dir: Path,
    tasks_path: Path | None = None,
    split: str = "train",
    limit: int = 0,
    source_path: Path = DEFAULT_TERMINUS_SOURCE_PATH,
    dataset: str = DEFAULT_TERMINUS_DATASET,
    env: str = DEFAULT_TERMINUS_ENV,
    harbor_command: str = DEFAULT_TERMINUS_HARBOR_COMMAND,
    solver_model: str = DEFAULT_TERMINUS_SOLVER_MODEL,
    solver_base_url: str = DEFAULT_TERMINUS_SOLVER_BASE_URL,
    solver_api_key_env: str = DEFAULT_TERMINUS_SOLVER_API_KEY_ENV,
    reasoning_effort: str = DEFAULT_TERMINUS_REASONING_EFFORT,
    trials: int = DEFAULT_TERMINUS_TRIALS,
    concurrency: int = DEFAULT_TERMINUS_CONCURRENCY,
    agent_timeout_multiplier: float = DEFAULT_TERMINUS_AGENT_TIMEOUT_MULTIPLIER,
    seed_agent_import_path: str = DEFAULT_TERMINUS_SEED_AGENT_IMPORT_PATH,
    include_secondary_baseline: bool = True,
    secondary_baseline_name: str = DEFAULT_TERMINUS_SECONDARY_BASELINE_NAME,
    secondary_baseline_import_path: str = DEFAULT_TERMINUS_SECONDARY_BASELINE_IMPORT_PATH,
    test_tasks: Sequence[str] | None = None,
    timeout_s: int = DEFAULT_TERMINUS_JOB_TIMEOUT_S,
    dry_run: bool = False,
    force: bool = False,
    pareto_quality_threshold: float = 0.125,
) -> dict[str, Any]:
    """Evaluate the Terminal-Bench 2.0 baseline scaffolds and seed the frontier.

    The primary baseline is Terminus-KIRA (``agents.baseline_kira``); the
    optional secondary baseline is vanilla Terminus-2
    (``agents.baseline_terminus2``), mirroring the meta-harness Phase-0 setup.
    """

    instances = load_terminus_instances(
        tasks_path, split=split, limit=limit, test_tasks=test_tasks
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = TerminusHarborRunner(
        instances=instances,
        out_dir=out_dir,
        source_path=source_path,
        dataset=dataset,
        env=env,
        harbor_command=harbor_command,
        solver_model=solver_model,
        solver_base_url=solver_base_url,
        solver_api_key_env=solver_api_key_env,
        reasoning_effort=reasoning_effort,
        trials=trials,
        concurrency=concurrency,
        agent_timeout_multiplier=agent_timeout_multiplier,
        seed_agent_import_path=seed_agent_import_path,
        timeout_s=timeout_s,
        dry_run=dry_run,
        force=force,
    )

    seeds: list[tuple[str, str, str]] = [
        (DEFAULT_TERMINUS_AGENT_NAME, DEFAULT_TERMINUS_AGENT_NAME, seed_agent_import_path),
    ]
    if include_secondary_baseline and secondary_baseline_import_path:
        seeds.append(
            (secondary_baseline_name, DEFAULT_TERMINUS_AGENT_NAME, secondary_baseline_import_path)
        )

    candidates: list[CandidateResult] = []
    points: list[ParetoPoint] = []
    for candidate_id, agent_name, import_path in seeds:
        candidate = {
            "name": candidate_id,
            "agent_name": agent_name,
            "scaffold_name": agent_name,
            "source_family": DEFAULT_TERMINUS_SOURCE_FAMILY,
            "terminus_agent_import_path": import_path,
            "source_project_path": str(Path(source_path).expanduser()),
        }
        result = runner.evaluate_candidate(
            candidate=candidate,
            candidate_id=candidate_id,
            agent_name=agent_name,
        )
        candidates.append(result)
        points.append(
            ParetoPoint(
                candidate_id=result.candidate_id,
                scaffold_name=result.scaffold_name,
                passrate=result.passrate,
                token_consuming=result.token_consuming,
                avg_token_consuming=result.avg_token_consuming,
                average_score=result.average_score,
                result_path=result.result_path,
                config=result.config,
            )
        )

    frontier_path = out_dir / "pareto_frontier.json"
    save_frontier(frontier_path, points, quality_gap_threshold=pareto_quality_threshold)
    summary = {
        "benchmark": "terminus",
        "target_system": DEFAULT_TERMINUS_AGENT_NAME,
        "dataset": dataset,
        "environment": env,
        "split": split,
        "limit": limit,
        "count": len(instances),
        "solver_model": solver_model,
        "reasoning_effort": reasoning_effort,
        "trials": trials,
        "concurrency": concurrency,
        "dry_run": dry_run,
        "force": force,
        "candidate_count": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
        "pareto_frontier_path": str(frontier_path),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


# ----------------------------------------------------------------------- utils


def _has_multiple_attempts(runs: Iterable[TerminusTrialRun], task_id: str) -> bool:
    seen: set[int] = set()
    for run in runs:
        if run.task_id == task_id:
            seen.add(run.attempt)
            if len(seen) > 1:
                return True
    return len(seen) > 1


def _score_breakdown(task_results: list[TaskResult]) -> dict[str, dict[str, object]]:
    if not task_results:
        return {"all": {"count": 0, "passrate": 0.0, "average_score": 0.0}}
    passes = sum(1 for item in task_results if item.passed)
    return {
        "all": {
            "count": len(task_results),
            "passrate": passes / len(task_results),
            "average_score": sum(item.score for item in task_results) / len(task_results),
        }
    }


def _load_candidate_result(
    result_path: Path,
    *,
    candidate_id: str,
    agent_name: str,
    config: dict[str, Any],
) -> CandidateResult | None:
    if not result_path.exists():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        candidate = CandidateResult.from_dict(payload["candidate"])
    except Exception:  # noqa: BLE001 - treat any read/parse failure as a cache miss
        return None
    if (
        candidate.candidate_id != candidate_id
        or candidate.scaffold_name != agent_name
        or candidate.config != config
    ):
        return None
    return candidate
