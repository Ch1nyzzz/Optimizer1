"""Central benchmark task registry for optimization entry points."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    """Stable metadata used to route a benchmark to its optimizer/base agent."""

    slug: str
    aliases: tuple[str, ...]
    benchmark: str
    base_agent_system: str
    optimizer_kind: str
    default_run_id: str
    description: str

    @property
    def cli_names(self) -> tuple[str, ...]:
        return (self.slug, *self.aliases)


LOCOMO_TASK = BenchmarkTaskSpec(
    slug="locomo",
    aliases=("locomo_subset", "memory_qa"),
    benchmark="locomo",
    base_agent_system="locomo_memory_scaffolds",
    optimizer_kind="locomo_memory",
    default_run_id="locomo_memory_opt",
    description="LOCOMO conversational-memory QA over memory scaffold base agents.",
)

LONGMEMEVAL_TASK = BenchmarkTaskSpec(
    slug="longmemeval",
    aliases=("long-mem-eval", "long_mem_eval", "lme"),
    benchmark="longmemeval",
    base_agent_system=LOCOMO_TASK.base_agent_system,
    optimizer_kind="longmemeval_memory",
    default_run_id="longmemeval_memory_opt",
    description="LongMemEval long-term memory QA over the LOCOMO/MemGPT memory scaffold base agent.",
)

SWEBENCH_TASK = BenchmarkTaskSpec(
    slug="swebench",
    aliases=("swe-bench", "swe_bench", "coding"),
    benchmark="swebench",
    base_agent_system="mini_swe_agent_source",
    optimizer_kind="coding_agent",
    default_run_id="swebench_mini_swe_agent_opt",
    description="SWE-bench-style software engineering tasks over a source-backed mini-SWE-agent coding agent.",
)

BENCHMARK_TASKS: tuple[BenchmarkTaskSpec, ...] = (
    LOCOMO_TASK,
    LONGMEMEVAL_TASK,
    SWEBENCH_TASK,
)

_TASK_BY_NAME = {
    name: task
    for task in BENCHMARK_TASKS
    for name in task.cli_names
}

TASK_CHOICES = tuple(sorted(_TASK_BY_NAME))


def normalize_task_name(value: str | None) -> str:
    """Return the canonical task slug for a CLI task name or alias."""

    raw = (value or LOCOMO_TASK.slug).strip()
    if raw not in _TASK_BY_NAME:
        choices = ", ".join(TASK_CHOICES)
        raise ValueError(f"unknown benchmark task {raw!r}; expected one of: {choices}")
    return _TASK_BY_NAME[raw].slug


def task_spec(value: str | None) -> BenchmarkTaskSpec:
    """Return the task spec for a CLI task name or alias."""

    return _TASK_BY_NAME[normalize_task_name(value)]
