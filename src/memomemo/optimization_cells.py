"""Prompt-guided optimization-cell registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationCell:
    """One prompt-guided optimization direction for a target system."""

    name: str
    target_system: str
    description: str
    focus_functions: tuple[str, ...]
    prompt_guidance: str


MEMGPT_OPTIMIZATION_CELLS = {
    "core_summary": OptimizationCell(
        name="core_summary",
        target_system="memgpt",
        description="Optimize core memory construction and summary/compaction behavior.",
        focus_functions=(
            "_build_core_memory",
            "_build_summary_message",
            "_compile_core_memory",
            "_compile_memory_metadata",
        ),
        prompt_guidance="Focus on how stable memory and compressed history are represented.",
    ),
    "memory_representation": OptimizationCell(
        name="memory_representation",
        target_system="memgpt",
        description="Optimize how conversation turns become recall messages and archival passages.",
        focus_functions=(
            "MemGPTSourceScaffold.build",
            "_build_recall_messages",
            "_build_archival_passages",
        ),
        prompt_guidance=(
            "Focus on message-to-memory transformation, archival chunking, "
            "and memory representation."
        ),
    ),
    "retrieval_policy": OptimizationCell(
        name="retrieval_policy",
        target_system="memgpt",
        description=(
            "Optimize memory-tier mixing, ranking, expansion, deduplication, "
            "and retrieval result formatting."
        ),
        focus_functions=(
            "MemGPTSourceScaffold.retrieve",
            "_hybrid_rank",
            "_expand_recall_indices",
            "_dedupe_hits",
            "_core_hit",
            "_format_archival_result",
            "_format_recall_result",
        ),
        prompt_guidance=(
            "Focus on retrieval policy and evidence assembly, not only scalar "
            "parameter tuning."
        ),
    ),
    "all": OptimizationCell(
        name="all",
        target_system="memgpt",
        description="Global redesign / fusion across all memgpt cells.",
        focus_functions=(),
        prompt_guidance="You may fuse ideas across multiple cells and redesign boundaries if justified.",
    ),
}

MINI_SWE_AGENT_TARGET = "mini_swe_agent_source"

MINI_SWE_AGENT_OPTIMIZATION_CELLS = {
    "issue_context": OptimizationCell(
        name="issue_context",
        target_system=MINI_SWE_AGENT_TARGET,
        description=(
            "Optimize issue understanding, repository exploration, and context selection "
            "before editing."
        ),
        focus_functions=(
            "minisweagent.config.benchmarks.swebench.yaml",
            "DefaultAgent.run",
            "DefaultAgent.step",
            "DefaultAgent.query",
            "Model.format_observation_messages",
        ),
        prompt_guidance=(
            "Focus on helping the agent localize relevant files and preserve concise "
            "state across turns without reading gold patches or scorer artifacts."
        ),
    ),
    "patch_planning": OptimizationCell(
        name="patch_planning",
        target_system=MINI_SWE_AGENT_TARGET,
        description=(
            "Optimize the coding-agent loop for forming, applying, and revising source "
            "patches."
        ),
        focus_functions=(
            "DefaultAgent.execute_actions",
            "minisweagent.models.utils.actions_text",
            "minisweagent.models.utils.actions_toolcall",
            "minisweagent.config.default.yaml",
            "minisweagent.config.benchmarks.swebench.yaml",
        ),
        prompt_guidance=(
            "Focus on disciplined edit sequencing, small diffs, and using observations "
            "to refine the patch rather than broad prompt-length or step-count changes."
        ),
    ),
    "verification_policy": OptimizationCell(
        name="verification_policy",
        target_system=MINI_SWE_AGENT_TARGET,
        description=(
            "Optimize when tests or lightweight checks are run and how failure feedback "
            "is folded back into the next action."
        ),
        focus_functions=(
            "DefaultAgent.query",
            "DefaultAgent.execute_actions",
            "minisweagent.environments.local.LocalEnvironment.execute",
            "minisweagent.environments.docker.DockerEnvironment.execute",
            "minisweagent.config.benchmarks.swebench.yaml",
        ),
        prompt_guidance=(
            "Focus on making verification targeted and interpretable so the agent fixes "
            "regressions without wasting the step budget."
        ),
    ),
    "submission_recovery": OptimizationCell(
        name="submission_recovery",
        target_system=MINI_SWE_AGENT_TARGET,
        description=(
            "Optimize final patch creation, submission detection, and recovery from "
            "LimitsExceeded or malformed final outputs."
        ),
        focus_functions=(
            "DefaultAgent.run",
            "DefaultAgent.serialize",
            "minisweagent.environments.local.LocalEnvironment.execute",
            "minisweagent.environments.docker.DockerEnvironment.execute",
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ),
        prompt_guidance=(
            "Focus on producing an official-eval usable git diff and avoiding empty, "
            "missing, or post-submit-mutated submissions."
        ),
    ),
    "all": OptimizationCell(
        name="all",
        target_system=MINI_SWE_AGENT_TARGET,
        description="Global redesign / fusion across the coding-agent control loop.",
        focus_functions=(),
        prompt_guidance=(
            "You may combine context selection, edit planning, verification, and "
            "submission reliability when the interaction between them is the mechanism."
        ),
    ),
}


def get_target_cells(target_system: str) -> list[OptimizationCell]:
    """Return optimization cells for the requested target system."""

    normalized = target_system.lower()
    if normalized == "memgpt":
        return list(MEMGPT_OPTIMIZATION_CELLS.values())
    if normalized in {MINI_SWE_AGENT_TARGET, "mini_swe_agent", "minisweagent"}:
        return list(MINI_SWE_AGENT_OPTIMIZATION_CELLS.values())
    return []


def get_cell(name: str, target_system: str = "memgpt") -> OptimizationCell:
    """Return one optimization cell by name."""

    normalized = target_system.lower()
    if normalized == "memgpt":
        cells = MEMGPT_OPTIMIZATION_CELLS
    elif normalized in {MINI_SWE_AGENT_TARGET, "mini_swe_agent", "minisweagent"}:
        cells = MINI_SWE_AGENT_OPTIMIZATION_CELLS
    else:
        raise KeyError(f"unknown target system: {target_system}")
    try:
        return cells[name]
    except KeyError as exc:
        raise KeyError(f"unknown {target_system} optimization cell: {name}") from exc
