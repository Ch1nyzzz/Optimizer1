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


TERMINUS_TARGET = "terminus_kira_source"

TERMINUS_OPTIMIZATION_CELLS = {
    "prompt_and_bootstrap": OptimizationCell(
        name="prompt_and_bootstrap",
        target_system=TERMINUS_TARGET,
        description=(
            "Optimize the system prompt and the initial context handed to the agent "
            "— including bootstrapping a snapshot of the sandbox environment before "
            "the agent loop starts."
        ),
        focus_functions=(
            "AgentHarness.run",
            "Terminus2._get_prompt_template_path",
            "Terminus2._get_completion_confirmation_message",
            "prompt-templates/terminus-kira.txt",
        ),
        prompt_guidance=(
            "Focus on what the agent sees before and during the run: a richer system "
            "prompt, an environment snapshot (cwd, file listing, available languages / "
            "tools / package managers, memory) injected into the initial message, or a "
            "better completion-confirmation checklist. General guidance only — no "
            "task-specific hints, never reference task names."
        ),
    ),
    "tool_interface": OptimizationCell(
        name="tool_interface",
        target_system=TERMINUS_TARGET,
        description=(
            "Optimize the native tool schema and the LLM call — tool definitions, new "
            "tools, reasoning effort, retries, parsing of tool calls into commands."
        ),
        focus_functions=(
            "Terminus2._call_llm_with_tools",
            "Terminus2._parse_tool_calls",
            "Terminus2._extract_tool_calls",
            "Terminus2._handle_llm_interaction",
            "TOOLS",
        ),
        prompt_guidance=(
            "Focus on the agent's action interface: clearer or additional tools, "
            "stricter argument validation, smarter retry/backoff on transient errors, "
            "or reasoning-effort tuning. One mechanism per candidate."
        ),
    ),
    "command_execution": OptimizationCell(
        name="command_execution",
        target_system=TERMINUS_TARGET,
        description=(
            "Optimize how commands run on the terminal and how their output is fed "
            "back — polling, marker-based early completion, output truncation, image "
            "reads."
        ),
        focus_functions=(
            "Terminus2._execute_commands",
            "Terminus2._limit_output_length",
            "Terminus2._execute_image_read",
        ),
        prompt_guidance=(
            "Focus on the execute→observe loop: adaptive command durations, early-exit "
            "polling, smarter truncation that preserves the signal-bearing tail, or "
            "handling of non-text artifacts. Avoid pure scalar tuning of timeouts."
        ),
    ),
    "episode_control": OptimizationCell(
        name="episode_control",
        target_system=TERMINUS_TARGET,
        description=(
            "Optimize the episode loop, context summarization on overflow, proactive "
            "summarization, and handoff between summarized segments."
        ),
        focus_functions=(
            "Terminus2._run_agent_loop",
            "Terminus2._summarize_context",
            "Terminus2._check_proactive_summarization",
            "Terminus2._unwind_messages_to_free_tokens",
        ),
        prompt_guidance=(
            "Focus on long-horizon control: when to summarize, what to keep across a "
            "summarization boundary, how to structure the episode loop, and how to "
            "recover from context-length and output-length errors."
        ),
    ),
    "all": OptimizationCell(
        name="all",
        target_system=TERMINUS_TARGET,
        description="Global redesign / fusion across the Terminus agent control loop.",
        focus_functions=(),
        prompt_guidance=(
            "You may combine prompt/bootstrap, tool interface, command execution, and "
            "episode control when the interaction between them is the mechanism. Still "
            "one falsifiable hypothesis per candidate."
        ),
    ),
}


GRAPH_COLOURING_TARGET = "graph_colouring_source"

GRAPH_COLOURING_OPTIMIZATION_CELLS = {
    "heuristic_core": OptimizationCell(
        name="heuristic_core",
        target_system=GRAPH_COLOURING_TARGET,
        description=(
            "Optimize the initial colouring strategy that produces the working "
            "palette before local search."
        ),
        focus_functions=(
            "colour_with_evolved",
            "colour_with_dsatur",
            "colour_with_welsh_powell",
            "src/algorithms/dsatur.cpp",
            "src/algorithms/welsh_powell.cpp",
        ),
        prompt_guidance=(
            "Focus on the initial assignment: degree-saturation orderings, "
            "Recursive Largest First, randomised tie-breaks, or independent-set "
            "extraction as a warm start. The tabu/SA inner loops are downstream — "
            "your job is to hand them a smaller starting palette."
        ),
    ),
    "local_search": OptimizationCell(
        name="local_search",
        target_system=GRAPH_COLOURING_TARGET,
        description=(
            "Optimize the iterative repair loop that drives palette-size "
            "reduction inside `evolved` (TabuCol, SA, or a successor)."
        ),
        focus_functions=(
            "colour_with_evolved",
            "colour_with_tabu",
            "colour_with_simulated_annealing",
            "src/algorithms/tabu.cpp",
            "src/algorithms/simulated_annealing.cpp",
        ),
        prompt_guidance=(
            "Focus on the inner search: tabu tenure schedules, conflict-driven "
            "neighbourhood restriction, partition-based moves, frequency-based "
            "diversification, aspiration criteria, restart policy. Avoid pure "
            "scalar tuning — change the move space or the acceptance rule."
        ),
    ),
    "hybridization": OptimizationCell(
        name="hybridization",
        target_system=GRAPH_COLOURING_TARGET,
        description=(
            "Combine multiple strategies inside `evolved` — warm-start, ensemble, "
            "post-processing, density-aware dispatch."
        ),
        focus_functions=(
            "colour_with_evolved",
            "src/algorithms/evolved.cpp",
            "build_algorithm_table",
        ),
        prompt_guidance=(
            "Focus on the composition: DSatur warm-start feeding TabuCol, "
            "Welsh-Powell post-shrink, path-relinking between TabuCol restarts, "
            "HEA-style population on top of the existing search, density-aware "
            "switching between solvers. Treat the upstream algorithms as a "
            "library and implement the orchestrator."
        ),
    ),
    "all": OptimizationCell(
        name="all",
        target_system=GRAPH_COLOURING_TARGET,
        description="Global redesign of the evolved heuristic.",
        focus_functions=(),
        prompt_guidance=(
            "You may reorganize the heuristic across warm-start, inner search, "
            "and post-processing when the interaction between them is the "
            "mechanism."
        ),
    ),
}


def _is_graph_colouring_target(target: str) -> bool:
    return target in {
        GRAPH_COLOURING_TARGET,
        "graph_colouring",
        "graphcolouring",
        "graphcolour",
    }


def _is_terminus_target(target: str) -> bool:
    return target in {
        TERMINUS_TARGET,
        "terminus",
        "terminus_kira",
        "terminuskira",
        "terminal_bench",
        "terminalbench",
    }


def get_target_cells(target_system: str) -> list[OptimizationCell]:
    """Return optimization cells for the requested target system."""

    normalized = target_system.lower()
    if normalized == "memgpt":
        return list(MEMGPT_OPTIMIZATION_CELLS.values())
    if normalized in {MINI_SWE_AGENT_TARGET, "mini_swe_agent", "minisweagent"}:
        return list(MINI_SWE_AGENT_OPTIMIZATION_CELLS.values())
    if _is_graph_colouring_target(normalized):
        return list(GRAPH_COLOURING_OPTIMIZATION_CELLS.values())
    if _is_terminus_target(normalized):
        return list(TERMINUS_OPTIMIZATION_CELLS.values())
    return []


def get_cell(name: str, target_system: str = "memgpt") -> OptimizationCell:
    """Return one optimization cell by name."""

    normalized = target_system.lower()
    if normalized == "memgpt":
        cells = MEMGPT_OPTIMIZATION_CELLS
    elif normalized in {MINI_SWE_AGENT_TARGET, "mini_swe_agent", "minisweagent"}:
        cells = MINI_SWE_AGENT_OPTIMIZATION_CELLS
    elif _is_graph_colouring_target(normalized):
        cells = GRAPH_COLOURING_OPTIMIZATION_CELLS
    elif _is_terminus_target(normalized):
        cells = TERMINUS_OPTIMIZATION_CELLS
    else:
        raise KeyError(f"unknown target system: {target_system}")
    try:
        return cells[name]
    except KeyError as exc:
        raise KeyError(f"unknown {target_system} optimization cell: {name}") from exc
