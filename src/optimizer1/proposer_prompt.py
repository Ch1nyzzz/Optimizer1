"""Prompt builder for proposer iterations.

The proposer's static contract — role, objective, search space, quality
gate, edit scope, workflow — lives in the per-benchmark skill at
``prompts/skills/<benchmark>/SKILL.md`` and is delivered to the agent
through the system-prompt channel (``--append-system-prompt`` for Claude,
``<workspace>/AGENTS.md`` for Codex). This module assembles only the
per-iteration *user message*: the assignment fields, reference-role and
bandit blocks, the available-files listing, and the ``pending_eval.json``
schema with live path substitutions.
"""

from __future__ import annotations

from pathlib import Path


_GRAPH_COLOURING_TARGETS = {
    "graph_colouring_source",
    "graph_colouring",
    "graphcolouring",
    "graphcolour",
}


def _is_graph_colouring(target_system: str) -> bool:
    return target_system.lower() in _GRAPH_COLOURING_TARGETS


def _optimization_subject(target_system: str) -> str:
    """Return a short phrase for what the proposer is optimizing."""

    normalized = target_system.lower()
    if normalized in {"mini_swe_agent_source", "mini_swe_agent", "minisweagent"}:
        return "source-backed coding agent control loop"
    if _is_graph_colouring(target_system):
        return "source-backed C++ graph-colouring heuristic"
    return "memory layer"


def _candidate_scaffold_name(target_system: str) -> str:
    """Return the scaffold/agent name shown in the candidate JSON example."""

    if target_system.lower().endswith("_source"):
        return target_system
    return f"{target_system}_source"


def _default_source_project_path(source_snapshot_dir: Path, target_system: str) -> str:
    """Return the source path candidates should point at in pending_eval.json."""

    if target_system.lower() in {"mini_swe_agent_source", "mini_swe_agent", "minisweagent"}:
        return f"{source_snapshot_dir}/candidate/upstream_source/mini-swe-agent"
    if _is_graph_colouring(target_system):
        return f"{source_snapshot_dir}/candidate/upstream_source/graph-colouring"
    return f"{source_snapshot_dir}/candidate/project_source"


def build_progressive_proposer_prompt(
    *,
    run_id: str,
    iteration: int,
    run_dir: Path,
    pending_eval_path: Path,
    summaries_dir: Path,
    include_summaries: bool = True,
    reference_iterations_dir: Path,
    generated_dir: Path,
    source_snapshot_dir: Path,
    budget: str,
    reference_iterations: tuple[int, ...],
    target_system: str,
    optimization_directions: tuple[str, ...],
    split: str,
    limit: int,
    selection_policy: str = "progressive",
    bandit_policy: dict[str, object] | None = None,
    benchmark_name: str = "LOCOMO conversational-memory QA",
    current_base_iter: int | None = None,
    current_base_passrate: float | None = None,
    current_base_average_score: float | None = None,
    state_path: Path | None = None,
    organized: bool = False,
    trace_harness_dir: Path | None = None,
) -> str:
    """Build the proposer's per-iteration user message.

    Returns only the iteration assignment: run/iteration metadata,
    reference-role and bandit blocks, the available-files listing, the
    edit scope, and the ``pending_eval.json`` schema. The proposer's
    static contract is delivered separately as the per-benchmark skill
    through the system-prompt channel, so it is never inlined here.
    """

    direction_lines = "\n".join(f"- {line}" for line in optimization_directions)
    focus_section = ""
    if direction_lines:
        focus_section = f"""
## Optimization Focus

You may choose one of these mechanism directions, combine them, or make an
overall system-level redesign:

{direction_lines}
"""
    workspace_dir = run_dir

    def show(path: Path) -> str:
        try:
            return str(path.relative_to(workspace_dir))
        except ValueError:
            return str(path)

    refs = ", ".join(f"iter_{item:03d}" for item in reference_iterations) or "none"
    if selection_policy == "progressive" and budget in {"low", "medium"}:
        best_label = (
            ", ".join(f"iter_{item:03d}" for item in reference_iterations) or "none"
        )
        reference_role_note = (
            f"- Progressive reference roles: best iteration(s): `{best_label}`.\n"
        )
    elif selection_policy == "progressive":
        reference_role_note = (
            "- Progressive reference roles: high budget includes all available raw "
            "reference iterations; use summaries to rank them.\n"
        )
    elif selection_policy == "bandit" and bandit_policy:
        best_iter_list = bandit_policy.get("best_iterations") or []
        best_label = (
            ", ".join(f"iter_{int(item):03d}" for item in best_iter_list)
            if best_iter_list
            else "none"
        )
        reference_role_note = (
            f"- Bandit reference roles: best iteration(s): `{best_label}`.\n"
        )
    elif selection_policy == "random":
        reference_role_note = (
            "- Baseline reference policy: random sample of up to 3 previous "
            "raw iterations; no metric ranking is implied by the selection.\n"
        )
    elif selection_policy == "recent":
        reference_role_note = (
            "- Baseline reference policy: most recent up to 3 previous raw "
            "iterations; no metric ranking is implied by the selection.\n"
        )
    elif selection_policy == "best":
        reference_role_note = (
            "- Baseline reference policy: top-3 previous raw iterations by "
            "train passrate.\n"
        )
    elif selection_policy == "curaii":
        if budget == "low":
            reference_role_note = (
                "- CuraII reference roles: the single best previous iteration "
                "(also the patch base initialised into `project_source/`).\n"
            )
        elif budget == "medium":
            reference_role_note = (
                "- CuraII reference roles: the top-3 previous iterations whose "
                "passrate strictly beats the seed baseline; the chosen patch "
                "base is one of them and is also initialised into "
                "`project_source/`.\n"
            )
        else:
            reference_role_note = (
                "- CuraII reference roles: all previous raw iterations are "
                "available for diagnosis; the chosen patch base is initialised "
                "into `project_source/`.\n"
            )
    else:
        reference_role_note = ""
    bandit_section = ""
    if selection_policy == "bandit":
        policy = bandit_policy or {}

        def listed(name: str) -> str:
            values = policy.get(name)
            if not isinstance(values, (list, tuple)) or not values:
                return "none"
            return ", ".join(f"`{item}`" for item in values[:12])

        bandit_section = f"""
## Bandit Context Policy

This iteration uses online file-utility estimates to suggest where to start.
Read `evolution_summary.jsonl` and `best_candidates.json` (when a `summaries/`
directory is provided) whenever you need to trace cross-iteration patterns or
identify a strong parent to build on.

The hot/other lists below are advisory and reflect historical reads only;
they do not restrict what you may read. If a file under "Other tracked files"
fills a diagnostic gap, read it.

- Hot files to inspect first: {listed("hot_files")}
- Other tracked files (read on demand if they fill a diagnostic gap): {listed("warm_files")}
"""
    refs_json = ", ".join(str(item) for item in reference_iterations)
    pending_eval_display = show(pending_eval_path)
    state_display = show(state_path) if state_path is not None else None
    summaries_display = show(summaries_dir)
    reference_display = show(reference_iterations_dir)
    if include_summaries:
        summaries_assignment_line = f"- Cumulative summaries: `{summaries_display}/`"
        summaries_files_block = (
            f"- `{summaries_display}/evolution_summary.jsonl` — full cumulative event history\n"
            f"  through the previous iteration.\n"
            f"- `{summaries_display}/best_candidates.json` — current passrate/average_score\n"
            f"  quality Pareto frontier candidates."
        )
    else:
        summaries_assignment_line = (
            "- Cumulative summaries: **not provided in this run** — there is no `summaries/` "
            "directory and no cumulative digest in this prompt. Judge prior iterations "
            f"directly from each bundle's `eval/`, `diff.patch`, and `diff_digest.md` under "
            f"`{reference_display}/iter_NNN/`."
        )
        summaries_files_block = (
            "- (no cumulative summary files in this run — inspect the raw iteration bundles "
            f"under `{reference_display}/iter_NNN/` instead)"
        )
    if organized and state_display is not None:
        if include_summaries:
            summaries_assignment_line = (
                f"- State snapshot: `{state_display}`\n"
                f"- Cumulative summaries: `{summaries_display}/`"
            )
            summaries_files_block = (
                f"- `{state_display}` — current optimizer state snapshot generated from RunStore. "
                "Read this first. It is not evidence and not a plan.\n"
                "- RunStore MCP tools — query structured modification, trace, and outcome facts. "
                "Do not open or copy the backing SQLite DB directly.\n"
                f"{summaries_files_block}"
            )
        else:
            summaries_assignment_line = (
                f"- State snapshot: `{state_display}`\n"
                "- Cumulative summaries: **not provided to the proposer in organized mode**."
            )
            summaries_files_block = (
                f"- `{state_display}` — current optimizer state snapshot generated from RunStore. "
                "Read this first. It is not evidence and not a plan.\n"
                "- RunStore MCP tools — query structured modification, trace, and outcome facts. "
                "Do not open or copy the backing SQLite DB directly.\n"
                "- (no cumulative summary files in organized mode)"
            )
    source_snapshot_display = show(source_snapshot_dir)
    generated_display = show(generated_dir)
    optimization_subject = _optimization_subject(target_system)
    candidate_scaffold_name = _candidate_scaffold_name(target_system)
    default_source_project_path = _default_source_project_path(
        Path(source_snapshot_display),
        target_system,
    )
    is_mini_swe_agent = target_system.lower() in {
        "mini_swe_agent_source",
        "mini_swe_agent",
        "minisweagent",
    }
    is_graph_colouring = _is_graph_colouring(target_system)
    if is_mini_swe_agent:
        source_path_note = (
            "`extra.source_project_path` must point to the edited mini-SWE-agent "
            "snapshot under `source_snapshot/candidate/upstream_source/mini-swe-agent`."
        )
    elif is_graph_colouring:
        source_path_note = (
            "`extra.source_project_path` must point to the edited graph-colouring "
            "snapshot under `source_snapshot/candidate/upstream_source/graph-colouring`."
        )
    else:
        source_path_note = (
            "`extra.source_project_path` must point to the edited snapshot project source "
            "when files under `project_source/src/optimizer1` are modified."
        )
    mini_swe_source_note = (
        f"- `{source_snapshot_display}/candidate/upstream_source/mini-swe-agent/` — "
        "primary editable mini-SWE-agent source tree for coding-agent mechanisms.\n"
        if is_mini_swe_agent
        else ""
    )
    graph_colouring_source_note = (
        (
            f"- `{source_snapshot_display}/candidate/upstream_source/graph-colouring/src/algorithms/` — "
            "editable C++ algorithm files. Mutate `evolved.cpp` (the seed delegates "
            "to TabuCol) and freely include / call the other algorithms (`dsatur.h`, "
            "`welsh_powell.h`, `tabu.h`, `simulated_annealing.h`, `genetic.h`, "
            "`exact_solver.h`) to build hybrid heuristics.\n"
            f"- `{source_snapshot_display}/candidate/upstream_source/graph-colouring/src/benchmark_runner.cpp` — "
            "editable dispatch / CLI entry. You MAY register additional algorithm "
            "names; the harness always invokes `--algorithm evolved`, so keep that "
            "entry working.\n"
            f"- `{source_snapshot_display}/candidate/upstream_source/graph-colouring/data/dimacs/` — "
            "read-only DIMACS instances used for evaluation.\n"
        )
        if is_graph_colouring
        else ""
    )
    mini_swe_edit_note = (
        "\nFor mini-SWE-agent candidates, edit "
        f"`{source_snapshot_display}/candidate/upstream_source/mini-swe-agent/**` "
        "for agent control-loop, prompt/config, action parsing, verification, or "
        "submission behavior, and point `extra.source_project_path` at that tree.\n"
        if is_mini_swe_agent
        else ""
    )
    graph_colouring_edit_note = (
        (
            "\nFor graph-colouring candidates, your editable surface is "
            f"`{source_snapshot_display}/candidate/upstream_source/graph-colouring/src/algorithms/**` "
            "and `src/benchmark_runner.cpp`. Do NOT edit `src/io/**` (the CSV "
            "writer is the integrity boundary), the Makefile, or anything outside "
            "the upstream copy. Point `extra.source_project_path` at the "
            "graph-colouring tree.\n\n"
            "Evaluation is lexicographic:\n"
            "1. PRIMARY  — colors_used per instance, lower is strictly better.\n"
            "2. TIEBREAK — runtime_ms, lower wins, but ONLY when colors_used "
            "matches.\n\n"
            "Do not trade more colours for less runtime — that is a regression. "
            "Do not read the chromatic-number metadata at runtime, hardcode a "
            "known-optimal lookup, or short-circuit the colouring search; those "
            "candidates are auto-rejected by the policy scanner.\n"
        )
        if is_graph_colouring
        else ""
    )

    if current_base_iter is not None:
        if current_base_passrate is not None:
            avg_part = (
                f", average_score {current_base_average_score:.4f}"
                if current_base_average_score is not None
                else ""
            )
            base_metric_clause = (
                f" (passrate {current_base_passrate:.4f}{avg_part})"
            )
        else:
            base_metric_clause = ""
        starting_point_block = (
            f"Your patch base is `iter_{current_base_iter:03d}`"
            f"{base_metric_clause}. `{source_snapshot_display}/candidate/project_source/` "
            f"is already initialized to that candidate's source — edit on top of it."
        )
    else:
        starting_point_block = f"""Every iteration starts from the clean source snapshot in
`{source_snapshot_display}/candidate/`. Historical iterations are diagnostic
references only. Do not treat any reference iteration as a source parent and do
not mechanically copy a prior candidate; implement one intentional mechanism
from the clean source."""

    trace_harness_section = ""
    if trace_harness_dir is not None:
        trace_display = show(trace_harness_dir)
        trace_harness_section = (
            "\n"
            f"- `{trace_display}/manifest.json` — trace harness manifest "
            "(benchmark, baseline reference, schema version).\n"
            f"- `{trace_display}/diagnostic/iter_NNN.md` — pre-rendered "
            "per-iteration diff vs baseline; sections are REGRESSED, "
            "PERSISTENT_FAIL, BREAKTHROUGH, plus counts-only STABLE_PASS / "
            "NO_BASELINE. Read this first to spot patterns.\n"
            f"- `{trace_display}/spans/iter_NNN/<candidate>.jsonl` — full "
            "structured traces (one per line; span data is "
            "benchmark-dependent and may be empty). Drill in when the markdown summary "
            "doesn't tell you enough.\n"
        )

    iteration_header = f"""# OptiHarness Proposer — iteration {iteration}

You are optimizing the {optimization_subject} for {benchmark_name}.

## Assignment

- Run id: `{run_id}`
- Target system: `{target_system}`
- Eval split: `{split}`
- Eval limit: `{limit}` (`0` means full split)
{summaries_assignment_line}
- Raw reference iterations: `{reference_display}/` ({refs})
{reference_role_note}
- Writable clean source snapshot: `{source_snapshot_display}/candidate/`
- Generated wrapper directory: `{generated_display}/`
- Required output: `{pending_eval_display}`

{starting_point_block}
{focus_section}
{bandit_section}

## Available Files

{summaries_files_block}
- `{reference_display}/` — raw iteration bundles copied into this workspace for
  detailed diagnosis. Cumulative summaries may mention iterations whose raw
  bundles are not present here.
- `{source_snapshot_display}/candidate/project_source/src/optimizer1/` — editable
  project source for this candidate.
- `{source_snapshot_display}/candidate/original_project_source/src/optimizer1/` —
  clean project source used for diffing and policy checks.
- `{source_snapshot_display}/candidate/upstream_source/` — copied upstream
  source when available.
{mini_swe_source_note}{graph_colouring_source_note}
- `{generated_display}/` — optional importable wrapper modules for this
  iteration.
{trace_harness_section}

## Edit Scope

You may edit only:

- `{source_snapshot_display}/candidate/**`
- `{generated_display}/**`
- `{pending_eval_display}`

All copied project source under
`{source_snapshot_display}/candidate/project_source/src/optimizer1/**` is editable
for this candidate, including scaffolds, base classes, model/prompt helpers,
dynamic-loading helpers, and utils.
{mini_swe_edit_note}{graph_colouring_edit_note}

## Required output for this iteration

Write exactly this JSON file:
`{pending_eval_display}`

Schema:

```json
{{
  "candidates": [
    {{
      "name": "short_unique_name",
      "scaffold_name": "{candidate_scaffold_name}",
      "top_k": 8,
      "window": 1,
      "source_family": "{target_system}",
      "reference_iterations": [{refs_json}],
      "build_tag": "stable_build_identifier",
      "source_snapshot_path": "{source_snapshot_display}",
      "extra": {{
        "source_project_path": "{default_source_project_path}"
      }},
      "hypothesis": "why this should improve passrate and/or average_score",
      "changes": "brief implementation summary"
    }}
  ]
}}
```

Iteration-specific note: {source_path_note}
"""

    return iteration_header
