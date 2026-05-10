"""Prompt builder for proposer iterations.

The static role / objective / quality-gate / edit-scope policies live
in ``prompts/proposer_system.md`` so they read like a contract rather
than a Python format string. This module assembles only the
per-iteration dynamic header (assignment fields, optional diagnoser /
trace harness / curai-stagnation / bandit blocks, and the
pending_eval.json schema with live path substitutions), then appends
the static role document at the end.
"""

from __future__ import annotations

from pathlib import Path

from optimizer1.prompts import load_role_prompt


def _optimization_subject(target_system: str) -> str:
    """Return a short phrase for what the proposer is optimizing."""

    normalized = target_system.lower()
    if normalized in {"mini_swe_agent_source", "mini_swe_agent", "minisweagent"}:
        return "source-backed coding agent control loop"
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
    return f"{source_snapshot_dir}/candidate/project_source"


def build_progressive_proposer_prompt(
    *,
    run_id: str,
    iteration: int,
    run_dir: Path,
    pending_eval_path: Path,
    summaries_dir: Path,
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
    stagnation_active: bool = False,
    stagnation_count: int = 0,
    historian_report_exists: bool = False,
    historian_via_subagent: bool = False,
    current_frontier_passrate: float | None = None,
    current_frontier_average_score: float | None = None,
    current_frontier_best_iter: int | None = None,
    current_base_iter: int | None = None,
    current_base_passrate: float | None = None,
    current_base_average_score: float | None = None,
    trace_harness_dir: Path | None = None,
    diagnoser_via_subagent: bool = False,
    subagent_mode: bool = False,
) -> str:
    """Build the proposer prompt for scoped progressive-context runs.

    ``subagent_mode=True`` selects the Claude Code subagent path: the
    role / workspace constraints are delivered via ``proposer.md`` as
    the subagent's system prompt and ``workspace.md`` as the
    auto-loaded ``<workspace>/CLAUDE.md``, so the user message must
    not duplicate them. Set ``False`` only for the (deprecated) legacy
    prompt-string injection path that concatenates the role document
    onto the user message.
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
    historian_section = ""
    if stagnation_active:
        if (
            current_frontier_passrate is not None
            and current_frontier_best_iter is not None
        ):
            frontier_anchor_clause = (
                f" against the current frontier-best passrate of "
                f"{current_frontier_passrate:.4f} at "
                f"iter_{current_frontier_best_iter:03d} (your candidate "
                f"must strictly exceed it to count as advance)"
            )
        else:
            frontier_anchor_clause = ""
        if historian_via_subagent:
            mode_clause = (
                "Invoke the `historian` subagent via the Task tool"
                if not historian_report_exists
                else (
                    "Invoke the `historian` subagent via the Task tool to "
                    "incrementally update the existing report"
                )
            )
            existing_clause = (
                ""
                if not historian_report_exists
                else (
                    " A previous `historian_report.md` from earlier in "
                    "this episode is already at the workspace root; the "
                    "subagent will update it in place rather than redo "
                    "the full diagnosis."
                )
            )
            historian_section = f"""
## Historian subagent (call first — stalled for {stagnation_count} iters)

The optimization has stalled for {stagnation_count} consecutive iterations{frontier_anchor_clause}.{existing_clause}

Before any other investigation:

1. {mode_clause}. It will read the recent stalled iters' diffs and
   traces, then write `historian_report.md` at the workspace root with
   shared dead-end pattern, working hypothesis, and directions to avoid.
   The historian's full contract is in `.claude/agents/historian.md`;
   you do not need to restate its instructions to it.
2. Read `historian_report.md` after the subagent returns.
3. Combine its **Directions to avoid** with the diagnoser's per-base
   failure modes when designing this iter's candidate. Reject any
   mechanism that matches a listed avoid-pattern.
"""
        else:
            existing_clause = (
                ""
                if not historian_report_exists
                else (
                    " A previous `historian_report.md` from earlier in "
                    "this episode is already at the workspace root."
                )
            )
            historian_section = f"""
## Stagnation context (stalled for {stagnation_count} iters)

The optimization has stalled for {stagnation_count} consecutive iterations{frontier_anchor_clause}.{existing_clause}

Read `historian_report.md` if it exists at the workspace root before
designing this iter's candidate. Combine its **Directions to avoid**
with the diagnoser's per-base failure modes. Reject any mechanism that
matches a listed avoid-pattern.
"""

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
Begin with the compact summaries (`candidate_score_table.json`,
`retrieval_diagnostics_summary.json`, `diff_summary.jsonl`). Read
`evolution_summary.jsonl` and `best_candidates.json` whenever you need to trace
cross-iteration patterns or identify a strong parent to build on.

The hot/other lists below are advisory and reflect historical reads only;
they do not restrict what you may read. If a file under "Other tracked files"
fills a diagnostic gap, read it.

- Hot files to inspect first: {listed("hot_files")}
- Other tracked files (read on demand if they fill a diagnostic gap): {listed("warm_files")}
"""
    refs_json = ", ".join(str(item) for item in reference_iterations)
    pending_eval_display = show(pending_eval_path)
    summaries_display = show(summaries_dir)
    reference_display = show(reference_iterations_dir)
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
    source_path_note = (
        "`extra.source_project_path` must point to the edited mini-SWE-agent "
        "snapshot under `source_snapshot/candidate/upstream_source/mini-swe-agent`."
        if is_mini_swe_agent
        else "`extra.source_project_path` must point to the edited snapshot project source "
        "when files under `project_source/src/optimizer1` are modified."
    )
    mini_swe_source_note = (
        f"- `{source_snapshot_display}/candidate/upstream_source/mini-swe-agent/` — "
        "primary editable mini-SWE-agent source tree for coding-agent mechanisms.\n"
        if is_mini_swe_agent
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
            "structured traces (one per line; spans cover retrieval, "
            "generation, and tools). Drill in when the markdown summary "
            "doesn't tell you enough.\n"
        )

    diagnoser_section = ""
    if diagnoser_via_subagent:
        diagnoser_section = """
## Diagnoser subagent (call first)

Before any other investigation, invoke the `diagnoser` subagent via
the Task tool. It will explore traces and source, then write
`diagnoser_report.md` at the workspace root. Read that report before
designing this iteration's candidate.
"""

    # Legacy path concatenates the agent's role document and the
    # workspace constraints into the user message. The Claude Code
    # subagent path bypasses this entirely (role goes via
    # --append-system-prompt and constraints via the workspace's
    # CLAUDE.md), so it never sees role_block.
    role_block = load_role_prompt("proposer") + "\n" + load_role_prompt("workspace")

    iteration_header = f"""# OptiHarness Proposer — iteration {iteration}

You are optimizing the {optimization_subject} for {benchmark_name}.

## Assignment

- Run id: `{run_id}`
- Target system: `{target_system}`
- Eval split: `{split}`
- Eval limit: `{limit}` (`0` means full split)
- Cumulative summaries: `{summaries_display}/`
- Raw reference iterations: `{reference_display}/` ({refs})
{reference_role_note}
- Writable clean source snapshot: `{source_snapshot_display}/candidate/`
- Generated wrapper directory: `{generated_display}/`
- Required output: `{pending_eval_display}`

{starting_point_block}
{diagnoser_section}
{focus_section}
{bandit_section}

## Available Files

- `{summaries_display}/evolution_summary.jsonl` — full cumulative event history
  through the previous iteration.
- `{summaries_display}/best_candidates.json` — current passrate/average_score
  quality Pareto frontier candidates.
- `{summaries_display}/candidate_score_table.json` — compact metrics for all
  evaluated candidates.
- `{summaries_display}/retrieval_diagnostics_summary.json` — cumulative failure
  and retrieval-pattern summary.
- `{summaries_display}/iteration_index.json` — paths for prior iteration
  artifacts.
- `{summaries_display}/diff_summary.jsonl` — compact source-change records.
- `{reference_display}/` — raw iteration bundles copied into this workspace for
  detailed diagnosis. Cumulative summaries may mention iterations whose raw
  bundles are not present here.
- `{source_snapshot_display}/candidate/project_source/src/optimizer1/` — editable
  project source for this candidate.
- `{source_snapshot_display}/candidate/original_project_source/src/optimizer1/` —
  clean project source used for diffing and policy checks.
- `{source_snapshot_display}/candidate/upstream_source/` — copied upstream
  source when available.
{mini_swe_source_note}
- `{generated_display}/` — optional importable wrapper modules for this
  iteration.
{trace_harness_section}
{historian_section}

## Edit Scope

You may edit only:

- `{source_snapshot_display}/candidate/**`
- `{generated_display}/**`
- `{pending_eval_display}`

All copied project source under
`{source_snapshot_display}/candidate/project_source/src/optimizer1/**` is editable
for this candidate, including scaffolds, base classes, model/prompt helpers,
dynamic-loading helpers, and utils.
{mini_swe_edit_note}

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

---

"""

    if subagent_mode or diagnoser_via_subagent:
        # In subagent mode the role document is loaded as the subagent
        # system prompt and <workspace>/CLAUDE.md provides the project
        # constraints, so the user message must not duplicate them.
        # ``diagnoser_via_subagent`` is kept as an alias for backward
        # compatibility with call sites that signal subagent mode by
        # gating on the diagnoser.
        return iteration_header
    return iteration_header + role_block
