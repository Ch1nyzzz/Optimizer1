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
    raw_data_policy: str = "raw LOCOMO data",
    curai_active: bool = False,
    curai_stagnation_count: int = 0,
    curai_stagnation_doc_exists: bool = False,
    current_frontier_passrate: float | None = None,
    current_frontier_average_score: float | None = None,
    current_frontier_best_iter: int | None = None,
    current_base_iter: int | None = None,
    current_base_passrate: float | None = None,
    current_base_average_score: float | None = None,
    trace_harness_dir: Path | None = None,
    diagnoser_report_path: Path | None = None,
    diagnoser_via_subagent: bool = False,
) -> str:
    """Build the proposer prompt for scoped progressive-context runs."""

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
    curai_section = ""
    if curai_active:
        if (
            current_frontier_passrate is not None
            and current_frontier_best_iter is not None
        ):
            avg_clause = (
                f", average_score {current_frontier_average_score:.4f}"
                if current_frontier_average_score is not None
                else ""
            )
            frontier_anchor_clause = (
                f" against the current Pareto frontier (best passrate "
                f"{current_frontier_passrate:.4f}{avg_clause} from "
                f"iter_{current_frontier_best_iter:03d}; your candidate must "
                f"reach passrate ≥ {current_frontier_passrate:.4f} or strictly "
                f"expand average_score on the frontier to count as progress)"
            )
        else:
            frontier_anchor_clause = ""
        if curai_stagnation_doc_exists:
            curai_section = f"""
## Stagnation Forensics (CuraI mode) — Incremental Update

The optimization has stalled for {curai_stagnation_count} consecutive iterations{frontier_anchor_clause}.
Progressive remains at `high` budget. A `stagnation.md` from earlier
iterations of this episode is already at the workspace root. Build on it
incrementally — do NOT redo the full historical diagnosis.

### Phase 1 — Diagnose (incremental)
1. Read `stagnation.md` first to load the existing diagnosis: mechanisms
   already tried, the shared dead-end pattern, the working hypothesis, and
   directions to avoid.
2. Read ONLY the most recent stalled iteration's `diff_digest.md` (the
   latest entry under `reference_iterations/iter_NNN/`) and, if needed,
   that single iteration's `diff.patch`. Do not re-read older iterations'
   diffs unless the new evidence specifically contradicts a claim already
   recorded in `stagnation.md`.
3. Update `stagnation.md` in place: append the new attempted mechanism,
   revise the working hypothesis if the new evidence shifts it, update the
   "directions to avoid" / "directions to explore" lists, and bump the
   stagnation streak. Keep the document under ~3000 words; prune older
   redundant entries by summarizing them.

### Phase 2 — Decide
After updating `stagnation.md`, write `pending_eval.json`. The candidate
must implement a direction NOT already recorded under "directions to
avoid". Add an optional `stagnation_analysis` field to the candidate JSON
pointing at the working hypothesis you are now testing. Reject mechanisms
structurally similar to the recently failed diffs catalogued in
`stagnation.md`.
"""
        else:
            curai_section = f"""
## Stagnation Forensics (CuraI mode) — Initial Diagnosis

The optimization has stalled for {curai_stagnation_count} consecutive iterations{frontier_anchor_clause}.
Progressive escalated to `high` budget; this is the first high-budget
iteration of this stagnation episode. Persist your reasoning to
`stagnation.md` so subsequent iterations can build on it incrementally
without re-reading every diff.

### Phase 1 — Diagnose
Use `candidate_score_table.json`, `best_candidates.json`, and
`evolution_summary.jsonl` to identify the most recent iterations whose
candidates did NOT advance the pareto frontier. For each of those stalled
iterations, read its `diff_digest.md` and then `diff.patch` under
`reference_iterations/iter_NNN/`. Record what mechanism was tried, what the
hypothesis claimed, and how `passrate` / `average_score` actually moved.
Identify the shared dead-end pattern across these stalled diffs.

After completing the diagnosis, write a `stagnation.md` file at the
workspace root using this structure (markdown):

    # Stagnation Forensics
    ## Episode summary
    - Current iteration: {iteration}
    - Stagnation streak: {curai_stagnation_count}
    ## Mechanisms attempted
    - iter_NNN — <name> — <delta passrate / avg_score> — <one-line outcome>
    ## Shared dead-end pattern
    <2-4 sentences>
    ## Working hypothesis
    <your current best theory of WHY the search is stuck>
    ## Directions to avoid
    - <pattern>
    ## Directions to explore
    - <direction with rationale>

### Phase 2 — Decide
Only after writing `stagnation.md` may you write `pending_eval.json`. Add
an optional `stagnation_analysis` field to the candidate JSON summarizing
the shared dead-end and the new mechanism direction this iteration
explores. Reject mechanisms that are structurally similar to the recent
stalled diffs.
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
`diagnoser_report.json` at the workspace root. Read that report
before designing this iteration's candidate. The subagent's full
contract is in `.claude/agents/diagnoser.md`; you do not need to
restate its instructions to it.

The report is a hypothesis input, not ground truth. You may reject
any direction it proposes, but record your reasoning in the
candidate's `hypothesis` field if you do. If the report and the raw
traces contradict each other, follow the traces and explain the
discrepancy.
"""
    elif diagnoser_report_path is not None:
        diagnoser_display = show(diagnoser_report_path)
        diagnoser_section = f"""
## Diagnoser Report (read first)

A diagnoser subagent has already explored this iteration's traces and
source snapshot and produced a structured failure-mode report at
`{diagnoser_display}`. Read it before doing your own investigation; it
exists to convert this iteration from an open-ended optimization step
into a directed fix task.

The report is a JSON object with these top-level fields:

- `summary` — overall diagnosis (1–3 paragraphs).
- `failure_modes[]` — each entry has `label`, `narrative`, `evidence`
  (`kind` is `"trace"` or `"source"`, with workspace-relative `path`,
  `lines`, `excerpt`, `comment`), `hypothesis`, and `directions` (each
  with `summary`, `rationale`, `scope`, `risk`).
- `context_observations[]` — additional facts the diagnoser flagged
  that do not constitute standalone failure modes.
- `open_questions[]` — items the diagnoser could not resolve; verify
  any you depend on before committing to a direction.

The report is a hypothesis input, not ground truth. You may reject any
direction it proposes, but record your reasoning in the candidate's
`hypothesis` field if you do. If the report and the raw traces
contradict each other, follow the traces and explain the discrepancy.
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
Do not read global run directories, global `candidate_results`, repo-root
`src/`, `references/vendor`, or {raw_data_policy}.
{curai_section}

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

    if diagnoser_via_subagent:
        # Claude Code auto-loads <workspace>/CLAUDE.md as the system
        # prompt, so the static role policies are already in context.
        # Avoid duplicating them in the user message.
        return iteration_header
    return iteration_header + role_block
