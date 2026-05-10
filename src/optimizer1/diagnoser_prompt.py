"""Prompt builder and report validator for the diagnoser subagent.

The diagnoser is invoked once per proposer iteration when ``--diagnose``
is on. Its job is to convert an open-ended optimization step into a
directed fix task by inspecting the workspace traces and source
snapshot, then producing a structured ``diagnoser_report.md`` that the
proposer reads as hypothesis input.

The static role / output contract / style guidance lives in
``prompts/diagnoser.md`` so it can be edited as a contract rather
than as a Python format string. This module only builds the
per-iteration dynamic header that points at the right run, iteration,
patch base, and trace harness directory.

Output is **markdown** (``diagnoser_report.md``). ``validate_report``
is a weak structural check (required top-level headings present), not
a strict schema validator — the proposer judges quality, this just
gates obvious malformations.
"""

from __future__ import annotations

from pathlib import Path

from optimizer1.prompts import load_role_prompt


REPORT_FILENAME = "diagnoser_report.md"


REQUIRED_SECTIONS: tuple[str, ...] = (
    "## Summary",
    "## Failure modes",
    "## Context observations",
    "## Open questions",
)


# Sections the diagnoser must NOT include. The proposer is the only
# entity that proposes fix directions; the diagnoser only describes
# failure modes with evidence.
FORBIDDEN_SECTIONS: tuple[str, ...] = (
    "## Directions",
    "## Directions to explore",
    "## Suggested fixes",
    "## Recommended next steps",
    "## Proposed patches",
)


def build_diagnoser_prompt(
    *,
    run_id: str,
    iteration: int,
    workspace_dir: Path,
    target_system: str,
    benchmark_name: str,
    reference_iterations: tuple[int, ...],
    current_base_iter: int | None,
    trace_harness_dir: Path | None,
    report_path: Path,
) -> str:
    """Build the diagnoser system prompt.

    The role / contract text comes from ``prompts/diagnoser.md``. Here
    we only assemble the dynamic "this iteration" header (run id,
    iteration, patch base clause, optional trace harness pointer,
    report path), then prepend the static role text so the agent sees
    one continuous prompt.
    """

    refs = ", ".join(f"iter_{item:03d}" for item in reference_iterations) or "none"
    if current_base_iter is not None:
        base_clause = (
            f"`iter_{current_base_iter:03d}` is the patch base. "
            f"`source_snapshot/candidate/project_source/` has been "
            f"initialized from that iteration's archived source — that "
            f"is the code the proposer is about to edit on top of."
        )
    else:
        base_clause = (
            "There is no curaii-style patch base for this iteration. "
            "`source_snapshot/candidate/project_source/` is the clean "
            "baseline source the proposer will edit."
        )

    trace_section = ""
    if trace_harness_dir is not None:
        try:
            trace_rel = trace_harness_dir.relative_to(workspace_dir)
        except ValueError:
            trace_rel = trace_harness_dir
        trace_section = (
            f"\n## Trace harness for this iteration\n\n"
            f"Structured traces are rooted at `{trace_rel}/`. Read in "
            f"this order before falling back to legacy logs:\n\n"
            f"- `{trace_rel}/manifest.json` — benchmark, baseline "
            f"reference, schema version.\n"
            f"- `{trace_rel}/diagnostic/iter_NNN.md` — pre-rendered "
            f"per-iteration diff vs baseline (REGRESSED / "
            f"PERSISTENT_FAIL / BREAKTHROUGH / counts-only "
            f"STABLE_PASS / NO_BASELINE).\n"
            f"- `{trace_rel}/spans/iter_NNN/<candidate>.jsonl` — full "
            f"structured spans (one per line; retrieval, generation, "
            f"tool calls).\n"
        )

    try:
        report_rel = report_path.relative_to(workspace_dir)
    except ValueError:
        report_rel = report_path

    role_block = load_role_prompt("diagnoser")
    dynamic_header = (
        f"# Diagnoser — iteration {iteration}\n\n"
        f"## Run context\n\n"
        f"- Run id: `{run_id}`\n"
        f"- Iteration: {iteration}\n"
        f"- Target system: `{target_system}`\n"
        f"- Benchmark: {benchmark_name}\n"
        f"- Reference iterations available: {refs}\n\n"
        f"{base_clause}\n"
        f"{trace_section}\n"
        f"## Required output for this iteration\n\n"
        f"Write the report to `{report_rel}` (the harness reads "
        f"`{REPORT_FILENAME}` from the workspace root).\n\n"
        f"---\n\n"
    )

    return dynamic_header + role_block


def validate_report(text: str) -> tuple[bool, str]:
    """Weak structural check: every required section heading is present
    and no forbidden section is. ``failure_modes`` must contain at
    least one ``### <label>`` sub-heading.

    Returns ``(ok, error_message)``. ``error_message`` is empty on
    success. Validation is intentionally weak — content quality is
    judged by the proposer who reads it, not by this validator.
    """

    if not isinstance(text, str) or not text.strip():
        return False, "report is empty"

    missing = [
        section for section in REQUIRED_SECTIONS if section not in text
    ]
    if missing:
        return False, f"missing required section(s): {', '.join(missing)}"

    forbidden_present = [
        section for section in FORBIDDEN_SECTIONS if section in text
    ]
    if forbidden_present:
        return False, (
            f"report contains forbidden section(s): "
            f"{', '.join(forbidden_present)}. Directions are the "
            f"proposer's responsibility, not the diagnoser's."
        )

    # Failure modes must have at least one labeled mode (### sub-heading)
    # between "## Failure modes" and the next "## " heading.
    fm_idx = text.find("## Failure modes")
    if fm_idx < 0:
        return False, "missing required section(s): ## Failure modes"
    after_fm = text[fm_idx + len("## Failure modes") :]
    next_top = after_fm.find("\n## ")
    fm_block = after_fm if next_top < 0 else after_fm[:next_top]
    if "### " not in fm_block:
        return False, (
            "Failure modes section must contain at least one labeled "
            "mode (### <label>)."
        )

    return True, ""
