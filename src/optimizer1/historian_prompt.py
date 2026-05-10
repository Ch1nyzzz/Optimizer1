"""Prompt builder and report validator for the historian subagent.

The historian is invoked when the optimization has stalled — the recent
N iterations failed to advance the best passrate. Its job is to map the
dead-end space across the streak (what was tried, why it failed, the
shared pattern) so the proposer can avoid retreading those paths. It
does **not** propose fix directions.

The static role / output contract / style guidance lives in
``prompts/historian.md``. This module assembles the per-stagnation
dynamic header (run id, current iter, streak, frontier members,
report path), then prepends the static role text.

Output is a **markdown** file (``historian_report.md``), not JSON —
``validate_report`` is a weak structural check (required top-level
headings present), not a strict schema validator.
"""

from __future__ import annotations

from pathlib import Path

from optimizer1.prompts import load_role_prompt


REPORT_FILENAME = "historian_report.md"


REQUIRED_SECTIONS: tuple[str, ...] = (
    "## Episode summary",
    "## Attempts",
    "## Shared dead-end pattern",
    "## Working hypothesis",
    "## Directions to avoid",
)


# Sections the historian must NOT include. The proposer is the only
# entity that proposes directions; surfacing these here would let the
# historian leak into the proposer's job.
FORBIDDEN_SECTIONS: tuple[str, ...] = (
    "## Directions to explore",
    "## Suggested fixes",
    "## Recommended next steps",
)


def build_historian_prompt(
    *,
    run_id: str,
    iteration: int,
    workspace_dir: Path,
    stagnation_streak: int,
    recent_stalled_iters: tuple[int, ...],
    frontier_iters: tuple[int, ...],
    frontier_best_passrate: float | None,
    frontier_best_iter: int | None,
    trace_harness_dir: Path | None,
    previous_report_exists: bool,
    report_path: Path,
) -> str:
    """Build the historian system prompt.

    The role / contract text comes from ``prompts/historian.md``. Here we
    only assemble the dynamic "this stagnation episode" header.
    """

    streak = ", ".join(f"iter_{item:03d}" for item in recent_stalled_iters) or "none"
    frontier = (
        ", ".join(f"iter_{item:03d}" for item in frontier_iters) or "none yet"
    )
    if frontier_best_iter is not None and frontier_best_passrate is not None:
        best_clause = (
            f"Best passrate so far: **{frontier_best_passrate:.4f}** at "
            f"`iter_{frontier_best_iter:03d}`."
        )
    else:
        best_clause = "Best passrate so far: not yet established."

    if previous_report_exists:
        mode_clause = (
            f"`{REPORT_FILENAME}` from an earlier iteration in this "
            f"stagnation episode is already at the workspace root. Read "
            f"it first and update incrementally — do not redo the full "
            f"historical diagnosis."
        )
    else:
        mode_clause = (
            f"This is the first historian invocation of this stagnation "
            f"episode; `{REPORT_FILENAME}` does not yet exist. Write a "
            f"fresh report covering the full streak."
        )

    trace_section = ""
    if trace_harness_dir is not None:
        try:
            trace_rel = trace_harness_dir.relative_to(workspace_dir)
        except ValueError:
            trace_rel = trace_harness_dir
        trace_section = (
            f"\n## Trace harness for this episode\n\n"
            f"Structured traces are rooted at `{trace_rel}/`. The trace "
            f"MCP tools query an indexed view of every recorded iter — "
            f"prefer them over reading raw jsonl when answering "
            f"cross-iter questions. Drill into spans only when a "
            f"specific task needs it.\n"
        )

    try:
        report_rel = report_path.relative_to(workspace_dir)
    except ValueError:
        report_rel = report_path

    role_block = load_role_prompt("historian")
    dynamic_header = (
        f"# Historian — iteration {iteration}\n\n"
        f"## Run context\n\n"
        f"- Run id: `{run_id}`\n"
        f"- Iteration: {iteration}\n"
        f"- Stagnation streak: {stagnation_streak}\n"
        f"- Recent stalled iters: {streak}\n"
        f"- Current frontier iters: {frontier}\n"
        f"- {best_clause}\n\n"
        f"{mode_clause}\n"
        f"{trace_section}\n"
        f"## Required output\n\n"
        f"Write the report to `{report_rel}` (the harness reads "
        f"`{REPORT_FILENAME}` from the workspace root).\n\n"
        f"---\n\n"
    )

    return dynamic_header + role_block


def validate_report(text: str) -> tuple[bool, str]:
    """Weak structural check: every required section heading is present
    and no forbidden section is.

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
            f"proposer's responsibility, not the historian's."
        )

    return True, ""
