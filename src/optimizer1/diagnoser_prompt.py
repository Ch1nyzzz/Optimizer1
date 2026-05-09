"""Prompt builder and report schema for the diagnoser subagent.

The diagnoser is invoked once per proposer iteration when ``--diagnose``
is on. Its job is to convert an open-ended optimization step into a
directed fix task by inspecting the workspace traces and source
snapshot, then producing a single ``diagnoser_report.json`` that the
proposer reads as hypothesis input.

The static role / output contract / style guidance lives in
``prompts/diagnoser_system.md`` so it can be edited as a contract
rather than as a Python format string. This module only builds the
per-iteration dynamic header that points at the right run, iteration,
patch base, and trace harness directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from optimizer1.prompts import load_role_prompt


REPORT_FILENAME = "diagnoser_report.json"


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

    The role / contract / schema text comes from
    ``prompts/diagnoser_system.md``. Here we only assemble the dynamic
    "this iteration" header (run id, iteration, patch base clause,
    optional trace harness pointer, report path), then prepend the
    static role text so the agent sees one continuous prompt.
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


REPORT_SCHEMA_EXAMPLE: dict[str, Any] = {
    "summary": "1-3 paragraph overall diagnosis.",
    "failure_modes": [
        {
            "label": "free-text label",
            "narrative": "multi-paragraph explanation tying trace observations to source code.",
            "evidence": [
                {
                    "kind": "trace",
                    "path": "traces/spans/iter_007/cand_a.jsonl",
                    "lines": [42, 58],
                    "excerpt": "verbatim trace fragment",
                    "comment": "why this excerpt is load-bearing",
                },
                {
                    "kind": "source",
                    "path": "source_snapshot/candidate/project_source/src/optimizer1/agent.py",
                    "lines": [120, 145],
                    "excerpt": "verbatim source fragment",
                    "comment": "how this code produces the trace behavior; note diff vs original_project_source if any",
                },
            ],
            "hypothesis": "causal theory in prose, may hedge.",
            "directions": [
                {
                    "summary": "short title for a candidate fix direction",
                    "rationale": "why this would resolve the failure",
                    "scope": "files/functions affected",
                    "risk": "side effects or limits",
                }
            ],
        }
    ],
    "context_observations": [
        "fact discovered during exploration that the proposer should know but is not itself a failure mode."
    ],
    "open_questions": [
        "question diagnoser could not resolve; proposer should verify before depending on it."
    ],
}


REPORT_TOP_LEVEL_KEYS = (
    "summary",
    "failure_modes",
    "context_observations",
    "open_questions",
)


def validate_report(payload: Any) -> tuple[bool, str]:
    """Lightweight schema validation.

    Returns ``(ok, error_message)``. ``error_message`` is empty on
    success. We deliberately accept extra keys and tolerate empty
    optional arrays — the report's narrative quality matters more than
    strict shape.
    """

    if not isinstance(payload, dict):
        return False, "report must be a JSON object"
    if "summary" not in payload or not isinstance(payload["summary"], str):
        return False, "missing or non-string `summary`"
    if not payload["summary"].strip():
        return False, "`summary` is empty"
    failure_modes = payload.get("failure_modes")
    if not isinstance(failure_modes, list) or not failure_modes:
        return False, "`failure_modes` must be a non-empty array"
    for index, mode in enumerate(failure_modes):
        if not isinstance(mode, dict):
            return False, f"failure_modes[{index}] must be an object"
        for required in ("label", "narrative", "hypothesis"):
            value = mode.get(required)
            if not isinstance(value, str) or not value.strip():
                return False, f"failure_modes[{index}].{required} must be a non-empty string"
        evidence = mode.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            return False, f"failure_modes[{index}].evidence must be a non-empty array"
        for j, entry in enumerate(evidence):
            if not isinstance(entry, dict):
                return False, f"failure_modes[{index}].evidence[{j}] must be an object"
            kind = entry.get("kind")
            if kind not in {"trace", "source"}:
                return False, (
                    f"failure_modes[{index}].evidence[{j}].kind must be 'trace' or 'source'"
                )
            if not isinstance(entry.get("path"), str) or not entry["path"].strip():
                return False, f"failure_modes[{index}].evidence[{j}].path must be a non-empty string"
        directions = mode.get("directions")
        if not isinstance(directions, list) or not directions:
            return False, f"failure_modes[{index}].directions must be a non-empty array"
        for j, direction in enumerate(directions):
            if not isinstance(direction, dict):
                return False, f"failure_modes[{index}].directions[{j}] must be an object"
            for required in ("summary", "rationale"):
                value = direction.get(required)
                if not isinstance(value, str) or not value.strip():
                    return False, (
                        f"failure_modes[{index}].directions[{j}].{required} must be a non-empty string"
                    )
    for optional_key in ("context_observations", "open_questions"):
        value = payload.get(optional_key)
        if value is not None and not isinstance(value, list):
            return False, f"`{optional_key}` must be a list when present"
    return True, ""
