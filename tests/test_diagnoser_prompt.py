"""Tests for the diagnoser subagent prompt + report schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from optimizer1.diagnoser_prompt import (
    REPORT_FILENAME,
    REPORT_SCHEMA_EXAMPLE,
    build_diagnoser_prompt,
    validate_report,
)


# ---- prompt builder ------------------------------------------------


def _build(**overrides) -> str:
    base: dict = dict(
        run_id="run_xyz",
        iteration=7,
        workspace_dir=Path("/tmp/ws"),
        target_system="memgpt",
        benchmark_name="LOCOMO conversational-memory QA",
        reference_iterations=(3, 5),
        current_base_iter=5,
        trace_harness_dir=Path("/tmp/ws/traces"),
        report_path=Path("/tmp/ws") / REPORT_FILENAME,
    )
    base.update(overrides)
    return build_diagnoser_prompt(**base)


def test_prompt_lists_run_context_and_iteration():
    text = _build()
    assert "iteration 7" in text
    assert "run_xyz" in text
    assert "memgpt" in text
    assert "iter_003, iter_005" in text


def test_prompt_describes_curaii_base_when_provided():
    text = _build(current_base_iter=5)
    assert "iter_005` is the patch base" in text
    # Source snapshot anchor still referenced.
    assert "source_snapshot/candidate/project_source/" in text


def test_prompt_describes_clean_base_when_no_curaii():
    text = _build(current_base_iter=None)
    assert "no curaii-style patch base" in text


def test_prompt_steers_to_mcp_tools_not_sqlite():
    text = _build()
    # Trace harness section is included, but SQL/sqlite3 are explicitly
    # discouraged in favor of the MCP tools registered on the
    # container. The MCP-tool steering text lives in the static role
    # markdown loaded from prompts/diagnoser_system.md.
    assert "Trace harness for this iteration" in text
    assert "MCP tools" in text
    assert "do not write SQL" in text
    assert "sqlite3 traces/index.db" not in text


def test_prompt_omits_trace_section_when_no_harness_dir():
    text = _build(trace_harness_dir=None)
    assert "Trace harness for this iteration" not in text


def test_prompt_advertises_report_path_relative_to_workspace():
    text = _build(
        workspace_dir=Path("/tmp/ws"),
        report_path=Path("/tmp/ws") / REPORT_FILENAME,
    )
    # Report path is rendered relative for unambiguous resolution.
    assert REPORT_FILENAME in text
    assert "/tmp/ws/diagnoser_report.json" not in text


def test_prompt_uses_optimizer1_paths_not_memomemo():
    text = _build()
    # Path examples must be optimizer1 — not the upstream memomemo
    # template they were ported from.
    assert "src/optimizer1/" in text
    assert "src/memomemo/" not in text


# ---- validate_report ----------------------------------------------


def _valid_report() -> dict:
    return {
        "summary": "we keep regressing on date-arithmetic queries.",
        "failure_modes": [
            {
                "label": "date arithmetic regression",
                "narrative": "trace shows the recall ranker drops the relevant date span.",
                "evidence": [
                    {
                        "kind": "trace",
                        "path": "traces/spans/iter_005/cand_a.jsonl",
                        "lines": [12, 24],
                        "excerpt": "...",
                        "comment": "load-bearing",
                    }
                ],
                "hypothesis": "the dedupe step collapses adjacent date spans.",
                "directions": [
                    {
                        "summary": "preserve date spans during dedupe",
                        "rationale": "keep evidence when key matches prior chunk",
                        "scope": "memgpt_scaffold._dedupe_hits",
                        "risk": "minor recall increase",
                    }
                ],
            }
        ],
        "context_observations": [],
        "open_questions": [],
    }


def test_validate_report_accepts_well_formed_payload():
    ok, err = validate_report(_valid_report())
    assert ok, err
    assert err == ""


def test_validate_report_rejects_non_object():
    ok, err = validate_report([])
    assert not ok
    assert "JSON object" in err


def test_validate_report_requires_summary():
    payload = _valid_report()
    payload["summary"] = ""
    ok, err = validate_report(payload)
    assert not ok
    assert "summary" in err


def test_validate_report_requires_at_least_one_failure_mode():
    payload = _valid_report()
    payload["failure_modes"] = []
    ok, err = validate_report(payload)
    assert not ok
    assert "failure_modes" in err


def test_validate_report_requires_evidence_per_failure_mode():
    payload = _valid_report()
    payload["failure_modes"][0]["evidence"] = []
    ok, err = validate_report(payload)
    assert not ok
    assert "evidence" in err


def test_validate_report_requires_known_evidence_kind():
    payload = _valid_report()
    payload["failure_modes"][0]["evidence"][0]["kind"] = "guess"
    ok, err = validate_report(payload)
    assert not ok
    assert "kind" in err


def test_validate_report_requires_directions():
    payload = _valid_report()
    payload["failure_modes"][0]["directions"] = []
    ok, err = validate_report(payload)
    assert not ok
    assert "directions" in err


def test_validate_report_tolerates_extra_keys():
    payload = _valid_report()
    payload["extra_key"] = "ignored"
    ok, err = validate_report(payload)
    assert ok, err


def test_schema_example_is_itself_invalid_placeholder():
    """The example is documentation, not a passing fixture — its
    `evidence.path` placeholders are valid strings, so this just
    sanity-checks that calling the validator on the example does not
    blow up."""

    ok, err = validate_report(REPORT_SCHEMA_EXAMPLE)
    # Example is well-formed enough to pass.
    assert ok, err
