"""Tests for the diagnoser subagent prompt + report validator."""

from __future__ import annotations

from pathlib import Path

from optimizer1.diagnoser_prompt import (
    FORBIDDEN_SECTIONS,
    REPORT_FILENAME,
    REQUIRED_SECTIONS,
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
    assert "source_snapshot/candidate/project_source/" in text


def test_prompt_describes_clean_base_when_no_curaii():
    text = _build(current_base_iter=None)
    assert "no curaii-style patch base" in text


def test_prompt_steers_to_mcp_tools_not_sqlite():
    text = _build()
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
    assert REPORT_FILENAME in text
    assert REPORT_FILENAME.endswith(".md")
    assert "/tmp/ws/diagnoser_report.md" not in text  # relative, not absolute


def test_prompt_advertises_markdown_report_not_json():
    text = _build()
    assert "diagnoser_report.md" in text
    assert "diagnoser_report.json" not in text


def test_prompt_uses_optimizer1_paths_not_memomemo():
    text = _build()
    assert "src/optimizer1/" in text
    assert "src/memomemo/" not in text


def test_prompt_role_contract_lists_required_sections():
    text = _build()
    for section in REQUIRED_SECTIONS:
        assert section in text, f"role contract missing required section {section}"


def test_prompt_role_contract_forbids_directions():
    text = _build()
    # Diagnoser must not propose directions; the role contract spells
    # this out so the agent does not emit such a section.
    assert "do not propose" in text.lower() or "must NOT" in text or "must not" in text
    assert "proposer" in text


# ---- validate_report ----------------------------------------------


def _valid_report() -> str:
    return """## Summary

The recall pipeline is dropping date-anchored chunks during dedup, so
date-arithmetic queries lose their bridging session.

## Failure modes

### date arithmetic regression

**Narrative**

Trace shows the recall ranker drops the relevant date span. The dedupe
step in `memgpt_scaffold.py` collapses adjacent date spans before they
reach the reranker.

**Evidence**

- `trace` `traces/spans/iter_005/cand_a.jsonl:12-24` — verbatim trace excerpt — load-bearing.
- `source` `source_snapshot/candidate/project_source/src/optimizer1/scaffolds/memgpt_scaffold.py:120-145` — verbatim source excerpt — collapses the date span.

**Hypothesis**

The dedupe step's similarity threshold is too aggressive for spans
that share calendar tokens. ~70% confident.

## Context observations

- The reranker only sees post-dedup chunks; any signal lost here is unrecoverable downstream.

## Open questions

- Is the dedupe threshold configurable, or hard-coded?
"""


def test_validate_report_accepts_well_formed_markdown():
    ok, err = validate_report(_valid_report())
    assert ok, err
    assert err == ""


def test_validate_report_rejects_empty_input():
    ok, err = validate_report("")
    assert not ok
    assert "empty" in err


def test_validate_report_rejects_non_string():
    ok, err = validate_report(None)  # type: ignore[arg-type]
    assert not ok


def test_validate_report_requires_each_section():
    for section in REQUIRED_SECTIONS:
        broken = _valid_report().replace(section, "## OTHER")
        ok, err = validate_report(broken)
        assert not ok
        assert section in err, f"validator did not flag missing {section}"


def test_validate_report_requires_at_least_one_failure_mode_label():
    # Strip the "### date arithmetic regression" sub-heading.
    bad = _valid_report().replace("### date arithmetic regression\n", "")
    ok, err = validate_report(bad)
    assert not ok
    assert "labeled mode" in err or "###" in err


def test_validate_report_rejects_forbidden_directions_section():
    """The diagnoser must not propose fix directions. Adding such a
    section should make the validator reject the report."""

    bad = _valid_report() + (
        "\n## Directions\n\n- preserve date spans during dedupe\n"
    )
    ok, err = validate_report(bad)
    assert not ok
    assert "Directions" in err
    assert "proposer" in err.lower()


def test_validate_report_rejects_each_forbidden_section():
    for forbidden in FORBIDDEN_SECTIONS:
        bad = _valid_report() + f"\n{forbidden}\n\n- something.\n"
        ok, err = validate_report(bad)
        assert not ok, f"validator missed forbidden section {forbidden}"
        # The section name (or its prefix when overlap exists) must
        # appear in the error.
        assert forbidden.split(" — ")[0] in err or forbidden.lstrip("# ").split()[0] in err


def test_required_and_forbidden_sections_do_not_overlap():
    assert set(REQUIRED_SECTIONS).isdisjoint(set(FORBIDDEN_SECTIONS))


def test_report_filename_is_markdown_not_json():
    """Hard-switch verification: filename suffix is .md."""

    assert REPORT_FILENAME.endswith(".md")
    assert REPORT_FILENAME == "diagnoser_report.md"
