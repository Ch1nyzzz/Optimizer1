"""Tests for the historian subagent prompt + report validator."""

from __future__ import annotations

from pathlib import Path

from optimizer1.historian_prompt import (
    FORBIDDEN_SECTIONS,
    REPORT_FILENAME,
    REQUIRED_SECTIONS,
    build_historian_prompt,
    validate_report,
)


# ---- prompt builder ------------------------------------------------


def _build(**overrides) -> str:
    base: dict = dict(
        run_id="run_xyz",
        iteration=18,
        workspace_dir=Path("/tmp/ws"),
        stagnation_streak=4,
        recent_stalled_iters=(14, 15, 16, 17),
        frontier_iters=(8, 10, 11),
        frontier_best_passrate=0.65,
        frontier_best_iter=10,
        trace_harness_dir=Path("/tmp/ws/traces"),
        previous_report_exists=False,
        report_path=Path("/tmp/ws") / REPORT_FILENAME,
    )
    base.update(overrides)
    return build_historian_prompt(**base)


def test_prompt_lists_run_context_streak_and_frontier():
    text = _build()
    assert "run_xyz" in text
    assert "iteration 18" in text
    assert "Stagnation streak: 4" in text
    assert "iter_014, iter_015, iter_016, iter_017" in text
    assert "iter_008, iter_010, iter_011" in text


def test_prompt_advertises_frontier_best_passrate_and_iter():
    text = _build(frontier_best_passrate=0.6543, frontier_best_iter=10)
    assert "0.6543" in text
    assert "iter_010" in text


def test_prompt_handles_no_frontier_best():
    text = _build(frontier_best_passrate=None, frontier_best_iter=None)
    assert "not yet established" in text


def test_prompt_signals_initial_diagnosis_when_no_previous_report():
    text = _build(previous_report_exists=False)
    assert "first historian invocation" in text
    assert "Write a fresh report" in text


def test_prompt_signals_incremental_update_when_previous_report_exists():
    text = _build(previous_report_exists=True)
    assert "Read it first" in text
    assert "update incrementally" in text


def test_prompt_includes_trace_section_when_harness_dir_present():
    text = _build(trace_harness_dir=Path("/tmp/ws/traces"))
    assert "Trace harness for this episode" in text
    assert "trace MCP tools" in text


def test_prompt_omits_trace_section_when_no_harness_dir():
    text = _build(trace_harness_dir=None)
    assert "Trace harness for this episode" not in text


def test_prompt_renders_report_path_relative_to_workspace():
    text = _build(
        workspace_dir=Path("/tmp/ws"),
        report_path=Path("/tmp/ws") / REPORT_FILENAME,
    )
    assert REPORT_FILENAME in text
    assert "/tmp/ws/historian_report.md" not in text


def test_prompt_includes_role_contract_required_sections():
    """The static role markdown must enumerate the required output
    sections so the agent knows the contract."""

    text = _build()
    for section in REQUIRED_SECTIONS:
        assert section in text, f"role contract missing required section {section}"


def test_prompt_warns_against_proposing_directions():
    text = _build()
    # Role contract must explicitly carve "directions" out of the
    # historian's job and assign them to the proposer.
    assert "do not" in text.lower()
    assert "propose" in text.lower()
    assert "proposer" in text
    # The forbidden section name is mentioned in the role contract so the
    # agent knows not to write it.
    assert "Directions to explore" in text


# ---- validate_report ----------------------------------------------


def _valid_report() -> str:
    return """## Episode summary

The optimization has stalled for 4 iters. Best passrate so far is 0.65 at iter_010.

## Attempts

- **iter_017** (patch_base=iter_010, Δpassrate=-0.03 vs base, Δpassrate=-0.05 vs frontier-best)
  - **Tried**: session-level recency reranker.
  - **Result**: compare_iterations(left=10, right=17) shows two breakthroughs offset by four regressions on bridging-session tasks.
  - **Why it failed**: the recency reweight underweights the canonical bridging session.

## Shared dead-end pattern

Iters 14-17 all reweight by recency without reranking against question features, which underweights the bridging session in tasks t_3 / t_5.

## Working hypothesis

The retrieval layer needs question-conditioned reranking, not unconditioned reweighting. Confidence ~70%; alternative is that the chunk boundary itself is wrong.

## Directions to avoid

- **unconditioned recency reweighting** — repeatedly regresses bridging-session tasks. Evidence: iters 14, 15, 17.
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


def test_validate_report_rejects_forbidden_directions_section():
    """The historian must not propose directions to explore. Adding
    such a section should make the validator reject the report."""

    bad = _valid_report() + (
        "\n## Directions to explore\n\n- try a question-conditioned reranker.\n"
    )
    ok, err = validate_report(bad)
    assert not ok
    assert "Directions to explore" in err
    assert "proposer" in err.lower()


def test_validate_report_rejects_each_forbidden_section():
    for forbidden in FORBIDDEN_SECTIONS:
        bad = _valid_report() + f"\n{forbidden}\n\n- something.\n"
        ok, err = validate_report(bad)
        assert not ok
        assert forbidden in err


def test_required_and_forbidden_sections_do_not_overlap():
    """Sanity: required and forbidden sets must be disjoint."""

    assert set(REQUIRED_SECTIONS).isdisjoint(set(FORBIDDEN_SECTIONS))
