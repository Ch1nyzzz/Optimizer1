"""Proposer prompt — trace_harness_dir parameter."""

from __future__ import annotations

from pathlib import Path

from optimizer1.proposer_prompt import build_progressive_proposer_prompt


def _build(**overrides) -> str:
    base: dict = dict(
        run_id="r",
        iteration=2,
        run_dir=Path("/tmp/r"),
        pending_eval_path=Path("/tmp/r/p.json"),
        summaries_dir=Path("/tmp/r/summaries"),
        reference_iterations_dir=Path("/tmp/r/refs"),
        generated_dir=Path("/tmp/r/gen"),
        source_snapshot_dir=Path("/tmp/r/src"),
        budget="medium",
        reference_iterations=(1,),
        target_system="memgpt",
        optimization_directions=(),
        split="train",
        limit=10,
    )
    base.update(overrides)
    return build_progressive_proposer_prompt(**base)


def test_prompt_omits_harness_section_when_dir_is_none():
    text = _build()
    assert "trace harness manifest" not in text
    assert "traces/spans/iter_NNN" not in text


def test_prompt_includes_harness_section_when_dir_provided():
    text = _build(trace_harness_dir=Path("/tmp/r/traces"))
    assert "trace harness manifest" in text
    assert "traces/manifest.json" in text
    assert "traces/diagnostic/iter_NNN.md" in text
    assert "traces/spans/iter_NNN/<candidate>.jsonl" in text
    # Rationale layer is removed — its path/table/CLI flag should not
    # appear in the trace-harness section anymore.
    assert "rationale/iter_NNN" not in text
    assert "rationales(iteration" not in text
    assert "--rationale-model" not in text
    # Raw SQL guidance and the SQLite schema dump are removed; proposer
    # discovers query tools via MCP registration instead.
    assert "sqlite3 traces/index.db" not in text
    assert "JOIN diffs d USING" not in text
    assert "Use `cat`, `grep`, `jq`, `sqlite3`, `rg`" not in text
    # MCP tool names are no longer duplicated into the prompt; the MCP
    # server's docstrings are the single source of truth.
    assert "trace_task_history" not in text
    assert "trace_persistent_failures" not in text
    assert "trace_similar" not in text


def test_prompt_uses_relative_path_when_under_run_dir():
    """When trace_harness_dir is below run_dir, the displayed path is
    relative — mirrors the show() helper's behavior used elsewhere."""

    text = _build(
        run_dir=Path("/tmp/r"),
        trace_harness_dir=Path("/tmp/r/traces"),
    )
    # No leading "/tmp/r" — only the relative segment.
    assert "/tmp/r/traces/manifest.json" not in text
    assert "`traces/manifest.json`" in text


def test_prompt_omits_diagnoser_section_when_report_path_is_none():
    text = _build()
    # The dynamic "Diagnoser Report (read first)" header must not be
    # emitted when no report path is supplied. The role policies (in
    # the appended CLAUDE.md / role.md content) may mention the file
    # name when describing what the diagnoser would produce, so we
    # test for the dynamic-section header instead of the bare path.
    assert "Diagnoser Report (read first)" not in text
    assert "Diagnoser subagent (call first)" not in text


def test_prompt_includes_diagnoser_section_when_report_path_provided():
    text = _build(diagnoser_report_path=Path("/tmp/r/diagnoser_report.json"))
    assert "Diagnoser Report (read first)" in text
    # Path is shown relative to run_dir.
    assert "`diagnoser_report.json`" in text
    assert "/tmp/r/diagnoser_report.json" not in text
    # Report schema fields are described so the proposer knows what to
    # consume.
    assert "failure_modes" in text
    assert "open_questions" in text


def test_prompt_uses_subagent_trigger_when_via_subagent_flag_set():
    """Claude proposer + diagnose runs the diagnoser as a Task-tool
    subagent. The prompt must instruct that explicitly *and* must not
    advertise a pre-written report path (none exists yet)."""

    text = _build(diagnoser_via_subagent=True)
    assert "Diagnoser subagent (call first)" in text
    assert "via\nthe Task tool" in text or "via the Task tool" in text
    assert ".claude/agents/diagnoser.md" in text
    # No pre-written report path in this mode.
    assert "Diagnoser Report (read first)" not in text


def test_prompt_omits_role_block_in_subagent_mode():
    """When the diagnoser runs as a subagent, Claude Code auto-loads
    CLAUDE.md from the workspace root, so the static role policies
    must NOT be duplicated into the user message."""

    text = _build(diagnoser_via_subagent=True)
    # These are static-policy markers from prompts/proposer_system.md;
    # they should disappear from the user-message prompt.
    assert "Quality Gate" not in text
    assert "Optimize for expected generalization" not in text


def test_prompt_keeps_role_block_in_classic_mode():
    text = _build()  # diagnoser_via_subagent default = False
    assert "Quality Gate" in text
    assert "Optimize for expected generalization" in text
