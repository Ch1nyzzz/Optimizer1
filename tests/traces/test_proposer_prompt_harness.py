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
    assert "traces/rationale/iter_NNN/<candidate>.md" in text
    assert "traces/spans/iter_NNN/<candidate>.jsonl" in text
    assert "traces/index.db" in text
    # Mentions the structured-query CLI subcommands.
    assert "python -m optimizer1.traces task-history" in text
    assert "persistent-failures" in text
    assert "candidate-outcome" in text


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
