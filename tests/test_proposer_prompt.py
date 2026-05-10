from pathlib import Path

from optimizer1.optimization_cells import get_target_cells
from optimizer1.proposer_prompt import build_progressive_proposer_prompt


def test_progressive_prompt_uses_workspace_summaries_and_reference_iterations():
    prompt = build_progressive_proposer_prompt(
        run_id="r",
        iteration=6,
        run_dir=Path("runs/r/proposer_calls/iter_006/workspace"),
        pending_eval_path=Path("runs/r/proposer_calls/iter_006/workspace/pending_eval.json"),
        summaries_dir=Path("runs/r/proposer_calls/iter_006/workspace/summaries"),
        reference_iterations_dir=Path(
            "runs/r/proposer_calls/iter_006/workspace/reference_iterations"
        ),
        generated_dir=Path("runs/r/proposer_calls/iter_006/workspace/generated"),
        source_snapshot_dir=Path("runs/r/proposer_calls/iter_006/workspace/source_snapshot"),
        budget="low",
        reference_iterations=(2, 3),
        target_system="memgpt",
        optimization_directions=("retrieval_policy: Improve evidence ranking.",),
        split="train",
        limit=0,
    )

    assert "summaries/evolution_summary.jsonl" in prompt
    assert "summaries/best_candidates.json" in prompt
    assert "summaries/candidate_score_table.json" in prompt
    assert "summaries/retrieval_diagnostics_summary.json" in prompt
    assert "summaries/diff_summary.jsonl" in prompt
    assert "OptiHarness Proposer" in prompt
    assert "Context budget" not in prompt
    assert "Context scope" not in prompt
    assert '"budget":' not in prompt
    assert "Optimization Focus" in prompt
    assert "mechanism directions" in prompt
    assert "retrieval_policy: Improve evidence ranking." in prompt
    assert "reference_iterations/" in prompt
    assert "iter_002, iter_003" in prompt
    assert "clean source snapshot" in prompt
    assert "diagnostic\nreferences only" in prompt
    assert "source parent" in prompt
    assert "UCB" not in prompt
    # Bandit policy section must not be emitted for non-bandit policies.
    # The role policies in CLAUDE.md may mention "bandit" in passing as
    # one of the optional iteration blocks, so we test the section
    # title rather than the bare word.
    assert "Bandit Context Policy" not in prompt
    assert "parent_candidate_id" not in prompt
    assert '"reference_iterations": [2, 3]' in prompt
    assert "`candidate_results/**`" in prompt
    assert "build/database-construction logic" in prompt
    assert "amem_source_path" not in prompt
    assert "mem0_source_path" not in prompt
    assert "memgpt_source_path" in prompt
    assert "fresh `source_base_dir`" in prompt
    assert "source bases" in prompt
    assert "expensive" in prompt


def test_progressive_prompt_requires_mechanism_changes_not_parameter_only():
    prompt = build_progressive_proposer_prompt(
        run_id="r",
        iteration=7,
        run_dir=Path("runs/r/proposer_calls/iter_007/workspace"),
        pending_eval_path=Path("runs/r/proposer_calls/iter_007/workspace/pending_eval.json"),
        summaries_dir=Path("runs/r/proposer_calls/iter_007/workspace/summaries"),
        reference_iterations_dir=Path(
            "runs/r/proposer_calls/iter_007/workspace/reference_iterations"
        ),
        generated_dir=Path("runs/r/proposer_calls/iter_007/workspace/generated"),
        source_snapshot_dir=Path("runs/r/proposer_calls/iter_007/workspace/source_snapshot"),
        budget="medium",
        reference_iterations=(1, 4, 5),
        target_system="memgpt",
        optimization_directions=(),
        split="train",
        limit=0,
    )

    assert "Parameter changes are allowed only as supporting details" in prompt
    assert "substantive change is only `top_k`, window size, thresholds" in prompt
    assert "Do not reduce recall solely to save tokens" in prompt
    assert "quality Pareto frontier over `passrate` and\n`average_score`" in prompt
    assert "Use gold answers only to classify failure\nmodes" in prompt
    assert "All copied project source under" in prompt
    assert "scaffolds, base classes, model/prompt helpers" in prompt
    assert "exactly one candidate" in prompt
    assert "top_k" in prompt
    assert '"top_k": [4, 8]' not in prompt


def test_default_prompt_uses_neutral_context_description():
    prompt = build_progressive_proposer_prompt(
        run_id="r",
        iteration=3,
        run_dir=Path("runs/r/proposer_calls/iter_003/workspace"),
        pending_eval_path=Path("runs/r/proposer_calls/iter_003/workspace/pending_eval.json"),
        summaries_dir=Path("runs/r/proposer_calls/iter_003/workspace/summaries"),
        reference_iterations_dir=Path(
            "runs/r/proposer_calls/iter_003/workspace/reference_iterations"
        ),
        generated_dir=Path("runs/r/proposer_calls/iter_003/workspace/generated"),
        source_snapshot_dir=Path("runs/r/proposer_calls/iter_003/workspace/source_snapshot"),
        budget="high",
        reference_iterations=(1, 2),
        target_system="memgpt",
        optimization_directions=(),
        split="train",
        limit=0,
        selection_policy="default",
    )

    assert "OptiHarness Proposer" in prompt
    assert "Context budget" not in prompt
    assert "Context scope" not in prompt
    assert '"budget":' not in prompt
    assert "Optimization Focus" not in prompt
    assert "mechanism directions" not in prompt
    assert "Cumulative summaries may mention iterations whose raw\n  bundles are not present here" in prompt


def test_random_recent_prompt_describes_baseline_reference_policy():
    common = {
        "run_id": "r",
        "iteration": 5,
        "run_dir": Path("runs/r/proposer_calls/iter_005/workspace"),
        "pending_eval_path": Path("runs/r/proposer_calls/iter_005/workspace/pending_eval.json"),
        "summaries_dir": Path("runs/r/proposer_calls/iter_005/workspace/summaries"),
        "reference_iterations_dir": Path(
            "runs/r/proposer_calls/iter_005/workspace/reference_iterations"
        ),
        "generated_dir": Path("runs/r/proposer_calls/iter_005/workspace/generated"),
        "source_snapshot_dir": Path("runs/r/proposer_calls/iter_005/workspace/source_snapshot"),
        "budget": "medium",
        "reference_iterations": (2, 3, 4),
        "target_system": "memgpt",
        "optimization_directions": (),
        "split": "train",
        "limit": 0,
    }

    random_prompt = build_progressive_proposer_prompt(
        **common,
        selection_policy="random",
    )
    recent_prompt = build_progressive_proposer_prompt(
        **common,
        selection_policy="recent",
    )

    assert "random sample of up to 3 previous" in random_prompt
    assert "most recent up to 3 previous" in recent_prompt
    assert "best iteration" not in random_prompt
    assert "worst iteration" not in recent_prompt

    best_prompt = build_progressive_proposer_prompt(
        **common,
        selection_policy="best",
    )
    assert "top-3 previous raw iterations by train passrate" in best_prompt
    assert "worst" not in best_prompt


def test_miniswe_prompt_uses_coding_agent_schema_and_focus():
    cells = get_target_cells("mini_swe_agent_source")
    directions = tuple(
        f"{cell.name}: {cell.description} Focus areas: "
        f"{', '.join(cell.focus_functions) if cell.focus_functions else 'all functions'}. "
        f"Guidance: {cell.prompt_guidance}"
        for cell in cells
    )
    prompt = build_progressive_proposer_prompt(
        run_id="r",
        iteration=5,
        run_dir=Path("runs/r/proposer_calls/iter_005/workspace"),
        pending_eval_path=Path("runs/r/proposer_calls/iter_005/workspace/pending_eval.json"),
        summaries_dir=Path("runs/r/proposer_calls/iter_005/workspace/summaries"),
        reference_iterations_dir=Path(
            "runs/r/proposer_calls/iter_005/workspace/reference_iterations"
        ),
        generated_dir=Path("runs/r/proposer_calls/iter_005/workspace/generated"),
        source_snapshot_dir=Path("runs/r/proposer_calls/iter_005/workspace/source_snapshot"),
        budget="medium",
        reference_iterations=(1, 2, 3),
        target_system="mini_swe_agent_source",
        optimization_directions=directions,
        split="train",
        limit=30,
        benchmark_name="SWE-bench coding-agent issue resolution",
    )

    assert "optimizing the source-backed coding agent control loop" in prompt
    assert "memory layer for SWE-bench" not in prompt
    assert "issue_context: Optimize issue understanding" in prompt
    assert "patch_planning: Optimize the coding-agent loop" in prompt
    assert "verification_policy: Optimize when tests" in prompt
    assert "submission_recovery: Optimize final patch creation" in prompt
    assert '"scaffold_name": "mini_swe_agent_source"' in prompt
    assert "mini_swe_agent_source_source" not in prompt
    assert "primary editable mini-SWE-agent source tree" in prompt
    assert "edit `source_snapshot/candidate/upstream_source/mini-swe-agent/**`" in prompt
    assert (
        '"source_project_path": "source_snapshot/candidate/upstream_source/mini-swe-agent"'
        in prompt
    )
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in prompt


def test_stagnation_prompt_invokes_historian_when_streak_active():
    """When stagnation_active=True, the prompt must steer the proposer
    to invoke the historian subagent. When inactive, no stagnation
    section should appear."""

    common = {
        "run_id": "r",
        "iteration": 8,
        "run_dir": Path("runs/r/proposer_calls/iter_008/workspace"),
        "pending_eval_path": Path("runs/r/proposer_calls/iter_008/workspace/pending_eval.json"),
        "summaries_dir": Path("runs/r/proposer_calls/iter_008/workspace/summaries"),
        "reference_iterations_dir": Path(
            "runs/r/proposer_calls/iter_008/workspace/reference_iterations"
        ),
        "generated_dir": Path("runs/r/proposer_calls/iter_008/workspace/generated"),
        "source_snapshot_dir": Path("runs/r/proposer_calls/iter_008/workspace/source_snapshot"),
        "reference_iterations": (1, 2, 3, 4, 5, 6, 7),
        "target_system": "memgpt",
        "optimization_directions": (),
        "split": "train",
        "limit": 0,
        "selection_policy": "pareto",
        "budget": "high",
    }

    stalled_prompt = build_progressive_proposer_prompt(
        **common,
        stagnation_active=True,
        stagnation_count=3,
        historian_via_subagent=True,
    )
    # Section heading mentions the historian and the streak length.
    assert "Historian subagent" in stalled_prompt
    assert "stalled for 3 iters" in stalled_prompt
    assert "stalled for 3 consecutive iterations" in stalled_prompt
    # Steer to Task tool invocation and the historian's report file.
    assert "Task tool" in stalled_prompt
    assert "historian_report.md" in stalled_prompt
    # Combining historian's avoid list with diagnoser's failure modes is the proposer's job.
    assert "Directions to avoid" in stalled_prompt
    assert "diagnoser" in stalled_prompt
    # The legacy Stagnation Forensics block must not be re-emitted.
    assert "Stagnation Forensics (CuraI mode)" not in stalled_prompt
    assert "Initial Diagnosis" not in stalled_prompt
    assert "stagnation.md" not in stalled_prompt

    not_stalled = build_progressive_proposer_prompt(
        **{**common, "budget": "medium"},
        stagnation_active=False,
        stagnation_count=0,
    )
    assert "Historian subagent" not in not_stalled
    assert "Stagnation context" not in not_stalled
    assert "historian_report.md" not in not_stalled


def test_curaii_prompt_announces_patch_base_iter():
    common = {
        "run_id": "r",
        "iteration": 9,
        "run_dir": Path("runs/r/proposer_calls/iter_009/workspace"),
        "pending_eval_path": Path("runs/r/proposer_calls/iter_009/workspace/pending_eval.json"),
        "summaries_dir": Path("runs/r/proposer_calls/iter_009/workspace/summaries"),
        "reference_iterations_dir": Path(
            "runs/r/proposer_calls/iter_009/workspace/reference_iterations"
        ),
        "generated_dir": Path("runs/r/proposer_calls/iter_009/workspace/generated"),
        "source_snapshot_dir": Path("runs/r/proposer_calls/iter_009/workspace/source_snapshot"),
        "reference_iterations": (3, 5, 7),
        "target_system": "memgpt",
        "optimization_directions": (),
        "split": "train",
        "limit": 0,
        "selection_policy": "curaii",
        "budget": "low",
    }

    base_prompt = build_progressive_proposer_prompt(
        **common,
        current_base_iter=4,
        current_base_passrate=0.4125,
        current_base_average_score=0.5453,
    )
    assert "Your patch base is `iter_004`" in base_prompt
    assert "passrate 0.4125" in base_prompt
    assert "average_score 0.5453" in base_prompt
    assert "edit on top of it" in base_prompt
    # The default "Every iteration starts from the clean source snapshot" line
    # must not be emitted when a base iter is supplied.
    assert "Every iteration starts from the clean source snapshot" not in base_prompt
    # CuraII-specific reference-role description
    assert "CuraII reference roles" in base_prompt

    # Without a base iter, behavior must match the historical wording.
    no_base_prompt = build_progressive_proposer_prompt(
        **common,
        current_base_iter=None,
    )
    assert "Your patch base is" not in no_base_prompt
    assert "Every iteration starts from the clean source snapshot" in no_base_prompt


def test_stagnation_prompt_anchors_streak_to_current_frontier_best():
    """Stagnation section announces the historical best the candidate
    must beat to advance."""

    common = {
        "run_id": "r",
        "iteration": 18,
        "run_dir": Path("runs/r/proposer_calls/iter_018/workspace"),
        "pending_eval_path": Path("runs/r/proposer_calls/iter_018/workspace/pending_eval.json"),
        "summaries_dir": Path("runs/r/proposer_calls/iter_018/workspace/summaries"),
        "reference_iterations_dir": Path(
            "runs/r/proposer_calls/iter_018/workspace/reference_iterations"
        ),
        "generated_dir": Path("runs/r/proposer_calls/iter_018/workspace/generated"),
        "source_snapshot_dir": Path("runs/r/proposer_calls/iter_018/workspace/source_snapshot"),
        "reference_iterations": (10, 11, 12, 13, 14, 15, 16, 17),
        "target_system": "memgpt",
        "optimization_directions": (),
        "split": "train",
        "limit": 0,
        "selection_policy": "pareto",
        "budget": "high",
        "stagnation_active": True,
        "stagnation_count": 5,
        "historian_via_subagent": True,
    }

    anchored = build_progressive_proposer_prompt(
        **common,
        current_frontier_passrate=0.4125,
        current_frontier_best_iter=2,
    )
    assert "stalled for 5 consecutive iterations against the current frontier-best passrate" in anchored
    assert "0.4125" in anchored
    assert "iter_002" in anchored
    assert "strictly exceed" in anchored

    fallback = build_progressive_proposer_prompt(**common)
    assert "stalled for 5 consecutive iterations" in fallback
    assert "frontier-best" not in fallback


def test_stagnation_prompt_switches_to_incremental_when_historian_report_exists():
    """When historian_report_exists=True, the prompt should signal an
    incremental update rather than initial diagnosis."""

    common = {
        "run_id": "r",
        "iteration": 12,
        "run_dir": Path("runs/r/proposer_calls/iter_012/workspace"),
        "pending_eval_path": Path("runs/r/proposer_calls/iter_012/workspace/pending_eval.json"),
        "summaries_dir": Path("runs/r/proposer_calls/iter_012/workspace/summaries"),
        "reference_iterations_dir": Path(
            "runs/r/proposer_calls/iter_012/workspace/reference_iterations"
        ),
        "generated_dir": Path("runs/r/proposer_calls/iter_012/workspace/generated"),
        "source_snapshot_dir": Path("runs/r/proposer_calls/iter_012/workspace/source_snapshot"),
        "reference_iterations": (5, 6, 7, 8, 9, 10, 11),
        "target_system": "memgpt",
        "optimization_directions": (),
        "split": "train",
        "limit": 0,
        "selection_policy": "pareto",
        "budget": "high",
    }

    incremental_prompt = build_progressive_proposer_prompt(
        **common,
        stagnation_active=True,
        stagnation_count=7,
        historian_report_exists=True,
        historian_via_subagent=True,
    )
    assert "incrementally update" in incremental_prompt
    assert "previous `historian_report.md`" in incremental_prompt
    assert "already at the workspace root" in incremental_prompt
    # Legacy stagnation.md path must not appear.
    assert "stagnation.md" not in incremental_prompt
    assert "Initial Diagnosis" not in incremental_prompt
    assert "Phase 1 — Diagnose" not in incremental_prompt


def test_bandit_prompt_includes_context_policy_without_leaking_to_default():
    prompt = build_progressive_proposer_prompt(
        run_id="r",
        iteration=4,
        run_dir=Path("runs/r/proposer_calls/iter_004/workspace"),
        pending_eval_path=Path("runs/r/proposer_calls/iter_004/workspace/pending_eval.json"),
        summaries_dir=Path("runs/r/proposer_calls/iter_004/workspace/summaries"),
        reference_iterations_dir=Path(
            "runs/r/proposer_calls/iter_004/workspace/reference_iterations"
        ),
        generated_dir=Path("runs/r/proposer_calls/iter_004/workspace/generated"),
        source_snapshot_dir=Path("runs/r/proposer_calls/iter_004/workspace/source_snapshot"),
        budget="low",
        reference_iterations=(2,),
        target_system="memgpt",
        optimization_directions=("retrieval_policy: Improve evidence ranking.",),
        split="train",
        limit=0,
        selection_policy="bandit",
        bandit_policy={
            "hot_files": ["summaries/candidate_score_table.json"],
            "warm_files": ["summaries/diff_summary.jsonl"],
            "cold_files": [],
            "best_iterations": [3],
        },
    )

    assert "Bandit Context Policy" in prompt
    assert "`candidate_score_table.json`" in prompt
    assert "Hot files to inspect first" in prompt
    assert "`summaries/candidate_score_table.json`" in prompt
    assert "Other tracked files" in prompt
    assert "Cold files to avoid" not in prompt
    assert "Bandit reference roles" in prompt
    assert "iter_003" in prompt
    assert "worst" not in prompt


def _markers_only_in_role_block() -> tuple[str, ...]:
    """A few unambiguous strings that come from proposer.md / workspace.md
    but never appear in any iteration_header. Anchored on these so a
    later prompt-template tweak doesn't silently break the assertion."""

    return (
        "## pending_eval.json conventions",
        "benchmark-scoped and incomplete",
    )


def test_subagent_mode_drops_role_block_from_user_message():
    """In subagent_mode the role/constraints reach the model via
    proposer.md (subagent system prompt) and CLAUDE.md (auto-loaded
    project context), so the user message must not duplicate them.
    Without subagent_mode the legacy injection still concatenates
    role_block at the tail."""

    common = dict(
        run_id="r",
        iteration=4,
        run_dir=Path("runs/r/proposer_calls/iter_004/workspace"),
        pending_eval_path=Path(
            "runs/r/proposer_calls/iter_004/workspace/pending_eval.json"
        ),
        summaries_dir=Path("runs/r/proposer_calls/iter_004/workspace/summaries"),
        reference_iterations_dir=Path(
            "runs/r/proposer_calls/iter_004/workspace/reference_iterations"
        ),
        generated_dir=Path("runs/r/proposer_calls/iter_004/workspace/generated"),
        source_snapshot_dir=Path(
            "runs/r/proposer_calls/iter_004/workspace/source_snapshot"
        ),
        budget="high",
        reference_iterations=(1, 2, 3),
        target_system="memgpt",
        optimization_directions=(),
        split="train",
        limit=0,
        selection_policy="default",
    )

    legacy_prompt = build_progressive_proposer_prompt(**common, subagent_mode=False)
    subagent_prompt = build_progressive_proposer_prompt(**common, subagent_mode=True)

    for marker in _markers_only_in_role_block():
        assert marker in legacy_prompt, (
            f"legacy prompt should still carry role_block marker {marker!r}"
        )
        assert marker not in subagent_prompt, (
            f"subagent_mode prompt must not duplicate role_block marker "
            f"{marker!r} in the user message"
        )
    # The iteration header must still be present in both modes.
    assert "OptiHarness Proposer" in subagent_prompt
    assert "## Assignment" in subagent_prompt
