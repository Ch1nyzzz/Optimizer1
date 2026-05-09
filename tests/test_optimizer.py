import json
from pathlib import Path
from types import SimpleNamespace

from memomemo.benchmark_workspaces import (
    LOCOMO_WORKSPACE_SPEC,
    MINIMAL_BENCHMARK_PACKAGE_INIT,
)
from memomemo import optimizer as optimizer_module
from memomemo.locomo_optimizer import LocomoOptimizer, LocomoOptimizerConfig
from memomemo.optimizer import _single_top_k
from memomemo.schemas import CandidateResult
import pytest


def test_default_proposer_evaluates_only_one_candidate(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))

    def fake_run_code_agent_prompt(*args, **kwargs):
        captured["prompt"] = args[0]
        optimizer.pending_eval_path.write_text(
            json.dumps(
                {
                    "candidates": [
                        {"name": "first", "scaffold_name": "bm25"},
                        {"name": "second", "scaffold_name": "bm25"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, timed_out=False, stderr="")

    captured = {}

    def fake_evaluate(iteration, proposed, examples):
        captured["iteration"] = iteration
        captured["proposed"] = proposed
        captured["examples"] = examples
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    result = optimizer._run_default_proposer_iteration(1, examples=[])

    assert result == []
    assert captured["iteration"] == 1
    assert captured["proposed"][0]["name"] == "first"
    assert captured["proposed"][0]["scaffold_name"] == "bm25"
    assert captured["proposed"][0]["budget"] == "high"
    assert captured["proposed"][0]["reference_iterations"] == []
    assert captured["examples"] == []
    assert "OptiHarness Proposer" in captured["prompt"]
    assert "Context budget" not in captured["prompt"]
    assert "Context scope" not in captured["prompt"]
    assert '"budget":' not in captured["prompt"]
    assert "Optimization Focus" not in captured["prompt"]
    assert "mechanism directions" not in captured["prompt"]
    snapshot = tmp_path / "proposer_calls" / "iter_001" / "source_snapshot" / "candidate"
    assert (snapshot / "project_source" / "src" / "memomemo" / "dynamic.py").exists()
    assert (snapshot / "project_source" / "src" / "memomemo" / "scaffolds" / "base.py").exists()
    assert not (snapshot / "project_source" / "src" / "memomemo" / "optimizer.py").exists()
    assert (snapshot / "upstream_source" / "MemGPT" / "letta" / "schemas" / "memory.py").exists()
    assert "candidate_count_adjusted" in optimizer.summary_path.read_text(
        encoding="utf-8"
    )


def test_default_proposer_retries_when_pending_eval_missing(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_sandbox="none",
        )
    )
    prompts = []

    def fake_run_code_agent_prompt(prompt, **kwargs):
        prompts.append(prompt)
        if len(prompts) == 2:
            (kwargs["cwd"] / "pending_eval.json").write_text(
                json.dumps({"candidates": [{"name": "first", "scaffold_name": "bm25"}]}),
                encoding="utf-8",
            )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            metrics={},
            usage=None,
            tool_access={},
        )

    captured = {}

    def fake_evaluate(iteration, proposed, examples):
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer._run_default_proposer_iteration(1, examples=[])

    assert len(prompts) == 2
    assert "Required Repair" in prompts[1]
    assert captured["proposed"][0]["name"] == "first"
    summary = optimizer.summary_path.read_text(encoding="utf-8")
    assert "proposer_missing_pending_retry" in summary


def test_default_proposer_accepts_top_level_candidate_list(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_sandbox="none",
        )
    )
    captured = {}

    def fake_run_code_agent_prompt(prompt, **kwargs):
        (kwargs["cwd"] / "pending_eval.json").write_text(
            json.dumps([{"name": "first", "scaffold_name": "bm25"}]),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            metrics={},
            usage=None,
            tool_access={},
        )

    def fake_evaluate(iteration, proposed, examples):
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer._run_default_proposer_iteration(1, examples=[])

    assert captured["proposed"][0]["name"] == "first"
    pending = json.loads(optimizer.pending_eval_path.read_text(encoding="utf-8"))
    assert pending["candidates"][0]["name"] == "first"
    assert pending["candidates"][0]["scaffold_name"] == "bm25"


def test_optimizer_rejects_non_opencode_proposer_agent(tmp_path):
    with pytest.raises(ValueError, match="Only the 'opencode' proposer agent"):
        LocomoOptimizer(
            LocomoOptimizerConfig(
                run_id="r",
                out_dir=tmp_path,
                proposer_agent="codex",
            )
        )


def test_optimizer_can_disable_default_docker_sandbox(tmp_path):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_sandbox="none",
        )
    )

    assert optimizer._proposer_sandbox_config() is None


@pytest.mark.parametrize(
    "selection_policy",
    ["progressive", "bandit", "random", "recent", "best"],
)
def test_adaptive_selection_policies_require_docker_sandbox(tmp_path, selection_policy):
    with pytest.raises(ValueError, match="requires --proposer-sandbox docker"):
        LocomoOptimizer(
            LocomoOptimizerConfig(
                run_id="r",
                out_dir=tmp_path,
                selection_policy=selection_policy,
                proposer_sandbox="none",
                proposer_docker_image="memo-proposer:test",
            )
        )


def test_optimizer_uses_opencode_default_docker_image(tmp_path):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_sandbox="docker",
            proposer_docker_image="",
        )
    )

    sandbox = optimizer._proposer_sandbox_config()

    assert sandbox is not None
    assert sandbox.docker_image == "docker-opencode:latest"


def test_optimizer_can_run_opencode_proposer_agent(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_agent="opencode",
            opencode_model="anthropic/claude-test",
        )
    )
    captured = {}

    def fake_run_code_agent_prompt(prompt, **kwargs):
        captured.update(kwargs)
        optimizer.pending_eval_path.write_text(
            json.dumps({"candidates": [{"name": "first", "scaffold_name": "bm25"}]}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            metrics={},
            usage=None,
            tool_access={},
        )

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", lambda iteration, proposed, examples: [])

    optimizer._run_default_proposer_iteration(1, examples=[])

    assert captured["agent"] == "opencode"
    assert captured["model"] == "anthropic/claude-test"
    assert captured["sandbox"].kind == "docker"
    assert captured["sandbox"].docker_image == "docker-opencode:latest"


@pytest.mark.parametrize(
    "selection_policy",
    ["progressive", "bandit", "random", "recent", "best", "curai", "curaii"],
)
def test_opencode_proposer_rejects_non_default_selection_policy(
    tmp_path, monkeypatch, selection_policy
):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_agent="opencode",
            selection_policy=selection_policy,
            proposer_docker_image="memo-proposer:test",
        )
    )
    monkeypatch.setattr(optimizer, "_load_examples", lambda: [])
    with pytest.raises(ValueError, match="opencode proposer currently supports"):
        optimizer.run()


def test_optimizer_rejects_baseline_with_mismatched_count(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            iterations=0,
            limit=0,
            baseline_dir=tmp_path / "baseline",
            scaffolds=("memgpt_source",),
        )
    )
    baseline = CandidateResult(
        candidate_id="memgpt_source_top12",
        scaffold_name="memgpt_source",
        passrate=0.4,
        average_score=0.5,
        token_consuming=100,
        avg_token_consuming=10,
        avg_prompt_tokens=8,
        avg_completion_tokens=2,
        count=1,
        config={"top_k": 12, "extra": {"source_family": "memgpt"}},
        result_path="baseline.json",
    )

    monkeypatch.setattr(optimizer, "_load_examples", lambda: [object(), object()])
    monkeypatch.setattr(
        optimizer_module,
        "load_baseline_candidates",
        lambda *args, **kwargs: [baseline.to_dict()],
    )

    with pytest.raises(ValueError, match="Baseline candidate count does not match"):
        optimizer.run()


def test_run_writes_initial_trace_slices(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            iterations=0,
            proposer_docker_image="memo-proposer:test",
        )
    )
    result_path = tmp_path / "candidate_results" / "seed.json"
    candidate = CandidateResult(
        candidate_id="seed",
        scaffold_name="memgpt_source",
        passrate=0.0,
        average_score=0.0,
        token_consuming=10,
        avg_token_consuming=5,
        avg_prompt_tokens=4,
        avg_completion_tokens=1,
        count=1,
        config={"top_k": 8, "extra": {"source_family": "memgpt"}},
        result_path=str(result_path),
    )

    def fake_run_initial_frontier(*args, **kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "candidate": candidate.to_dict(),
                    "tasks": [
                        {
                            "task_id": "hard-case",
                            "question": "question",
                            "gold_answer": "gold",
                            "prediction": "pred",
                            "score": 0.0,
                            "passed": False,
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "retrieved": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"candidates": [candidate.to_dict()]}

    monkeypatch.setattr(optimizer, "_load_examples", lambda: [object()])
    monkeypatch.setattr(optimizer_module, "run_initial_frontier", fake_run_initial_frontier)

    optimizer.run()

    low = json.loads((tmp_path / "trace_slices" / "low" / "seed.json").read_text())
    assert low["case_limit"] == 10
    assert low["cases"][0]["task_id"] == "hard-case"


def test_candidate_code_policy_rejects_runtime_trace_and_scorer_access(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "bad_candidate.py").write_text(
        "\n".join(
            [
                "from memomemo.metrics import score_prediction",
                "",
                "def leak():",
                "    return open('runs/r/candidate_results/iter001.json').read()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))

    violations = optimizer._candidate_code_policy_violations(
        {
            "module": "bad_candidate",
            "class": "BadCandidate",
            "candidate_root": str(generated),
        }
    )

    markers = {item["marker"] for item in violations}
    assert "candidate_results" in markers
    assert "score_prediction" in markers


def test_candidate_code_policy_rejects_raw_locomo_access_in_source_snapshot(tmp_path):
    project_source = tmp_path / "snap" / "candidate" / "project_source"
    scaffold_dir = project_source / "src" / "memomemo" / "scaffolds"
    scaffold_dir.mkdir(parents=True)
    (scaffold_dir / "memgpt_scaffold.py").write_text(
        "from pathlib import Path\nPath('data/locomo/locomo10.json').read_text()\n",
        encoding="utf-8",
    )
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))

    violations = optimizer._candidate_code_policy_violations(
        {
            "scaffold_name": "memgpt_source",
            "extra": {"source_project_path": str(project_source)},
        }
    )

    assert any(item["marker"] == "data/locomo" for item in violations)


def test_source_candidate_policy_allows_preexisting_scorer_import(tmp_path):
    candidate_source = tmp_path / "snap" / "candidate" / "project_source"
    original_source = tmp_path / "snap" / "candidate" / "original_project_source"
    for root in (candidate_source, original_source):
        scaffold_dir = root / "src" / "memomemo" / "scaffolds"
        scaffold_dir.mkdir(parents=True)
        (scaffold_dir / "base.py").write_text(
            "from memomemo.metrics import retrieval_oracle_prediction\n",
            encoding="utf-8",
        )
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))

    violations = optimizer._candidate_code_policy_violations(
        {
            "scaffold_name": "memgpt_source",
            "extra": {"source_project_path": str(candidate_source)},
        }
    )

    assert violations == []


def test_source_candidate_policy_rejects_new_scorer_import(tmp_path):
    candidate_source = tmp_path / "snap" / "candidate" / "project_source"
    original_source = tmp_path / "snap" / "candidate" / "original_project_source"
    for root in (candidate_source, original_source):
        scaffold_dir = root / "src" / "memomemo" / "scaffolds"
        scaffold_dir.mkdir(parents=True)
        (scaffold_dir / "base.py").write_text(
            "class BaseMemoryScaffold:\n    pass\n",
            encoding="utf-8",
        )
    (candidate_source / "src" / "memomemo" / "scaffolds" / "memgpt_scaffold.py").write_text(
        "from memomemo.metrics import retrieval_oracle_prediction\n",
        encoding="utf-8",
    )
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))

    violations = optimizer._candidate_code_policy_violations(
        {
            "scaffold_name": "memgpt_source",
            "extra": {"source_project_path": str(candidate_source)},
        }
    )

    assert any(item["marker"] == "memomemo.metrics" for item in violations)


def test_single_top_k_uses_first_value_from_list():
    assert _single_top_k([4, 8]) == (4, True)
    assert _single_top_k([8]) == (8, False)
    assert _single_top_k(12) == (12, False)


def test_optimizer_copies_declared_locomo_source_scope(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    dest = tmp_path / "context"

    optimizer._copy_project_source_context(dest)
    optimizer._copy_upstream_source_context("memgpt", dest)

    manifest = json.loads((dest / "project_source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"] == LOCOMO_WORKSPACE_SPEC.benchmark
    assert sorted(manifest["source_files"]) == sorted(LOCOMO_WORKSPACE_SPEC.source_files)
    assert (dest / "project_source" / "src" / "memomemo" / "dynamic.py").exists()
    assert (
        dest / "project_source" / "src" / "memomemo" / "__init__.py"
    ).read_text(encoding="utf-8") == MINIMAL_BENCHMARK_PACKAGE_INIT
    assert not (dest / "project_source" / "src" / "memomemo" / "scaffolds" / "mem0_scaffold.py").exists()
    assert (dest / "project_source" / "src" / "memomemo" / "scaffolds" / "memgpt_scaffold.py").exists()
    assert not (dest / "project_source" / "src" / "memomemo" / "optimizer.py").exists()
    assert not (dest / "project_source" / "src" / "memomemo" / "baseline.py").exists()
    assert not (dest / "project_source" / "src" / "memomemo" / "tau_banking.py").exists()
    assert not (dest / "project_source" / "src" / "memomemo" / "tau_agents").exists()
    assert not (dest / "project_source" / "src" / "memomemo" / "text_classification.py").exists()
    assert not (dest / "upstream_source" / "mem0").exists()
    assert (dest / "upstream_source" / "MemGPT" / "letta" / "schemas" / "memory.py").exists()
    assert not (dest / "upstream_source" / "MemoryBank-SiliconFriend").exists()


def test_source_snapshot_uses_single_candidate_dir(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    call_dir = tmp_path / "call"
    call_dir.mkdir()

    snapshot_root = optimizer._build_source_snapshot_workspace(
        iteration=3,
        source_family="memgpt",
        call_dir=call_dir,
    )

    manifest = json.loads((snapshot_root / "manifest.json").read_text(encoding="utf-8"))
    assert (snapshot_root / "candidate" / "SNAPSHOT.md").exists()
    assert not (snapshot_root / "candidate_a").exists()
    assert not (snapshot_root / "candidate_b").exists()
    assert manifest["candidate_dir"] == str(snapshot_root / "candidate")
    assert manifest["benchmark"] == LOCOMO_WORKSPACE_SPEC.benchmark
    assert manifest["primary_source_file"] == LOCOMO_WORKSPACE_SPEC.primary_source_file
    assert sorted(manifest["project_source_files"]) == sorted(LOCOMO_WORKSPACE_SPEC.source_files)
    assert "trace_scope" not in manifest
    assert "optimization_cell" not in manifest
    assert "cost_level" not in manifest
    assert "slots" not in manifest


def _make_curaii_candidate(
    iteration: int,
    passrate: float,
    average_score: float = 0.0,
    token_consuming: int = 1,
) -> CandidateResult:
    return CandidateResult(
        candidate_id=f"iter{iteration:03d}_x",
        scaffold_name="memgpt_source",
        passrate=passrate,
        average_score=average_score,
        token_consuming=token_consuming,
        avg_token_consuming=float(token_consuming),
        avg_prompt_tokens=0.0,
        avg_completion_tokens=0.0,
        count=1,
        config={"extra": {"source_family": "memgpt"}},
        result_path="",
    )


def _seed_curaii_iter_dirs(optimizer, iterations: list[int]) -> None:
    for iteration in iterations:
        proj = (
            optimizer.run_dir
            / "proposer_calls"
            / f"iter_{iteration:03d}"
            / "source_snapshot"
            / "candidate"
            / "project_source"
        )
        proj.mkdir(parents=True, exist_ok=True)


def _seed_candidate(passrate: float, *, scaffold: str = "memgpt_source_top12") -> CandidateResult:
    """Produce a seed candidate (no `iterNNN_` prefix) used as baseline."""

    return CandidateResult(
        candidate_id=scaffold,
        scaffold_name="memgpt_source",
        passrate=passrate,
        average_score=passrate,
        token_consuming=1,
        avg_token_consuming=1.0,
        avg_prompt_tokens=0.0,
        avg_completion_tokens=0.0,
        count=1,
        config={"extra": {"source_family": "memgpt"}},
        result_path="",
    )


def test_seed_passrate_picks_max_among_seed_candidates(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    candidates = [
        _seed_candidate(0.16, scaffold="memgpt_source_top12"),
        _seed_candidate(0.20, scaffold="bm25_source_top8"),
        _make_curaii_candidate(1, 0.55),  # has iter prefix → not a seed
        _make_curaii_candidate(2, 0.30),
    ]
    assert optimizer._seed_passrate(candidates) == pytest.approx(0.20)


def test_seed_passrate_returns_zero_when_no_seeds(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    assert optimizer._seed_passrate([]) == 0.0
    # All candidates have `iter` prefix → no seeds.
    assert optimizer._seed_passrate(
        [_make_curaii_candidate(1, 0.5)]
    ) == 0.0


def test_curaii_low_budget_selects_best1_when_above_baseline(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    _seed_curaii_iter_dirs(optimizer, [1, 2, 3])
    candidates = [
        _make_curaii_candidate(1, 0.50),
        _make_curaii_candidate(2, 0.45),
        _make_curaii_candidate(3, 0.40),
    ]
    base_iter, refs = optimizer._curaii_select_for_budget(
        candidates,
        iteration=10,
        budget="low",
        baseline_passrate=0.30,
    )
    # best1 by passrate is iter_1; refs is exactly that single iter.
    assert base_iter == 1
    assert refs == (1,)


def test_curaii_low_budget_falls_back_when_best_not_above_baseline(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    _seed_curaii_iter_dirs(optimizer, [1, 2])
    # All candidates tied at the baseline — strict > gate fails for both.
    candidates = [
        _make_curaii_candidate(1, 0.362),
        _make_curaii_candidate(2, 0.362),
    ]
    base_iter, refs = optimizer._curaii_select_for_budget(
        candidates,
        iteration=5,
        budget="low",
        baseline_passrate=0.362,
    )
    # Empty pool → (None, None) signals clean-baseline + default refs.
    assert base_iter is None
    assert refs is None


def test_curaii_medium_budget_picks_random_from_pool_and_includes_base_in_refs(
    tmp_path, monkeypatch
):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    _seed_curaii_iter_dirs(optimizer, [1, 2, 3, 4, 5])
    candidates = [
        _make_curaii_candidate(1, 0.55),
        _make_curaii_candidate(2, 0.50),
        _make_curaii_candidate(3, 0.45),
        _make_curaii_candidate(4, 0.40),  # > baseline 0.36 but outside top-3
        _make_curaii_candidate(5, 0.35),  # ≤ baseline → excluded
    ]
    # Force random.choice to pick the second-ranked (iter_2).
    monkeypatch.setattr(
        optimizer_module.random, "choice", lambda seq: seq[1]
    )
    base_iter, refs = optimizer._curaii_select_for_budget(
        candidates,
        iteration=10,
        budget="medium",
        baseline_passrate=0.36,
    )
    assert base_iter == 2
    # Refs = the (≤3) qualified pool sorted by iteration; iter_4 (rank 4) and
    # iter_5 (≤baseline) are excluded.
    assert refs == (1, 2, 3)


def test_curaii_high_budget_returns_none_refs_to_pass_through(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    _seed_curaii_iter_dirs(optimizer, [1, 2, 3])
    candidates = [
        _make_curaii_candidate(1, 0.55),
        _make_curaii_candidate(2, 0.50),
        _make_curaii_candidate(3, 0.45),
    ]
    monkeypatch.setattr(
        optimizer_module.random, "choice", lambda seq: seq[0]
    )
    base_iter, refs = optimizer._curaii_select_for_budget(
        candidates,
        iteration=10,
        budget="high",
        baseline_passrate=0.30,
    )
    assert base_iter == 1
    # high budget delegates ref selection to _reference_iterations_for_budget.
    assert refs is None


def test_curaii_baseline_gate_excludes_tied_and_worse(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    _seed_curaii_iter_dirs(optimizer, [1, 2, 3])
    # iter_1 ties baseline; iter_2 below baseline; only iter_3 strictly beats it.
    candidates = [
        _make_curaii_candidate(1, 0.40),
        _make_curaii_candidate(2, 0.30),
        _make_curaii_candidate(3, 0.50),
    ]
    base_iter, refs = optimizer._curaii_select_for_budget(
        candidates,
        iteration=10,
        budget="low",
        baseline_passrate=0.40,
    )
    assert base_iter == 3
    assert refs == (3,)


def test_curaii_skips_candidates_without_snapshot_on_disk(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    # Only iter_002 has its archived snapshot on disk; iter_001's bundle was
    # pruned even though its CandidateResult still ranks higher.
    _seed_curaii_iter_dirs(optimizer, [2])
    candidates = [
        _make_curaii_candidate(1, 0.60),
        _make_curaii_candidate(2, 0.50),
    ]
    base_iter, refs = optimizer._curaii_select_for_budget(
        candidates,
        iteration=5,
        budget="low",
        baseline_passrate=0.30,
    )
    assert base_iter == 2
    assert refs == (2,)


def test_curaii_handles_fewer_than_three_eligible(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    _seed_curaii_iter_dirs(optimizer, [1, 2])
    candidates = [
        _make_curaii_candidate(1, 0.50),
        _make_curaii_candidate(2, 0.45),
    ]
    monkeypatch.setattr(
        optimizer_module.random, "choice", lambda seq: seq[len(seq) - 1]
    )
    base_iter, refs = optimizer._curaii_select_for_budget(
        candidates,
        iteration=5,
        budget="medium",
        baseline_passrate=0.30,
    )
    assert base_iter in {1, 2}
    assert refs == (1, 2)


def test_reference_bundle_prunes_trace_slices_to_budget_access(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    src = tmp_path / "proposer_calls" / "iter_001"
    for level in ("low", "medium", "high"):
        trace_dir = src / "trace_slices" / level
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "candidate.json").write_text("{}", encoding="utf-8")

    low_dest = tmp_path / "low_bundle"
    optimizer._copy_iteration_bundle(src, low_dest, trace_scope="last1")
    assert (low_dest / "trace_slices" / "low" / "candidate.json").exists()
    assert not (low_dest / "trace_slices" / "medium").exists()
    assert not (low_dest / "trace_slices" / "high").exists()

    medium_dest = tmp_path / "medium_bundle"
    optimizer._copy_iteration_bundle(src, medium_dest, trace_scope="last3")
    assert not (medium_dest / "trace_slices" / "low").exists()
    assert (medium_dest / "trace_slices" / "medium" / "candidate.json").exists()
    assert not (medium_dest / "trace_slices" / "high").exists()

    high_dest = tmp_path / "high_bundle"
    optimizer._copy_iteration_bundle(src, high_dest, trace_scope="all")
    assert (high_dest / "trace_slices" / "low" / "candidate.json").exists()
    assert (high_dest / "trace_slices" / "medium" / "candidate.json").exists()
    assert (high_dest / "trace_slices" / "high" / "candidate.json").exists()


def test_append_summary_writes_only_global_summary(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    candidate = _candidate("memgpt_source_top8")

    optimizer._append_summary(iteration=0, candidate=candidate)

    summary = tmp_path / "evolution_summary.jsonl"
    assert summary.exists()
    assert "memgpt_source_top8" in summary.read_text(encoding="utf-8")
    assert not (tmp_path / "cells").exists()


def test_progressive_workspace_outputs_sync_and_normalize_candidate_paths(
    tmp_path, monkeypatch
):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    call_dir = tmp_path / "proposer_calls" / "iter_004"
    workspace = call_dir / "workspace"
    captured = {}

    monkeypatch.setattr(
        optimizer,
        "_build_progressive_workspace",
        lambda **kwargs: (workspace, (1,)),
    )

    def fake_run_code_agent_prompt(prompt, **kwargs):
        assert kwargs["cwd"] == workspace
        (workspace / "generated").mkdir(parents=True)
        (workspace / "generated" / "progressive_candidate.py").write_text(
            "class ProgressiveCandidate: pass\n",
            encoding="utf-8",
        )
        (workspace / "pending_eval.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "name": "progressive_candidate",
                            "module": "progressive_candidate",
                            "class": "ProgressiveCandidate",
                            "extra": {
                                "source_project_path": "source_snapshot/candidate/project_source",
                                "source_base_dir": "source_base_variants/progressive_candidate",
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            metrics={},
            usage=None,
            tool_access={
                "files_read": {},
                "files_written": {
                    "generated/progressive_candidate.py": {"writes": 1},
                    "pending_eval.json": {"writes": 1},
                },
            },
        )

    def fake_evaluate(iteration, proposed, examples):
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer._run_progressive_proposer_iteration(
        4,
        [_candidate("seed")],
        examples=[],
        budget="low",
        adaptive=True,
    )

    proposed = captured["proposed"]
    assert (tmp_path / "generated" / "progressive_candidate.py").exists()
    assert proposed[0]["candidate_root"] == str(tmp_path / "generated")
    assert "optimization_target" not in proposed[0]
    assert "bandit_arm" not in proposed[0]
    assert proposed[0]["extra"]["source_project_path"] == str(
        (call_dir / "source_snapshot" / "candidate" / "project_source").resolve()
    )
    assert proposed[0]["extra"]["source_base_dir"] == str(
        (workspace / "source_base_variants" / "progressive_candidate").resolve()
    )
    pending = json.loads((tmp_path / "pending_eval.json").read_text(encoding="utf-8"))
    assert pending["candidates"][0]["candidate_root"] == str(tmp_path / "generated")
    assert "bandit_arm" not in pending["candidates"][0]


def test_progressive_proposer_retries_invalid_pending_eval_json(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_sandbox="none",
        )
    )
    call_dir = tmp_path / "proposer_calls" / "iter_004"
    workspace = call_dir / "workspace"
    prompts = []
    captured = {}

    monkeypatch.setattr(
        optimizer,
        "_build_progressive_workspace",
        lambda **kwargs: (workspace, (1,)),
    )

    def fake_run_code_agent_prompt(prompt, **kwargs):
        prompts.append(prompt)
        assert kwargs["cwd"] == workspace
        workspace.mkdir(parents=True, exist_ok=True)
        if len(prompts) == 1:
            (workspace / "pending_eval.json").write_text(
                '{"candidates": [{"name": "bad\\candidate", "scaffold_name": "bm25"}]}',
                encoding="utf-8",
            )
        else:
            (workspace / "pending_eval.json").write_text(
                json.dumps({"candidates": [{"name": "fixed", "scaffold_name": "bm25"}]}),
                encoding="utf-8",
            )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            metrics={},
            usage=None,
            tool_access={
                "files_read": {},
                "files_written": {"pending_eval.json": {"writes": 1}},
            },
        )

    def fake_evaluate(iteration, proposed, examples):
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer._run_progressive_proposer_iteration(
        4,
        [_candidate("seed")],
        examples=[],
        budget="medium",
        adaptive=True,
    )

    assert len(prompts) == 2
    assert "invalid JSON" in prompts[1]
    assert captured["proposed"][0]["name"] == "fixed"
    summary = (tmp_path / "evolution_summary.jsonl").read_text(encoding="utf-8")
    assert "proposer_invalid_pending_retry" in summary
    assert "proposer_invalid_pending_rejected" not in summary


def test_progressive_docker_sandbox_maps_container_workspace_paths(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            selection_policy="progressive",
            proposer_sandbox="docker",
            proposer_docker_image="memo-proposer:test",
            proposer_docker_workspace="/workspace",
        )
    )
    call_dir = tmp_path / "proposer_calls" / "iter_004"
    workspace = call_dir / "workspace"
    captured = {}

    monkeypatch.setattr(
        optimizer,
        "_build_progressive_workspace",
        lambda **kwargs: (workspace, (1,)),
    )

    def fake_run_code_agent_prompt(prompt, **kwargs):
        captured["sandbox"] = kwargs["sandbox"]
        assert kwargs["cwd"] == workspace
        (workspace / "generated").mkdir(parents=True)
        (workspace / "source_snapshot" / "candidate" / "project_source" / "src").mkdir(
            parents=True
        )
        (workspace / "pending_eval.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "name": "progressive_source_candidate",
                            "scaffold_name": "memgpt_source",
                            "extra": {
                                "source_project_path": (
                                    "/workspace/source_snapshot/candidate/project_source"
                                ),
                                "source_base_dir": "/workspace/source_base_variants/build",
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            metrics={},
            usage=None,
            tool_access={
                "files_read": {"/workspace/summaries/best_candidates.json": {"reads": 1}},
                "files_written": {"/workspace/pending_eval.json": {"writes": 1}},
            },
        )

    def fake_evaluate(iteration, proposed, examples):
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer._run_progressive_proposer_iteration(
        4,
        [_candidate("seed")],
        examples=[],
        budget="low",
        adaptive=True,
    )

    assert captured["sandbox"].kind == "docker"
    assert captured["sandbox"].docker_image == "memo-proposer:test"
    assert captured["proposed"][0]["extra"]["source_project_path"] == str(
        (call_dir / "source_snapshot" / "candidate" / "project_source").resolve()
    )
    assert captured["proposed"][0]["extra"]["source_base_dir"] == str(
        (workspace / "source_base_variants" / "build").resolve()
    )
    summary = (tmp_path / "evolution_summary.jsonl").read_text(encoding="utf-8")
    assert "proposer_access_rejected" not in summary


def test_progressive_access_violation_retries_with_boundary_feedback(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_sandbox="none",
        )
    )
    workspace = tmp_path / "proposer_calls" / "iter_004" / "workspace"
    prompts = []
    build_calls = []
    captured = {}

    def fake_build_workspace(**kwargs):
        build_calls.append(kwargs)
        if workspace.exists():
            import shutil

            shutil.rmtree(workspace)
        (workspace / "generated").mkdir(parents=True)
        return workspace, (1,)

    monkeypatch.setattr(optimizer, "_build_progressive_workspace", fake_build_workspace)

    def fake_run_code_agent_prompt(prompt, **kwargs):
        prompts.append(prompt)
        assert kwargs["cwd"] == workspace
        if len(prompts) == 1:
            return SimpleNamespace(
                returncode=0,
                timed_out=False,
                stderr="",
                metrics={},
                usage=None,
                tool_access={"files_read": {str(tmp_path / "generated" / "leak.py"): {"reads": 1}}},
            )
        (workspace / "generated" / "progressive_retry_candidate.py").write_text(
            "class ProgressiveRetryCandidate: pass\n",
            encoding="utf-8",
        )
        (workspace / "pending_eval.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "name": "progressive_retry_candidate",
                            "module": "progressive_retry_candidate",
                            "class": "ProgressiveRetryCandidate",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            metrics={},
            usage=None,
            tool_access={
                "files_read": {"summaries/best_candidates.json": {"reads": 1}},
                "files_written": {
                    "generated/progressive_retry_candidate.py": {"writes": 1},
                    "pending_eval.json": {"writes": 1},
                },
            },
        )

    def fake_evaluate(iteration, proposed, examples):
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer._run_progressive_proposer_iteration(
        4,
        [_candidate("seed")],
        examples=[],
        budget="low",
        adaptive=True,
    )

    assert len(build_calls) == 2
    assert len(prompts) == 2
    assert "Retry Required: Filesystem Boundary Violation" in prompts[1]
    assert str(tmp_path / "generated" / "leak.py") in prompts[1]
    assert captured["proposed"][0]["name"] == "progressive_retry_candidate"
    summary = (tmp_path / "evolution_summary.jsonl").read_text(encoding="utf-8")
    assert "proposer_access_retry" in summary
    assert "proposer_access_rejected" not in summary


def test_progressive_docker_allows_container_internal_access(tmp_path):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            selection_policy="progressive",
            proposer_sandbox="docker",
            proposer_docker_image="memo-proposer:test",
        )
    )

    result = SimpleNamespace(
        tool_access={
            "files_read": {
                "/tmp/.claude/projects/-workspace/memory/MEMORY.md": {"reads": 1},
                "/etc/hosts": {"reads": 1},
            },
            "files_written": {
                "/tmp/.claude/projects/-workspace/memory/notes.md": {"writes": 1},
            },
        }
    )

    assert optimizer._proposer_access_violations(
        result,
        workspace_dir=tmp_path / "proposer_calls" / "iter_004" / "workspace",
    ) == []


def test_progressive_budget_schedule_transitions(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))

    assert [optimizer._progressive_budget_for_iteration(item) for item in range(1, 6)] == [
        "low",
        "low",
        "low",
        "low",
        "low",
    ]

    seed = _scored_candidate("seed", passrate=0.2)
    same = _scored_candidate("iter005_same_top8", passrate=0.2)
    optimizer._update_progressive_state(
        iteration=5,
        budget="low",
        previous_best_passrate=0.2,
        candidates=[seed, same],
        evaluated=[same],
    )
    assert optimizer._progressive_budget_for_iteration(6) == "medium"

    medium_same = _scored_candidate("iter006_same_top8", passrate=0.2)
    optimizer._update_progressive_state(
        iteration=6,
        budget="medium",
        previous_best_passrate=0.2,
        candidates=[seed, same, medium_same],
        evaluated=[medium_same],
    )
    assert optimizer._progressive_budget_for_iteration(7) == "high"

    improved = _scored_candidate("iter007_better_top8", passrate=0.3)
    optimizer._update_progressive_state(
        iteration=7,
        budget="high",
        previous_best_passrate=0.2,
        candidates=[seed, same, medium_same, improved],
        evaluated=[improved],
    )
    assert optimizer._progressive_budget_for_iteration(8) == "low"


def test_progressive_reference_selection_uses_best_iterations_only(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    for iteration in (1, 2, 3, 4):
        call_dir = tmp_path / "proposer_calls" / f"iter_{iteration:03d}"
        call_dir.mkdir(parents=True)
        (call_dir / "assignment.json").write_text("{}", encoding="utf-8")

    candidates = [
        _scored_candidate("iter001_mid_top8", passrate=0.3, average_score=0.4, tokens=200),
        _scored_candidate("iter002_best_top8", passrate=0.7, average_score=0.5, tokens=300),
        _scored_candidate("iter003_worst_top8", passrate=0.1, average_score=0.2, tokens=100),
        _scored_candidate("iter004_second_top8", passrate=0.6, average_score=0.6, tokens=150),
    ]

    assert optimizer._reference_iterations_for_budget(
        "low",
        iteration=5,
        candidates=candidates,
    ) == (2,)
    assert optimizer._reference_iterations_for_budget(
        "medium",
        iteration=5,
        candidates=candidates,
    ) == (2, 4, 1)
    assert optimizer._reference_iterations_for_budget(
        "high",
        iteration=5,
        candidates=candidates,
    ) == (1, 2, 3, 4)


def test_random_and_recent_reference_selection_use_three_raw_iterations(tmp_path):
    candidates = [
        _scored_candidate("iter001_a_top8", passrate=0.1),
        _scored_candidate("iter002_b_top8", passrate=0.2),
        _scored_candidate("iter003_c_top8", passrate=0.3),
        _scored_candidate("iter004_d_top8", passrate=0.4),
        _scored_candidate("iter005_e_top8", passrate=0.5),
    ]
    for iteration in (1, 2, 3, 4, 5):
        call_dir = tmp_path / "proposer_calls" / f"iter_{iteration:03d}"
        (call_dir / "eval").mkdir(parents=True)
        (call_dir / "eval" / "candidate_result.compact.json").write_text(
            "{}",
            encoding="utf-8",
        )

    recent = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            selection_policy="recent",
            proposer_docker_image="memo-proposer:test",
        )
    )
    assert recent._reference_iterations_for_budget(
        "medium",
        iteration=6,
        candidates=candidates,
    ) == (3, 4, 5)

    random_policy = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            selection_policy="random",
            proposer_docker_image="memo-proposer:test",
        )
    )
    refs = random_policy._reference_iterations_for_budget(
        "medium",
        iteration=6,
        candidates=candidates,
    )
    assert len(refs) == 3
    assert refs == random_policy._reference_iterations_for_budget(
        "medium",
        iteration=6,
        candidates=candidates,
    )
    assert set(refs).issubset({1, 2, 3, 4, 5})


def test_best_reference_selection_picks_top_three_by_passrate(tmp_path):
    candidates = [
        _scored_candidate("iter001_a_top8", passrate=0.10),
        _scored_candidate("iter002_b_top8", passrate=0.55),
        _scored_candidate("iter003_c_top8", passrate=0.20),
        _scored_candidate("iter004_d_top8", passrate=0.45),
        _scored_candidate("iter005_e_top8", passrate=0.30),
    ]
    for iteration in (1, 2, 3, 4, 5):
        call_dir = tmp_path / "proposer_calls" / f"iter_{iteration:03d}"
        (call_dir / "eval").mkdir(parents=True)
        (call_dir / "eval" / "candidate_result.compact.json").write_text(
            "{}",
            encoding="utf-8",
        )

    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            selection_policy="best",
            proposer_docker_image="memo-proposer:test",
        )
    )
    refs = optimizer._reference_iterations_for_budget(
        "medium",
        iteration=6,
        candidates=candidates,
    )
    # top-3 passrates are iter002 (0.55), iter004 (0.45), iter005 (0.30); sorted ascending
    assert refs == (2, 4, 5)


def test_best_candidates_uses_passrate_average_score_pareto_frontier(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    candidates = [
        _scored_candidate("iter001_low_both_top8", passrate=0.4, average_score=0.4),
        _scored_candidate("iter002_high_passrate_top8", passrate=0.7, average_score=0.5),
        _scored_candidate("iter003_high_score_top8", passrate=0.6, average_score=0.7),
        _scored_candidate("iter004_dominated_top8", passrate=0.6, average_score=0.5),
    ]

    optimizer._save_best_candidates(candidates)

    payload = json.loads(optimizer.frontier_path.read_text(encoding="utf-8"))
    assert [item["candidate_id"] for item in payload] == [
        "iter002_high_passrate_top8",
        "iter003_high_score_top8",
    ]


def test_bandit_quality_reward_uses_passrate_not_average_score(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    low_pass_high_score = _scored_candidate(
        "iter001_low_pass_high_score_top8",
        passrate=0.3,
        average_score=0.9,
    )
    high_pass_low_score = _scored_candidate(
        "iter002_high_pass_low_score_top8",
        passrate=0.6,
        average_score=0.1,
    )

    assert optimizer._best_quality_value([low_pass_high_score, high_pass_low_score]) == 0.6


def test_bandit_state_ignores_existing_quality_history(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    optimizer.bandit_state_path.write_text(
        json.dumps(
            {
                "total_iters": 2,
                "success_iters": 1,
                "global_reward_sum": 0.0,
                "quality_history": [0.9, 0.8],
                "passrate_history": [0.2, 0.25],
                "files": {},
            }
        ),
        encoding="utf-8",
    )
    call_dir = tmp_path / "proposer_calls" / "iter_003"
    (call_dir / "agent").mkdir(parents=True)
    (call_dir / "agent" / "tool_access.json").write_text(
        json.dumps({"files_read": {}}),
        encoding="utf-8",
    )
    (call_dir / "diff.patch").write_text("", encoding="utf-8")

    optimizer._update_bandit_state(
        iteration=3,
        previous_best_passrate=0.25,
        previous_best_quality=0.9,
        evaluated=[_scored_candidate("iter003_better_top8", passrate=0.3, average_score=0.1)],
        call_dir=call_dir,
    )

    state = json.loads(optimizer.bandit_state_path.read_text(encoding="utf-8"))
    assert "quality_history" not in state
    assert state["passrate_history"] == [0.2, 0.25, pytest.approx(0.3)]


def test_run_test_frontier_evaluates_all_quality_frontier_candidates(tmp_path, monkeypatch):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            test_frontier=True,
            test_limit=2,
        )
    )
    candidates = [
        _scored_candidate("iter001_low_both_top8", passrate=0.4, average_score=0.4),
        _scored_candidate("iter002_high_passrate_top8", passrate=0.7, average_score=0.5),
        _scored_candidate("iter003_high_score_top8", passrate=0.6, average_score=0.7),
        _scored_candidate("iter004_dominated_top8", passrate=0.6, average_score=0.5),
    ]
    loaded_specs = []

    def fake_load_candidate_scaffold(spec, *, project_root):
        loaded_specs.append(dict(spec))
        return SimpleNamespace(name=spec["name"])

    class FakeRunner:
        def __init__(self, examples, out_dir):
            self.examples = examples
            self.out_dir = out_dir

        def evaluate_scaffold(self, *, scaffold, scaffold_name, config, candidate_id):
            return CandidateResult(
                candidate_id=candidate_id,
                scaffold_name=scaffold_name,
                passrate=0.5,
                average_score=0.6,
                token_consuming=20,
                avg_token_consuming=10,
                avg_prompt_tokens=8,
                avg_completion_tokens=2,
                count=len(self.examples),
                config=config.to_dict(),
                result_path=str(self.out_dir / "candidate_results" / f"{candidate_id}.json"),
            )

    monkeypatch.setattr(
        optimizer,
        "_load_examples_for_split",
        lambda split, limit: [object(), object()],
    )
    monkeypatch.setattr(
        optimizer,
        "_make_evaluation_runner",
        lambda examples, *, out_dir=None: FakeRunner(examples, out_dir),
    )
    monkeypatch.setattr(optimizer_module, "load_candidate_scaffold", fake_load_candidate_scaffold)

    summary = optimizer._run_test_frontier(candidates)

    assert summary["split"] == "test"
    assert summary["limit"] == 2
    assert summary["train_frontier_count"] == 2
    assert summary["evaluated_count"] == 2
    assert summary["failed_count"] == 0
    assert [item["original_candidate_id"] for item in loaded_specs] == [
        "iter002_high_passrate_top8",
        "iter003_high_score_top8",
    ]
    assert [item["candidate_id"] for item in loaded_specs] == [
        "test_iter002_high_passrate_top8",
        "test_iter003_high_score_top8",
    ]
    assert (tmp_path / "test_frontier" / "test_results.json").exists()
    assert (tmp_path / "test_frontier" / "test_pareto_frontier.json").exists()


def test_progressive_workspace_copies_full_summaries_and_selected_raw_refs(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    optimizer.summary_path.write_text(
        "\n".join(
            json.dumps({"iteration": item, "candidate": {"candidate_id": f"iter{item:03d}_x"}})
            for item in range(1, 5)
        )
        + "\n",
        encoding="utf-8",
    )
    optimizer.frontier_path.write_text("[]\n", encoding="utf-8")
    optimizer.candidate_score_table_path.write_text("[]\n", encoding="utf-8")
    optimizer.retrieval_diagnostics_summary_path.write_text("[]\n", encoding="utf-8")
    optimizer.iteration_index_path.write_text("[]\n", encoding="utf-8")
    optimizer.diff_summary_path.write_text(
        json.dumps({"iteration": 4, "files_changed": ["x.py"]}) + "\n",
        encoding="utf-8",
    )
    for iteration in (1, 2):
        call_dir = tmp_path / "proposer_calls" / f"iter_{iteration:03d}"
        (call_dir / "eval").mkdir(parents=True)
        (call_dir / "eval" / "candidate_result.compact.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (call_dir / "trace_slices" / "low").mkdir(parents=True)
        (call_dir / "trace_slices" / "low" / "candidate.json").write_text(
            "{}",
            encoding="utf-8",
        )

    candidates = [
        _scored_candidate("iter001_best_top8", passrate=0.8),
        _scored_candidate("iter002_worst_top8", passrate=0.1),
    ]
    workspace, refs = optimizer._build_progressive_workspace(
        iteration=3,
        budget="low",
        existing_candidates=candidates,
        call_dir=tmp_path / "proposer_calls" / "iter_003",
    )

    assert refs == (1,)
    summary_lines = (workspace / "summaries" / "evolution_summary.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(summary_lines) == 4
    assert (workspace / "summaries" / "diff_summary.jsonl").exists()
    assert (
        workspace
        / "reference_iterations"
        / "iter_001"
        / "eval"
        / "candidate_result.compact.json"
    ).exists()
    assert (
        workspace
        / "source_snapshot"
        / "candidate"
        / "original_project_source"
        / "src"
        / "memomemo"
        / "dynamic.py"
    ).exists()
    assert (workspace / "workspace_manifest.json").exists()
    assert (workspace / "access_policy.json").exists()


def test_default_high_budget_workspace_copies_all_reference_iterations(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    for iteration in (1, 2, 3):
        call_dir = tmp_path / "proposer_calls" / f"iter_{iteration:03d}"
        (call_dir / "eval").mkdir(parents=True)
        (call_dir / "pending_eval.json").write_text(
            json.dumps({"candidates": [{"name": f"iter{iteration:03d}"}]}),
            encoding="utf-8",
        )
        (call_dir / "eval" / "candidate_result.compact.json").write_text(
            json.dumps({"candidate_id": f"iter{iteration:03d}_candidate_top8"}),
            encoding="utf-8",
        )
    candidates = [
        _scored_candidate("iter001_first_top8", passrate=0.2),
        _scored_candidate("iter002_second_top8", passrate=0.3),
        _scored_candidate("iter003_third_top8", passrate=0.4),
    ]

    workspace, refs = optimizer._build_progressive_workspace(
        iteration=4,
        budget="high",
        existing_candidates=candidates,
        call_dir=tmp_path / "proposer_calls" / "iter_004",
    )

    assert refs == (1, 2, 3)
    assignment = json.loads((workspace / "assignment.json").read_text(encoding="utf-8"))
    assert assignment["budget"] == "high"
    assert assignment["trace_scope"] == "all"
    for iteration in refs:
        assert (
            workspace
            / "reference_iterations"
            / f"iter_{iteration:03d}"
            / "eval"
            / "candidate_result.compact.json"
        ).exists()


def test_recent_workspace_copies_only_three_selected_reference_iterations(tmp_path):
    optimizer = LocomoOptimizer(
        LocomoOptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            selection_policy="recent",
            proposer_docker_image="memo-proposer:test",
        )
    )
    for iteration in (1, 2, 3, 4):
        call_dir = tmp_path / "proposer_calls" / f"iter_{iteration:03d}"
        (call_dir / "eval").mkdir(parents=True)
        (call_dir / "eval" / "candidate_result.compact.json").write_text(
            json.dumps({"candidate_id": f"iter{iteration:03d}_candidate_top8"}),
            encoding="utf-8",
        )
    candidates = [
        _scored_candidate("iter001_first_top8", passrate=0.4),
        _scored_candidate("iter002_second_top8", passrate=0.3),
        _scored_candidate("iter003_third_top8", passrate=0.2),
        _scored_candidate("iter004_fourth_top8", passrate=0.1),
    ]

    workspace, refs = optimizer._build_progressive_workspace(
        iteration=5,
        budget="medium",
        existing_candidates=candidates,
        call_dir=tmp_path / "proposer_calls" / "iter_005",
    )

    assert refs == (2, 3, 4)
    ref_dir = workspace / "reference_iterations"
    assert sorted(path.name for path in ref_dir.iterdir()) == [
        "iter_002",
        "iter_003",
        "iter_004",
    ]
    assert not (ref_dir / "iter_001").exists()


def test_bandit_first_iteration_bootstraps_workspace_and_access_advisory(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    policy = optimizer._bandit_policy_for_workspace(iteration=1, candidates=[])
    workspace, refs = optimizer._build_progressive_workspace(
        iteration=1,
        budget=policy["budget"],
        existing_candidates=[],
        call_dir=tmp_path / "proposer_calls" / "iter_001",
        reference_iterations_override=tuple(policy["reference_iterations"]),
        trace_scope_override=policy["trace_scope"],
        bandit_policy=policy,
    )

    assert policy["budget"] == "low"
    assert policy["trace_scope"] == "last1"
    assert refs == ()
    assignment = json.loads((workspace / "assignment.json").read_text(encoding="utf-8"))
    assert assignment["bandit_policy"]["hot_files"]
    assert "summaries/evolution_summary.jsonl" in policy["hot_files"]
    assert "summaries/best_candidates.json" in policy["hot_files"]
    assert "summaries/iteration_index.json" in policy["hot_files"]
    assert "pending_eval.json" in policy["hot_files"]
    access_policy = json.loads((workspace / "access_policy.json").read_text(encoding="utf-8"))
    assert access_policy["hot_paths"] == policy["hot_files"]
    assert "read_budget_lines_by_path" in access_policy


def test_bandit_state_updates_file_rewards_and_policy_scores(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    call_dir = tmp_path / "proposer_calls" / "iter_001"
    (call_dir / "agent").mkdir(parents=True)
    (call_dir / "agent" / "tool_access.json").write_text(
        json.dumps(
            {
                "files_read": {
                    "summaries/candidate_score_table.json": {"reads": 1, "lines": 40},
                    "source_snapshot/candidate/project_source/src/memomemo/scaffolds/memgpt_scaffold.py": {
                        "reads": 2,
                        "lines": 220,
                    },
                },
                "files_written": {
                    "source_snapshot/candidate/project_source/src/memomemo/scaffolds/memgpt_scaffold.py": {
                        "writes": 1
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (call_dir / "diff.patch").write_text(
        "diff --git a/src/memomemo/scaffolds/memgpt_scaffold.py "
        "b/src/memomemo/scaffolds/memgpt_scaffold.py\n",
        encoding="utf-8",
    )

    evaluated = [_scored_candidate("iter001_better_top8", passrate=0.35)]
    optimizer._update_bandit_state(
        iteration=1,
        previous_best_passrate=0.2,
        evaluated=evaluated,
        call_dir=call_dir,
    )

    state = json.loads(optimizer.bandit_state_path.read_text(encoding="utf-8"))
    source = state["files"][
        "source_snapshot/candidate/project_source/src/memomemo/scaffolds/memgpt_scaffold.py"
    ]
    assert state["total_iters"] == 1
    assert state["success_iters"] == 1
    assert state["passrate_history"] == [pytest.approx(0.35)]
    assert source["read_iters"] == 1
    assert source["success_iters"] == 1
    # Pre-window iterations use scaled improvement (raw delta * 10, clipped to
    # ±2.0) so a 0.35 passrate vs 0.2 prior yields reward = 1.5.
    assert source["reward_sum"] == pytest.approx(1.5)
    assert source["read_calls"] == 2
    assert source["read_lines"] == 220
    assert source["write_iters"] == 1
    # memgpt_scaffold.py is a required (always-include) file, so it is no
    # longer ranked against discretionary reads — its policy_score is None.
    assert source["policy_score"] is None


def test_bandit_failed_proposer_records_read_cost_without_success(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    call_dir = tmp_path / "proposer_calls" / "iter_001"
    (call_dir / "agent").mkdir(parents=True)
    (call_dir / "agent" / "tool_access.json").write_text(
        json.dumps({"files_read": {"summaries/evolution_summary.jsonl": {"reads": 1, "lines": 900}}}),
        encoding="utf-8",
    )

    optimizer._update_bandit_state(
        iteration=1,
        previous_best_passrate=0.2,
        evaluated=[],
        call_dir=call_dir,
    )

    state = json.loads(optimizer.bandit_state_path.read_text(encoding="utf-8"))
    row = state["files"]["summaries/evolution_summary.jsonl"]
    assert state["success_iters"] == 0
    assert row["read_iters"] == 1
    assert row["success_iters"] == 0
    assert row["reward_sum"] < 0


def test_bandit_iters_from_policy_paths_extracts_reference_iterations(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    paths = [
        "source_snapshot/candidate/project_source/src/memomemo/model.py",
        "reference_iterations/iter_017/pending_eval.json",
        "reference_iterations/iter_005/eval/retrieval_diagnostics.json",
        "reference_iterations/iter_017/diff.patch",   # duplicate iter
        "summaries/candidate_score_table.json",
    ]
    assert optimizer._iters_from_policy_paths(paths) == [17, 5]


def test_bandit_policy_high_budget_after_long_stagnation(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    optimizer.bandit_state_path.write_text(
        json.dumps({"total_iters": 12, "success_iters": 4, "files": {"x": {"policy_score": 0.1}}}),
        encoding="utf-8",
    )
    candidates = [_scored_candidate("iter001_better_top8", passrate=0.5)]
    candidates += [
        _scored_candidate(f"iter{n:03d}_step_top8", passrate=0.3) for n in range(2, 13)
    ]
    for n in range(1, 13):
        (tmp_path / "proposer_calls" / f"iter_{n:03d}").mkdir(parents=True, exist_ok=True)

    policy = optimizer._bandit_policy_for_workspace(iteration=13, candidates=candidates)

    assert policy["budget"] == "high"
    assert policy["trace_scope"] == "all"
    assert sorted(policy["reference_iterations"]) == list(range(1, 13))
    assert policy["cold_files"] == []


def test_bandit_policy_hot_paths_drive_reference_selection(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    optimizer.bandit_state_path.write_text(
        json.dumps(
            {
                "total_iters": 5,
                "success_iters": 2,
                "files": {
                    "reference_iterations/iter_004/pending_eval.json": {"policy_score": 0.3},
                    "reference_iterations/iter_002/diff.patch": {"policy_score": 0.2},
                    "reference_iterations/iter_001/pending_eval.json": {"policy_score": 0.1},
                    "source_snapshot/candidate/project_source/src/memomemo/model.py": {
                        "policy_score": 0.05
                    },
                },
                "last_policy": {
                    "hot_files": [
                        "reference_iterations/iter_004/pending_eval.json",
                        "reference_iterations/iter_002/diff.patch",
                    ],
                    "warm_files": [
                        "reference_iterations/iter_001/pending_eval.json",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    candidates = [
        _scored_candidate("iter001_a_top8", passrate=0.4),
        _scored_candidate("iter002_b_top8", passrate=0.45),
        _scored_candidate("iter003_c_top8", passrate=0.3),
        _scored_candidate("iter004_d_top8", passrate=0.5),
        _scored_candidate("iter005_e_top8", passrate=0.5),
    ]
    for n in range(1, 6):
        (tmp_path / "proposer_calls" / f"iter_{n:03d}").mkdir(parents=True, exist_ok=True)

    policy = optimizer._bandit_policy_for_workspace(iteration=6, candidates=candidates)

    refs = list(policy["reference_iterations"])
    # hot iters come first (4, 2), warm next (1), then fallback fills up to cap
    assert refs[0] == 4
    assert refs[1] == 2
    # cap = 5 for medium, 2 for low; either way iter_004 and iter_002 must be ahead of fallback-only iters
    assert len(refs) <= 5


def test_bandit_v4_reference_iterations_low_and_medium(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    state = {
        "files": {
            "reference_iterations/iter_004/pending_eval.json": {"policy_score": 0.3},
            "reference_iterations/iter_002/diff.patch": {"policy_score": 0.2},
        },
        "last_policy": {
            "hot_files": ["reference_iterations/iter_004/pending_eval.json"],
            "warm_files": ["reference_iterations/iter_002/diff.patch"],
        },
    }
    candidates = [
        _scored_candidate("iter001_a_top8", passrate=0.4),
        _scored_candidate("iter002_b_top8", passrate=0.30),
        _scored_candidate("iter003_c_top8", passrate=0.45),
        _scored_candidate("iter004_d_top8", passrate=0.55),  # best
        _scored_candidate("iter005_e_top8", passrate=0.50),
    ]
    for n in range(1, 6):
        (tmp_path / "proposer_calls" / f"iter_{n:03d}").mkdir(parents=True, exist_ok=True)

    # low: hot_iters [4, 2] prepended to base [best1=4] -> dedup [4, 2], cap=3
    refs_low = optimizer._bandit_reference_iterations(
        iteration=6, candidates=candidates, state=state, budget="low",
    )
    assert refs_low == (4, 2)

    # medium: hot_iters [4, 2] + base [best3=4,5,3] -> dedup [4, 2, 5, 3], cap=5
    refs_med = optimizer._bandit_reference_iterations(
        iteration=6, candidates=candidates, state=state, budget="medium",
    )
    assert refs_med == (4, 2, 5, 3)

    # high: full available history regardless of hot_iters
    refs_high = optimizer._bandit_reference_iterations(
        iteration=6, candidates=candidates, state=state, budget="high",
    )
    assert refs_high == (1, 2, 3, 4, 5)


def test_bandit_state_normalizes_diff_host_paths(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    call_dir = tmp_path / "proposer_calls" / "iter_001"
    (call_dir / "agent").mkdir(parents=True)
    (call_dir / "agent" / "tool_access.json").write_text(
        json.dumps(
            {
                "files_read": {
                    "/workspace/source_snapshot/candidate/project_source/src/memomemo/scaffolds/memgpt_scaffold.py": {
                        "reads": 1,
                        "lines": 220,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (call_dir / "diff.patch").write_text(
        "diff --git a/runs/r/proposer_calls/iter_001/source_snapshot/candidate/original_project_source/src/memomemo/scaffolds/memgpt_scaffold.py "
        "b/runs/r/proposer_calls/iter_001/source_snapshot/candidate/project_source/src/memomemo/scaffolds/memgpt_scaffold.py\n",
        encoding="utf-8",
    )

    optimizer._update_bandit_state(
        iteration=1,
        previous_best_passrate=0.2,
        evaluated=[_scored_candidate("iter001_better_top8", passrate=0.35)],
        call_dir=call_dir,
    )

    state = json.loads(optimizer.bandit_state_path.read_text(encoding="utf-8"))
    keys = list(state["files"].keys())
    assert all(not key.startswith("runs/") for key in keys), keys
    target_key = "source_snapshot/candidate/project_source/src/memomemo/scaffolds/memgpt_scaffold.py"
    assert target_key in state["files"], keys
    row = state["files"][target_key]
    assert row["changed_iters"] == 1
    assert row["read_iters"] == 1


def test_bandit_state_ignores_non_path_shell_arguments(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    call_dir = tmp_path / "proposer_calls" / "iter_001"
    (call_dir / "agent").mkdir(parents=True)
    (call_dir / "agent" / "tool_access.json").write_text(
        json.dumps(
            {
                "files_read": {
                    "summaries/candidate_score_table.json": {"reads": 1, "lines": 20},
                    "src/memomemo/scaffolds/memgpt_scaffold.py": {
                        "reads": 1,
                        "lines": 220,
                    },
                    "memomemo/scaffolds/memgpt_scaffold.py": {
                        "reads": 1,
                        "lines": 220,
                    },
                    "/tmp/.claude/projects/-workspace/memory/MEMORY.md": {
                        "reads": 1,
                        "lines": 80,
                    },
                    "s/$/\\n/": {"reads": 1, "lines": 1},
                    'failed_tasks[] | [.question, .prediction] | @tsv': {
                        "reads": 1,
                        "lines": 800,
                    },
                    'select(.iteration>=20) | [.iteration, .candidate.passrate] | @tsv': {
                        "reads": 1,
                        "lines": 800,
                    },
                },
                "files_written": {
                    "source_snapshot/candidate/project_source/src/memomemo/scaffolds/memgpt_scaffold.py": {
                        "writes": 1
                    },
                    ".foo | @json": {"writes": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    (call_dir / "diff.patch").write_text("", encoding="utf-8")

    optimizer._update_bandit_state(
        iteration=1,
        previous_best_passrate=0.2,
        evaluated=[_scored_candidate("iter001_better_top8", passrate=0.35)],
        call_dir=call_dir,
    )

    state = json.loads(optimizer.bandit_state_path.read_text(encoding="utf-8"))
    assert "summaries/candidate_score_table.json" in state["files"]
    assert (
        "source_snapshot/candidate/project_source/src/memomemo/scaffolds/memgpt_scaffold.py"
        in state["files"]
    )
    assert not any("failed_tasks[]" in path for path in state["files"])
    assert not any("select(.iteration" in path for path in state["files"])
    assert not any(path.startswith("src/memomemo/") for path in state["files"])
    assert not any(path.startswith("memomemo/") for path in state["files"])
    assert not any(".claude/projects" in path for path in state["files"])
    assert "s/$/\\n/" not in state["files"]
    assert ".foo | @json" not in state["files"]


def test_optimizer_records_and_aggregates_proposer_metrics(tmp_path):
    optimizer = LocomoOptimizer(LocomoOptimizerConfig(run_id="r", out_dir=tmp_path))
    result = SimpleNamespace(
        returncode=0,
        timed_out=False,
        usage={"total_cost_usd": 0.25},
        metrics={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 15,
            "total_reported_tokens": 150,
            "estimated_cost_usd": 0.25,
            "duration_s": 3.5,
            "tool_calls": 4,
            "tool_counts": {"Read": 2, "Write": 1, "Bash": 1},
            "read_file_calls": 2,
            "unique_files_read": 1,
            "read_lines": 42,
            "write_file_calls": 1,
            "written_lines": 7,
        },
        tool_access={
            "files_read": {"src/memomemo/optimizer.py": {"reads": 2, "lines": 42}},
            "files_written": {"runs/r/generated/candidate.py": {"writes": 1, "lines_written": 7}},
            "grep_requests": [],
            "tool_counts": {"Read": 2, "Write": 1, "Bash": 1},
        },
    )

    optimizer._append_proposer_result_event(
        iteration=1,
        result=result,
        selection_policy="progressive",
        extra={"budget": "low"},
    )

    row = json.loads(optimizer.summary_path.read_text(encoding="utf-8").strip())
    aggregate = optimizer._aggregate_proposer_metrics()

    assert row["event"] == "proposer_result"
    assert row["files_read"] == {"src/memomemo/optimizer.py": {"reads": 2, "lines": 42}}
    assert aggregate["calls"] == 1
    assert aggregate["estimated_cost_usd"] == 0.25
    assert aggregate["input_tokens"] == 100
    assert aggregate["tool_counts"] == {"Bash": 1, "Read": 2, "Write": 1}
    assert aggregate["read_file_calls"] == 2
    assert aggregate["unique_files_read"] == 1
    assert aggregate["written_lines"] == 7


def _candidate(candidate_id: str) -> CandidateResult:
    return CandidateResult(
        candidate_id=candidate_id,
        scaffold_name=candidate_id,
        passrate=0.1,
        average_score=0.2,
        token_consuming=100,
        avg_token_consuming=10,
        avg_prompt_tokens=8,
        avg_completion_tokens=2,
        count=10,
        config={"extra": {"source_family": "memgpt"}},
        result_path=f"{candidate_id}.json",
    )


def _scored_candidate(
    candidate_id: str,
    *,
    passrate: float,
    average_score: float = 0.2,
    tokens: int = 100,
) -> CandidateResult:
    return CandidateResult(
        candidate_id=candidate_id,
        scaffold_name="memgpt_source",
        passrate=passrate,
        average_score=average_score,
        token_consuming=tokens,
        avg_token_consuming=float(tokens),
        avg_prompt_tokens=8,
        avg_completion_tokens=2,
        count=10,
        config={"extra": {"source_family": "memgpt"}},
        result_path=f"{candidate_id}.json",
    )
