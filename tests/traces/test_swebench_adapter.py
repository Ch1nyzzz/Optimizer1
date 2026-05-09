"""SwebenchAdapter — registration and metadata mapping."""

from __future__ import annotations

from memomemo.traces import get_adapter, has_adapter
from memomemo.traces.adapters.swebench import SwebenchAdapter


def _sample_task(*, passed: bool = False) -> dict:
    return {
        "task_id": "astropy__astropy-12907",
        "question": "Long issue description here...",
        "gold_answer": "",
        "prediction": "",
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "retrieved": [],
        "metadata": {
            "benchmark": "swebench",
            "agent": "mini-swe-agent",
            "repo": "astropy/astropy",
            "base_commit": "abcdef1234567890abcdef",
            "patch_path": "/tmp/runs/.../patch.diff",
            "task_dir": "/tmp/runs/.../task_dir",
            "returncode": 0,
            "evaluator_returncode": 0 if passed else 1,
            "duration_s": 412.5,
        },
    }


def test_swebench_adapter_registered():
    assert has_adapter("swebench")
    assert get_adapter("swebench").name == "swebench"


def test_summary_maps_metadata_into_qa_compatible_slots():
    adapter = SwebenchAdapter()
    trace = adapter.build_trace(
        iteration=3,
        candidate_id="cand_x",
        task=_sample_task(passed=True),
    )
    assert trace.benchmark == "swebench"
    assert trace.task_id == "astropy__astropy-12907"

    summary = trace.summary
    # Question now reads as "<repo>@<short-commit>" — a renderable
    # one-liner the diagnostic markdown can show without further work.
    assert summary["question"] == "astropy/astropy@abcdef1234"
    assert summary["gold"] == "tests pass"
    assert summary["prediction"].startswith("patch: ")
    assert summary["passed"] is True
    assert summary["repo"] == "astropy/astropy"
    assert summary["duration_s"] == 412.5


def test_prediction_falls_back_when_no_patch():
    adapter = SwebenchAdapter()
    task = _sample_task(passed=False)
    task["metadata"]["patch_path"] = ""
    trace = adapter.build_trace(iteration=1, candidate_id="c", task=task)
    assert trace.summary["prediction"] == "<no patch produced>"


def test_agent_span_carries_pointer_to_task_dir():
    adapter = SwebenchAdapter()
    trace = adapter.build_trace(
        iteration=1,
        candidate_id="c",
        task=_sample_task(passed=False),
    )
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.kind == "agent"
    assert span.output["passed"] is False
    assert span.metadata["task_dir"].endswith("task_dir")
    assert span.metadata["repo"] == "astropy/astropy"


def test_swebench_optimizer_can_init_under_harness_backend(tmp_path):
    """Sanity: SwebenchOptimizer with --trace-backend=harness no longer
    raises 'no adapter registered for swebench'."""

    from memomemo.swebench_optimizer import (
        SwebenchOptimizer,
        SwebenchOptimizerConfig,
    )

    cfg = SwebenchOptimizerConfig(
        run_id="r",
        out_dir=tmp_path,
        iterations=0,
        proposer_docker_image="test",
        trace_backend="harness",
    )
    optimizer = SwebenchOptimizer(cfg)
    assert optimizer.trace_harness is not None
    assert optimizer.trace_harness.benchmark == "swebench"
