"""Locomo adapter — verifies registration + benchmark label."""

from __future__ import annotations

from memomemo.traces import get_adapter, has_adapter
from memomemo.traces.adapters.locomo import LocomoAdapter


def _sample_task() -> dict:
    return {
        "task_id": "locomo::sess1::q3",
        "question": "Where did Jane go on Saturday?",
        "gold_answer": "the farmer's market",
        "prediction": "the farmer's market",
        "score": 1.0,
        "passed": True,
        "prompt_tokens": 1200,
        "completion_tokens": 18,
        "retrieved": [
            {
                "text": "Saturday morning farmer's market trip",
                "score": 0.91,
                "source": "archival",
                "metadata": {},
            }
        ],
    }


def test_locomo_adapter_registered():
    assert has_adapter("locomo")
    assert get_adapter("locomo").name == "locomo"


def test_locomo_adapter_builds_trace_with_correct_benchmark_label():
    adapter = LocomoAdapter()
    trace = adapter.build_trace(
        iteration=4,
        candidate_id="cand_x",
        task=_sample_task(),
    )
    assert trace.benchmark == "locomo"
    assert trace.task_id == "locomo::sess1::q3"
    assert trace.summary["passed"] is True
    assert trace.spans[0].kind == "retrieval"
    assert trace.spans[0].output["documents"][0]["score"] == 0.91


def test_locomo_optimizer_can_init_under_harness_backend(tmp_path):
    """Sanity: LocomoOptimizer with --trace-backend=harness no longer
    raises 'no adapter registered for locomo'."""

    from memomemo.locomo_optimizer import LocomoOptimizer, LocomoOptimizerConfig

    cfg = LocomoOptimizerConfig(
        run_id="r",
        out_dir=tmp_path,
        iterations=0,
        proposer_docker_image="test",
        trace_backend="harness",
    )
    optimizer = LocomoOptimizer(cfg)
    assert optimizer.trace_harness is not None
    assert optimizer.trace_harness.benchmark == "locomo"
