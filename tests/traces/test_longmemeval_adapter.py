"""Adapter unit tests — case dict → Trace, no I/O."""

from __future__ import annotations

from memomemo.traces import get_adapter
from memomemo.traces.adapters.longmemeval import LongMemEvalAdapter


def _sample_task() -> dict:
    return {
        "task_id": "LME::s::abc",
        "question": "How much cashback?",
        "gold_answer": "$0.75",
        "prediction": "unknown",
        "score": 0.0,
        "passed": False,
        "prompt_tokens": 1669,
        "completion_tokens": 24,
        "retrieved": [
            {
                "text": "session 1 about saveMart",
                "score": 0.82,
                "source": "archival",
                "metadata": {"memory_tier": "core"},
            },
            {
                "text": "session 2 unrelated",
                "score": 0.78,
                "source": "archival",
                "metadata": {},
            },
        ],
    }


def test_longmemeval_adapter_registered_under_correct_name():
    adapter = get_adapter("longmemeval")
    assert adapter.name == "longmemeval"


def test_build_trace_basic_shape():
    adapter = LongMemEvalAdapter()
    trace = adapter.build_trace(
        iteration=7,
        candidate_id="cand_xyz",
        task=_sample_task(),
    )

    assert trace.trace_id == "iter007_cand_xyz_LME::s::abc"
    assert trace.iteration == 7
    assert trace.candidate_id == "cand_xyz"
    assert trace.task_id == "LME::s::abc"
    assert trace.benchmark == "longmemeval"

    assert trace.summary["question"] == "How much cashback?"
    assert trace.summary["gold"] == "$0.75"
    assert trace.summary["passed"] is False
    assert trace.summary["score"] == 0.0
    assert trace.summary["prompt_tokens"] == 1669

    # diff is filled by the indexer in M2; in M1 it is None.
    assert trace.diff is None


def test_build_trace_spans_retrieval_and_generation():
    adapter = LongMemEvalAdapter()
    trace = adapter.build_trace(
        iteration=1,
        candidate_id="c1",
        task=_sample_task(),
    )

    assert [span.kind for span in trace.spans] == ["retrieval", "generation"]
    retrieval, generation = trace.spans

    docs = retrieval.output["documents"]
    assert len(docs) == 2
    assert [doc["rank"] for doc in docs] == [1, 2]
    assert docs[0]["score"] == 0.82
    assert docs[0]["content"] == "session 1 about saveMart"
    assert docs[0]["metadata"] == {"memory_tier": "core"}

    assert retrieval.output["total_returned"] == 2
    assert retrieval.input == {"query": "How much cashback?"}

    assert generation.output["content"] == "unknown"
    assert generation.metadata["completion_tokens"] == 24


def test_build_trace_handles_missing_retrieval():
    adapter = LongMemEvalAdapter()
    task = _sample_task()
    task["retrieved"] = None
    trace = adapter.build_trace(iteration=2, candidate_id="c1", task=task)
    retrieval = trace.spans[0]
    assert retrieval.output["documents"] == []
    assert retrieval.output["total_returned"] == 0


def test_build_trace_skips_non_dict_hits():
    adapter = LongMemEvalAdapter()
    task = _sample_task()
    task["retrieved"] = [
        {"text": "kept", "score": 1.0, "source": "x", "metadata": {}},
        "garbage_string",
        None,
        {"text": "kept2", "score": 0.5, "source": "y", "metadata": {}},
    ]
    trace = adapter.build_trace(iteration=2, candidate_id="c1", task=task)
    docs = trace.spans[0].output["documents"]
    assert len(docs) == 2
    # Rank is contiguous over kept hits only.
    assert [doc["rank"] for doc in docs] == [1, 2]
