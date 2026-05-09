"""schema.py — Trace ↔ JSONL roundtrip."""

from __future__ import annotations

import json

from memomemo.traces import (
    Span,
    Trace,
    read_jsonl,
    trace_from_dict,
    trace_to_dict,
    write_jsonl,
)


def _trace(idx: int) -> Trace:
    return Trace(
        trace_id=f"iter001_c1_task_{idx}",
        iteration=1,
        candidate_id="c1",
        task_id=f"task_{idx}",
        benchmark="longmemeval",
        summary={"score": float(idx) / 10, "passed": idx % 2 == 0},
        diff=None,
        spans=[
            Span(
                id="s1",
                kind="retrieval",
                input={"query": f"q{idx}"},
                output={"documents": [{"rank": 1, "score": 0.5, "content": "x"}]},
            ),
            Span(id="s2", kind="generation", output={"content": f"a{idx}"}),
        ],
    )


def test_trace_dict_roundtrip_preserves_fields():
    original = _trace(3)
    payload = trace_to_dict(original)
    restored = trace_from_dict(payload)
    assert restored.trace_id == original.trace_id
    assert restored.summary == original.summary
    assert len(restored.spans) == 2
    assert restored.spans[0].kind == "retrieval"
    assert restored.spans[0].output["documents"][0]["score"] == 0.5


def test_jsonl_roundtrip(tmp_path):
    traces = [_trace(i) for i in range(5)]
    path = tmp_path / "spans" / "iter_001" / "c1.jsonl"
    write_jsonl(path, traces)

    assert path.exists()
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 5
    # Every line is a valid JSON document.
    for line in raw_lines:
        parsed = json.loads(line)
        assert "trace_id" in parsed
        assert isinstance(parsed["spans"], list)

    restored = read_jsonl(path)
    assert len(restored) == 5
    assert [t.trace_id for t in restored] == [t.trace_id for t in traces]


def test_read_jsonl_missing_file_returns_empty(tmp_path):
    assert read_jsonl(tmp_path / "no_such.jsonl") == []
