"""Tests for RationaleWriter and its TraceHarness integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optimizer1.schemas import CandidateResult
from optimizer1.traces.harness import TraceHarness
from optimizer1.traces.rationale import RationaleWriter, _parse_json_response


# ---- json parsing helper ------------------------------------------


def test_parse_json_response_strips_fences():
    raw = "```json\n{\"hypothesis\": \"x\"}\n```"
    parsed = _parse_json_response(raw)
    assert isinstance(parsed, dict)
    assert parsed["hypothesis"] == "x"


def test_parse_json_response_handles_prefix_text():
    raw = "Sure, here is the JSON:\n{\"hypothesis\": \"x\"}\n"
    parsed = _parse_json_response(raw)
    assert isinstance(parsed, dict) and parsed["hypothesis"] == "x"


def test_parse_json_response_returns_error_on_garbage():
    assert isinstance(_parse_json_response("not json at all"), str)


def test_parse_json_response_rejects_top_level_list():
    assert isinstance(_parse_json_response("[1, 2, 3]"), str)


# ---- RationaleWriter ----------------------------------------------


def _candidate(tmp_path: Path, *, candidate_id: str = "cand_x") -> CandidateResult:
    result_path = tmp_path / "candidate_results" / f"{candidate_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "candidate": {
                    "candidate_id": candidate_id,
                    "scaffold_name": "memgpt_source",
                },
                "tasks": [
                    {"task_id": "t1", "passed": True, "score": 1.0},
                    {"task_id": "t2", "passed": False, "score": 0.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return CandidateResult(
        candidate_id=candidate_id,
        scaffold_name="memgpt_source",
        passrate=0.5,
        average_score=0.5,
        token_consuming=10,
        avg_token_consuming=5,
        avg_prompt_tokens=4,
        avg_completion_tokens=1,
        count=2,
        config={"top_k": 8},
        result_path=str(result_path),
    )


def test_rationale_writer_produces_markdown_and_indexes_fields(tmp_path):
    captured: dict[str, list[dict[str, str]]] = {}

    def fake_caller(messages):
        captured["messages"] = messages
        return json.dumps(
            {
                "hypothesis": "raise top_k for long-context recall",
                "change": "memgpt_scaffold.py top_k 5 -> 10",
                "outcome": {
                    "passrate_delta": 0.05,
                    "regressed_tasks": ["t2"],
                    "breakthrough_tasks": [],
                },
                "diagnosis": "t2 failed; noise from extra retrievals overwhelmed signal",
                "next_hypothesis_signal": "make top_k question-type-dependent",
            }
        )

    writer = RationaleWriter(
        root=tmp_path / "traces",
        model="m",
        base_url="http://unused",
        llm_caller=fake_caller,
    )

    result = writer.write(
        iteration=3,
        candidate_id="cand_x",
        pending_eval={"intent": "raise top_k"},
        tasks=[
            {"task_id": "t1", "passed": True, "score": 1.0},
            {"task_id": "t2", "passed": False, "score": 0.0},
        ],
        diff_text="diff --git a/x b/x\n",
        status_counts={"regressed": 1, "stable_pass": 1},
        passrate=0.5,
        mean_score=0.5,
    )

    assert result.path.exists()
    text = result.path.read_text(encoding="utf-8")
    assert "raise top_k for long-context recall" in text
    assert "memgpt_scaffold" in text
    assert result.hypothesis == "raise top_k for long-context recall"
    assert result.diagnosis.startswith("t2 failed")
    assert result.parsed is not None


def test_rationale_writer_degrades_on_llm_error(tmp_path):
    def boom(messages):
        raise RuntimeError("upstream timeout")

    writer = RationaleWriter(
        root=tmp_path / "traces",
        model="m",
        base_url="http://unused",
        llm_caller=boom,
    )
    result = writer.write(
        iteration=1,
        candidate_id="cand_x",
        pending_eval=None,
        tasks=[],
        diff_text="",
        status_counts={},
        passrate=None,
        mean_score=None,
    )
    text = result.path.read_text(encoding="utf-8")
    assert "rationale unavailable" in text
    assert "upstream timeout" in text
    assert result.parsed is None


def test_rationale_writer_degrades_on_parse_error(tmp_path):
    def garbage(messages):
        return "I refuse to comply, here is some prose."

    writer = RationaleWriter(
        root=tmp_path / "traces",
        model="m",
        base_url="http://unused",
        llm_caller=garbage,
    )
    result = writer.write(
        iteration=1,
        candidate_id="cand_x",
        pending_eval=None,
        tasks=[],
        diff_text="",
        status_counts={},
        passrate=None,
        mean_score=None,
    )
    assert result.parsed is None
    assert "rationale unavailable" in result.path.read_text(encoding="utf-8")


# ---- harness integration ------------------------------------------


def test_harness_skips_rationale_for_baseline_iter_zero(tmp_path):
    """iter_0 of a self-baseline run should NOT produce rationales."""

    candidate = _candidate(tmp_path)
    seed_diff = tmp_path / "proposer_calls" / "iter_000" / "diff.patch"
    seed_diff.parent.mkdir(parents=True)
    seed_diff.write_text(
        "diff --git a/src/foo.py b/src/foo.py\nindex 0..1\n",
        encoding="utf-8",
    )
    calls: list[dict] = []

    def fake_caller(messages):
        calls.append({"messages": messages})
        return json.dumps({"hypothesis": "x", "change": "x"})

    writer = RationaleWriter(
        root=tmp_path / "traces",
        model="m",
        base_url="http://unused",
        llm_caller=fake_caller,
    )
    harness = TraceHarness(
        run_dir=tmp_path,
        benchmark="longmemeval",
        rationale_writer=writer,
    )
    harness.record_iteration(iteration=0, candidates=[candidate])

    assert not (tmp_path / "traces" / "rationale").exists()
    assert calls == []


def test_harness_writes_rationale_for_iter_one(tmp_path):
    """iter>=1 in a self-baseline run should trigger rationale."""

    seed = _candidate(tmp_path, candidate_id="seed")
    second = _candidate(tmp_path, candidate_id="cand_x")

    # iter 0 (baseline) and iter 1 each get a diff.patch.
    for it in (0, 1):
        d = tmp_path / "proposer_calls" / f"iter_{it:03d}" / "diff.patch"
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(
            f"diff --git a/src/foo_{it}.py b/src/foo_{it}.py\n",
            encoding="utf-8",
        )

    fake_payload = json.dumps(
        {
            "hypothesis": "h",
            "change": "c",
            "outcome": {
                "passrate_delta": 0.0,
                "regressed_tasks": [],
                "breakthrough_tasks": [],
            },
            "diagnosis": "d (mentions src/foo_1.py)",
            "next_hypothesis_signal": "n",
        }
    )

    def fake_caller(messages):
        return fake_payload

    writer = RationaleWriter(
        root=tmp_path / "traces",
        model="m",
        base_url="http://unused",
        llm_caller=fake_caller,
    )
    harness = TraceHarness(
        run_dir=tmp_path,
        benchmark="longmemeval",
        rationale_writer=writer,
    )
    harness.record_iteration(iteration=0, candidates=[seed])
    harness.record_iteration(iteration=1, candidates=[second])

    rationale_path = tmp_path / "traces" / "rationale" / "iter_001" / "cand_x.md"
    assert rationale_path.exists()
    assert "src/foo_1.py" in rationale_path.read_text(encoding="utf-8")

    # rationales table populated.
    import sqlite3

    with sqlite3.connect(tmp_path / "traces" / "index.db") as conn:
        rows = list(
            conn.execute(
                "SELECT iteration, candidate_id, hypothesis, diagnosis, next_signal "
                "FROM rationales"
            )
        )
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1] == "cand_x"
    assert rows[0][2] == "h"
    assert rows[0][4] == "n"
