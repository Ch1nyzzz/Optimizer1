"""TraceHarness end-to-end — fixture candidate JSON → traces/ tree."""

from __future__ import annotations

import json
from pathlib import Path

from optimizer1.schemas import CandidateResult
from optimizer1.traces import TraceHarness, read_jsonl


def _write_candidate_result(
    *,
    tmp_path: Path,
    candidate_id: str,
    tasks: list[dict],
) -> CandidateResult:
    result_path = tmp_path / "candidate_results" / f"{candidate_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = CandidateResult(
        candidate_id=candidate_id,
        scaffold_name="memgpt",
        passrate=sum(1 for t in tasks if t.get("passed")) / max(1, len(tasks)),
        average_score=sum(t.get("score", 0.0) for t in tasks) / max(1, len(tasks)),
        token_consuming=100,
        avg_token_consuming=10,
        avg_prompt_tokens=8,
        avg_completion_tokens=2,
        count=len(tasks),
        config={"top_k": 8},
        result_path=str(result_path),
    )
    result_path.write_text(
        json.dumps({"candidate": candidate.to_dict(), "tasks": tasks}),
        encoding="utf-8",
    )
    return candidate


def _make_tasks(n: int) -> list[dict]:
    return [
        {
            "task_id": f"task_{i}",
            "question": f"q{i}",
            "gold_answer": f"gold{i}",
            "prediction": f"pred{i}",
            "score": float(i) / n,
            "passed": i % 2 == 0,
            "prompt_tokens": 100 + i,
            "completion_tokens": 5,
            "retrieved": [
                {
                    "text": f"doc{j} for task{i}",
                    "score": 1.0 - 0.1 * j,
                    "source": "archival",
                    "metadata": {"rank": j},
                }
                for j in range(3)
            ],
        }
        for i in range(n)
    ]


def test_harness_writes_manifest_and_spans(tmp_path):
    candidate = _write_candidate_result(
        tmp_path=tmp_path,
        candidate_id="cand_a",
        tasks=_make_tasks(5),
    )

    harness = TraceHarness(
        run_dir=tmp_path,
        benchmark="longmemeval",
        baseline_path=None,
    )
    written = harness.record_iteration(iteration=3, candidates=[candidate])

    # Manifest exists with expected fields.
    manifest_path = tmp_path / "traces" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmark"] == "longmemeval"
    assert manifest["is_baseline_run"] is True
    assert manifest["baseline_path"] is None

    # One JSONL written, at the expected path.
    assert len(written) == 1
    expected = tmp_path / "traces" / "spans" / "iter_003" / "cand_a.jsonl"
    assert written[0] == expected
    assert expected.exists()

    # Contents reload to 5 traces with correct ids.
    traces = read_jsonl(expected)
    assert len(traces) == 5
    assert [t.task_id for t in traces] == [f"task_{i}" for i in range(5)]
    assert all(t.iteration == 3 for t in traces)
    assert all(t.candidate_id == "cand_a" for t in traces)
    assert all(t.benchmark == "longmemeval" for t in traces)


def test_harness_with_baseline_path_records_in_manifest(tmp_path):
    baseline_dir = tmp_path / "baseline_run" / "traces"
    baseline_dir.mkdir(parents=True)

    harness = TraceHarness(
        run_dir=tmp_path / "current_run",
        benchmark="longmemeval",
        baseline_path=baseline_dir,
    )
    harness.ensure_manifest()

    manifest = json.loads(
        (tmp_path / "current_run" / "traces" / "manifest.json").read_text("utf-8")
    )
    assert manifest["is_baseline_run"] is False
    assert manifest["baseline_path"] == str(baseline_dir)


def test_harness_skips_candidates_with_unreadable_result(tmp_path, capsys):
    # candidate.result_path points to a non-existent file
    candidate = CandidateResult(
        candidate_id="cand_missing",
        scaffold_name="memgpt",
        passrate=0.0,
        average_score=0.0,
        token_consuming=0,
        avg_token_consuming=0,
        avg_prompt_tokens=0,
        avg_completion_tokens=0,
        count=0,
        config={},
        result_path=str(tmp_path / "no_such.json"),
    )
    harness = TraceHarness(run_dir=tmp_path, benchmark="longmemeval")
    written = harness.record_iteration(iteration=0, candidates=[candidate])
    assert written == []


def test_harness_multi_candidate_separates_files(tmp_path):
    a = _write_candidate_result(
        tmp_path=tmp_path,
        candidate_id="cand_a",
        tasks=_make_tasks(3),
    )
    b = _write_candidate_result(
        tmp_path=tmp_path,
        candidate_id="cand_b",
        tasks=_make_tasks(2),
    )
    harness = TraceHarness(run_dir=tmp_path, benchmark="longmemeval")
    written = harness.record_iteration(iteration=1, candidates=[a, b])
    assert {p.name for p in written} == {"cand_a.jsonl", "cand_b.jsonl"}
    assert len(read_jsonl(written[0])) == 3
    assert len(read_jsonl(written[1])) == 2


def test_harness_materializes_sqlite_after_record(tmp_path):
    """End-to-end: record_iteration(0) → traces/index.db with all
    rows tagged 'baseline' (iter_0 is the implicit baseline when no
    external --trace-baseline is supplied)."""

    import sqlite3

    candidate = _write_candidate_result(
        tmp_path=tmp_path,
        candidate_id="cand_a",
        tasks=_make_tasks(4),
    )
    harness = TraceHarness(run_dir=tmp_path, benchmark="longmemeval")
    harness.record_iteration(iteration=0, candidates=[candidate])

    db_path = tmp_path / "traces" / "index.db"
    assert db_path.exists()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        traces_rows = conn.execute(
            "SELECT trace_id, iteration, task_id, passed FROM traces ORDER BY task_id"
        ).fetchall()
        diff_rows = conn.execute(
            "SELECT status FROM diffs"
        ).fetchall()
        manifest_rows = conn.execute(
            "SELECT key, value FROM manifest"
        ).fetchall()

    assert len(traces_rows) == 4
    assert all(row["iteration"] == 0 for row in traces_rows)
    # iter_0 is the implicit baseline → all 'baseline'.
    assert {row["status"] for row in diff_rows} == {"baseline"}
    by_key = {row["key"]: row["value"] for row in manifest_rows}
    assert by_key["benchmark"] == "longmemeval"


def test_harness_writes_diagnostic_markdown(tmp_path):
    """record_iteration(0) → diagnostic/iter_000.md exists with the
    simplified Baseline-Run header (iter_0 is implicit baseline)."""

    candidate = _write_candidate_result(
        tmp_path=tmp_path,
        candidate_id="cand_a",
        tasks=_make_tasks(3),
    )
    harness = TraceHarness(run_dir=tmp_path, benchmark="longmemeval")
    harness.record_iteration(iteration=0, candidates=[candidate])

    md_path = tmp_path / "traces" / "diagnostic" / "iter_000.md"
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "Baseline Run" in text
    assert "Iteration 0" in text


def test_iter1_diffs_against_self_iter0_when_no_external_baseline(tmp_path):
    """End-to-end iter_0 → iter_1 within a single run.
    iter_0 records as baseline; iter_1 must produce diff statuses
    (regressed, breakthrough, etc.) against iter_0."""

    import sqlite3

    # iter_0 — t1 passes, t2 fails
    iter0_cand = _write_candidate_result(
        tmp_path=tmp_path / "iter0",
        candidate_id="bl",
        tasks=[
            {**_make_tasks(2)[0], "task_id": "t1", "passed": True, "score": 1.0},
            {**_make_tasks(2)[1], "task_id": "t2", "passed": False, "score": 0.0},
        ],
    )
    # iter_1 — flips both: t1 fails (regressed), t2 passes (breakthrough)
    iter1_cand = _write_candidate_result(
        tmp_path=tmp_path / "iter1",
        candidate_id="opt",
        tasks=[
            {**_make_tasks(2)[0], "task_id": "t1", "passed": False, "score": 0.1},
            {**_make_tasks(2)[1], "task_id": "t2", "passed": True, "score": 0.9},
        ],
    )

    harness = TraceHarness(run_dir=tmp_path, benchmark="longmemeval")
    harness.record_iteration(iteration=0, candidates=[iter0_cand])
    harness.record_iteration(iteration=1, candidates=[iter1_cand])

    with sqlite3.connect(tmp_path / "traces" / "index.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT t.iteration, t.task_id, d.status "
            "FROM traces t JOIN diffs d USING (trace_id) "
            "ORDER BY t.iteration, t.task_id"
        ).fetchall()

    by_key = {(row["iteration"], row["task_id"]): row["status"] for row in rows}
    # iter_0 is the implicit baseline.
    assert by_key[(0, "t1")] == "baseline"
    assert by_key[(0, "t2")] == "baseline"
    # iter_1 diffs against iter_0 — t1 regressed, t2 breakthrough.
    assert by_key[(1, "t1")] == "regressed"
    assert by_key[(1, "t2")] == "breakthrough"


def test_iter1_no_baseline_status_when_iter0_skipped(tmp_path):
    """If the harness records iter_1 without iter_0 having been
    recorded (e.g. --skip-scaffold-eval), traces fall back to
    'no_baseline' rather than blowing up."""

    cand = _write_candidate_result(
        tmp_path=tmp_path,
        candidate_id="opt",
        tasks=_make_tasks(3),
    )
    harness = TraceHarness(run_dir=tmp_path, benchmark="longmemeval")
    harness.record_iteration(iteration=1, candidates=[cand])

    import sqlite3

    with sqlite3.connect(tmp_path / "traces" / "index.db") as conn:
        rows = conn.execute("SELECT status FROM diffs").fetchall()
    assert {r[0] for r in rows} == {"no_baseline"}


def test_harness_with_external_baseline_classifies_diffs(tmp_path):
    """Diff against an external baseline run: each task gets a status."""

    import sqlite3

    # Step 1: build a baseline run by running the harness without a
    # baseline_path. Two tasks: t1 passes, t2 fails.
    baseline_run = tmp_path / "baseline_run"
    baseline_candidate = _write_candidate_result(
        tmp_path=baseline_run,
        candidate_id="bl",
        tasks=[
            {**_make_tasks(2)[0], "task_id": "t1", "passed": True, "score": 1.0},
            {**_make_tasks(2)[1], "task_id": "t2", "passed": False, "score": 0.0},
        ],
    )
    bl_harness = TraceHarness(run_dir=baseline_run, benchmark="longmemeval")
    bl_harness.record_iteration(iteration=0, candidates=[baseline_candidate])

    # Step 2: run a current run that flips both tasks (t1 fails,
    # t2 passes) — should produce regressed + breakthrough.
    current_run = tmp_path / "current_run"
    curr_candidate = _write_candidate_result(
        tmp_path=current_run,
        candidate_id="opt",
        tasks=[
            {**_make_tasks(2)[0], "task_id": "t1", "passed": False, "score": 0.1},
            {**_make_tasks(2)[1], "task_id": "t2", "passed": True, "score": 0.9},
        ],
    )
    curr_harness = TraceHarness(
        run_dir=current_run,
        benchmark="longmemeval",
        baseline_path=baseline_run / "traces",
    )
    curr_harness.record_iteration(iteration=1, candidates=[curr_candidate])

    with sqlite3.connect(current_run / "traces" / "index.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT t.task_id, d.status FROM traces t JOIN diffs d USING (trace_id) "
            "ORDER BY t.task_id"
        ).fetchall()

    by_task = {row["task_id"]: row["status"] for row in rows}
    assert by_task == {"t1": "regressed", "t2": "breakthrough"}
