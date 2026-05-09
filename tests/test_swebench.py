from __future__ import annotations

import json

from memomemo.swebench import (
    DEFAULT_MINI_SWE_AGENT_NAME,
    SwebenchInstance,
    _format_command,
    load_swebench_instances,
    run_swebench_frontier,
)
from memomemo.swebench_optimizer import SwebenchOptimizer, SwebenchOptimizerConfig


def test_load_swebench_instances_from_jsonl_selects_split_and_limit(tmp_path) -> None:
    data_path = tmp_path / "instances.jsonl"
    rows = [
        {"instance_id": "a", "problem_statement": "fix a", "split": "train"},
        {"instance_id": "b", "problem_statement": "fix b", "split": "test"},
        {"instance_id": "c", "problem_statement": "fix c", "split": "train"},
    ]
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    instances = load_swebench_instances(data_path, split="train", limit=1)

    assert [item.task_id for item in instances] == ["a"]


def test_run_swebench_frontier_dry_run_writes_candidate_result(tmp_path) -> None:
    data_path = tmp_path / "instances.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "instance_id": "repo__issue-1",
                    "problem_statement": "Fix the regression.",
                    "repo": "owner/repo",
                    "split": "train",
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = run_swebench_frontier(
        out_dir=tmp_path / "run",
        data_path=data_path,
        dry_run=True,
    )

    assert summary["benchmark"] == "swebench"
    assert summary["target_system"] == DEFAULT_MINI_SWE_AGENT_NAME
    assert summary["count"] == 1
    result_path = summary["candidates"][0]["result_path"]
    payload = json.loads(open(result_path, encoding="utf-8").read())
    assert payload["candidate"]["passrate"] == 0.0
    assert payload["tasks"][0]["metadata"]["dry_run"] is True
    assert payload["tasks"][0]["metadata"]["patch_path"].startswith(str(tmp_path))


def test_swebench_command_placeholders_are_absolute(tmp_path) -> None:
    source = tmp_path / "mini-swe-agent"
    task_dir = tmp_path / "run" / "agent_runs" / "candidate" / "repo__issue-1"
    instance_path = task_dir / "instance.json"
    patch_path = task_dir / "patch.diff"
    instance = SwebenchInstance(task_id="repo__issue-1", problem_statement="Fix it.")

    command = _format_command(
        "python runner.py --source-path {source_path} --instance-path {instance_path} "
        "--patch-path {patch_path} --task-dir {task_dir}",
        source_path=source,
        task_dir=task_dir,
        instance_path=instance_path,
        patch_path=patch_path,
        instance=instance,
    )

    assert command == [
        "python",
        "runner.py",
        "--source-path",
        str(source.resolve()),
        "--instance-path",
        str(instance_path.resolve()),
        "--patch-path",
        str(patch_path.resolve()),
        "--task-dir",
        str(task_dir.resolve()),
    ]


def test_swebench_instance_serializes_instance_id_alias() -> None:
    instance = SwebenchInstance(task_id="repo__issue-1", problem_statement="Fix it.")

    payload = instance.to_dict()

    assert payload["task_id"] == "repo__issue-1"
    assert payload["instance_id"] == "repo__issue-1"


def test_swebench_optimizer_copies_mini_source_snapshot(tmp_path) -> None:
    source = tmp_path / "mini-swe-agent"
    package = source / "mini_swe_agent"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    data_path = tmp_path / "instances.json"
    data_path.write_text(
        json.dumps([{"instance_id": "x", "problem_statement": "fix x"}]),
        encoding="utf-8",
    )
    optimizer = SwebenchOptimizer(
        SwebenchOptimizerConfig(
            run_id="r",
            out_dir=tmp_path / "run",
            data_path=data_path,
            mini_swe_agent_source_path=source,
            dry_run=True,
        )
    )
    call_dir = tmp_path / "call"
    call_dir.mkdir()

    snapshot = optimizer._build_source_snapshot_workspace(
        iteration=1,
        source_family=DEFAULT_MINI_SWE_AGENT_NAME,
        call_dir=call_dir,
        target_system=DEFAULT_MINI_SWE_AGENT_NAME,
    )

    copied = snapshot / "candidate" / "upstream_source" / "mini-swe-agent"
    assert (copied / "mini_swe_agent" / "__init__.py").exists()
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"] == "swebench"
    assert manifest["mini_swe_agent_source"] == str(copied)


def test_swebench_candidate_uses_extra_source_project_path(tmp_path) -> None:
    default_source = tmp_path / "default-mini"
    edited_source = tmp_path / "edited-mini"
    default_source.mkdir()
    edited_source.mkdir()
    optimizer = SwebenchOptimizer(
        SwebenchOptimizerConfig(
            run_id="r",
            out_dir=tmp_path / "run",
            data_path=tmp_path / "instances.json",
            mini_swe_agent_source_path=default_source,
            dry_run=True,
        )
    )
    candidate = {
        "name": "edited",
        "extra": {
            "source_project_path": str(edited_source),
        },
    }

    optimizer._normalize_candidate_source_project_path(candidate)

    assert candidate["source_project_path"] == str(edited_source)
