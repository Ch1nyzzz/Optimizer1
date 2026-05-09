"""Tests for the OpenCode proposer runner.

The module is still named ``claude_runner`` for compatibility with the
rest of the optimizer (see the docstring on ``ClaudeResult``), but it
only supports the OpenCode CLI in this Optimizer1 build.
"""

import json
from types import SimpleNamespace

import pytest

from memomemo.claude_runner import (
    DEFAULT_OPENCODE_MODEL,
    ProposerSandboxConfig,
    _extract_opencode_result,
    _extract_opencode_tool_access,
    _extract_session_metrics,
    run_code_agent_prompt,
    run_opencode_prompt,
)


def test_extract_opencode_result_and_tool_access_from_jsonl():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "tool_name": "Read",
                    "input": {"file_path": "/repo/src/a.py"},
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool_name": "apply_patch",
                    "input": {"path": "/repo/src/b.py", "content": "one\ntwo"},
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "result": "done",
                    "usage": {"input_tokens": 9, "output_tokens": 4},
                }
            ),
        ]
    )

    text, usage = _extract_opencode_result(raw_stdout)
    access = _extract_opencode_tool_access(raw_stdout, cwd="/repo")

    assert text == "done"
    assert usage == {"usage": {"input_tokens": 9, "output_tokens": 4}}
    assert access["files_read"] == {"src/a.py": {"reads": 1, "lines": 0}}
    assert access["files_written"] == {"src/b.py": {"writes": 1, "lines_written": 2}}


def test_extract_opencode_result_and_tool_access_from_item_events():
    raw_stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread_1"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_0",
                        "type": "agent_message",
                        "text": "I will inspect the repo.",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                        "command": "/bin/bash -lc pwd",
                        "status": "in_progress",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                        "command": "/bin/bash -lc pwd",
                        "aggregated_output": "/repo\n",
                        "exit_code": 0,
                        "status": "completed",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "item_2", "type": "agent_message", "text": "DONE"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 50,
                        "cached_input_tokens": 40,
                        "output_tokens": 7,
                    },
                }
            ),
        ]
    )

    text, usage = _extract_opencode_result(raw_stdout)
    access = _extract_opencode_tool_access(raw_stdout, cwd="/repo")
    metrics = _extract_session_metrics(usage=usage, tool_access=access, duration_s=1.2)

    assert text == "I will inspect the repo.\nDONE"
    assert usage == {
        "usage": {"input_tokens": 50, "cached_input_tokens": 40, "output_tokens": 7}
    }
    assert access["tool_counts"] == {"Shell": 1}
    assert access["tool_uses"] == [
        {"id": None, "name": "Shell", "input": {"command": "/bin/bash -lc pwd"}}
    ]
    assert metrics["input_tokens"] == 50
    assert metrics["output_tokens"] == 7
    assert metrics["cache_read_input_tokens"] == 40
    assert metrics["tool_calls"] == 1


def test_extract_opencode_tool_access_counts_shell_file_reads_and_grep_requests():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                        "command": "/bin/bash -lc \"sed -n '1,20p' /repo/src/a.py && jq '.' /repo/runs/out.json\"",
                        "aggregated_output": "line one\nline two\n",
                        "exit_code": 0,
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_2",
                        "type": "command_execution",
                        "command": "rg -n 'needle' /repo/src /repo/tests",
                        "aggregated_output": "/repo/src/a.py:1:needle\n",
                        "exit_code": 0,
                    },
                }
            ),
        ]
    )

    access = _extract_opencode_tool_access(raw_stdout, cwd="/repo")

    assert access["tool_counts"] == {"Shell": 2}
    assert access["files_read"] == {
        "runs/out.json": {"reads": 1, "lines": 0},
        "src/a.py": {"reads": 1, "lines": 0},
    }
    assert access["grep_requests"] == [
        {"pattern": "needle", "path": "/repo/src, /repo/tests", "glob": None}
    ]
    assert access["tool_uses"][0]["shell_files_read"] == ["src/a.py", "runs/out.json"]


def test_extract_session_metrics_summarizes_tokens_cost_and_tools():
    tool_access = {
        "tool_uses": [{"name": "Read"}, {"name": "Write"}],
        "tool_counts": {"Read": 1, "Write": 1},
        "files_read": {"src/a.py": {"reads": 2, "lines": 12}},
        "files_written": {"src/b.py": {"writes": 1, "lines_written": 3}},
    }
    usage = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 7,
        },
        "total_cost_usd": 0.012345,
    }

    metrics = _extract_session_metrics(
        usage=usage,
        tool_access=tool_access,
        duration_s=1.23456,
    )

    assert metrics["total_tokens"] == 15
    assert metrics["total_reported_tokens"] == 24
    assert metrics["estimated_cost_usd"] == 0.012345
    assert metrics["duration_s"] == 1.235
    assert metrics["tool_calls"] == 2
    assert metrics["read_file_calls"] == 2
    assert metrics["unique_files_read"] == 1
    assert metrics["read_lines"] == 12
    assert metrics["write_file_calls"] == 1
    assert metrics["written_lines"] == 3


def test_run_opencode_prompt_constructs_command_and_records_output(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    raw_stdout = "Plan: write a small fix.\nDONE"
    calls = []

    def fake_which(name):
        return f"/bin/{name}" if name == "opencode" else None

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=raw_stdout, stderr="")

    monkeypatch.setattr("memomemo.claude_runner.shutil.which", fake_which)
    monkeypatch.setattr("memomemo.claude_runner.subprocess.run", fake_run)

    result = run_opencode_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
        model="anthropic/claude-test",
    )

    assert result.command[:3] == ("opencode", "run", "--print-logs")
    assert result.command[-3:] == ("--model", "anthropic/claude-test", "-")
    assert calls == [
        (
            result.command,
            {
                "input": "prompt",
                "cwd": str(repo_dir.resolve()),
                "text": True,
                "capture_output": True,
                "timeout": 2400,
            },
        )
    ]
    assert result.returncode == 0
    assert result.stdout == raw_stdout
    assert result.usage is None
    meta = json.loads((tmp_path / "logs" / "iter_001" / "meta.json").read_text())
    assert meta["command"][0] == "opencode"
    assert meta["returncode"] == 0


def test_run_opencode_prompt_omits_model_flag_when_blank(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    def fake_which(name):
        return f"/bin/{name}" if name == "opencode" else None

    monkeypatch.setattr("memomemo.claude_runner.shutil.which", fake_which)
    monkeypatch.setattr(
        "memomemo.claude_runner.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = run_opencode_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
        model="",
    )

    assert "--model" not in result.command
    assert result.command == ("opencode", "run", "--print-logs", "-")


def test_run_opencode_prompt_reports_missing_cli(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    monkeypatch.setattr("memomemo.claude_runner.shutil.which", lambda name: None)

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be invoked when CLI is missing")

    monkeypatch.setattr("memomemo.claude_runner.subprocess.run", fail_run)

    result = run_opencode_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_missing",
    )

    assert result.returncode is None
    assert result.timed_out is False
    assert "opencode CLI not found" in result.stderr
    assert result.command == (
        "opencode",
        "run",
        "--print-logs",
        "--model",
        DEFAULT_OPENCODE_MODEL,
        "-",
    )
    meta = json.loads((tmp_path / "logs" / "iter_missing" / "meta.json").read_text())
    assert meta["returncode"] is None


def test_run_opencode_prompt_can_run_inside_docker_sandbox(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    raw_stdout = json.dumps(
        {
            "type": "tool_call",
            "tool_name": "Read",
            "input": {"file_path": "/workspace/src/a.py"},
        }
    )

    def fake_which(name):
        return "/bin/docker" if name == "docker" else None

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=raw_stdout, stderr="")

    monkeypatch.setattr("memomemo.claude_runner.shutil.which", fake_which)
    monkeypatch.setattr("memomemo.claude_runner.subprocess.run", fake_run)

    result = run_opencode_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_docker",
        sandbox=ProposerSandboxConfig(
            kind="docker",
            docker_image="memo-proposer:test",
        ),
    )

    assert result.command[:4] == ("docker", "run", "--rm", "-i")
    assert "memo-proposer:test" in result.command
    image_index = result.command.index("memo-proposer:test")
    assert result.command[image_index + 1] == "opencode"
    assert result.command[image_index + 2] == "run"
    assert result.command[-1] == "-"
    assert result.tool_access["files_read"] == {"src/a.py": {"reads": 1, "lines": 0}}


def test_run_code_agent_prompt_dispatches_to_opencode(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    captured = {}

    def fake_run_opencode(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "memomemo.claude_runner.run_opencode_prompt", fake_run_opencode
    )

    result = run_code_agent_prompt(
        "prompt",
        agent="opencode",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
        model="anthropic/claude-test",
    )

    assert result.stdout == "ok"
    assert captured["prompt"] == "prompt"
    assert captured["model"] == "anthropic/claude-test"
    assert captured["cwd"] == repo_dir
    assert captured["sandbox"] is None


@pytest.mark.parametrize("agent", ["claude", "codex", "kimi", "anything-else"])
def test_run_code_agent_prompt_rejects_non_opencode_agents(tmp_path, agent):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    with pytest.raises(ValueError, match="only 'opencode' is supported"):
        run_code_agent_prompt(
            "prompt",
            agent=agent,
            cwd=repo_dir,
            log_dir=tmp_path / "logs",
            name="iter_001",
            model="some-model",
        )
