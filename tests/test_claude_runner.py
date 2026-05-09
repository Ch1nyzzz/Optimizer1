"""Tests for the Claude Code proposer runner.

Optimizer1 only supports Claude Code as the proposer; the OpenCode /
Codex / Kimi runners that previously lived alongside it have been
removed. These tests cover stream-json parsing, command construction,
the docker sandbox integration, and the agent dispatch surface.
"""

import json
import os
from types import SimpleNamespace

import pytest

from optimizer1.claude_runner import (
    DEFAULT_CLAUDE_MODEL,
    ProposerSandboxConfig,
    _claude_env,
    _extract_claude_result,
    _extract_claude_tool_access,
    _extract_session_metrics,
    run_claude_prompt,
    run_code_agent_prompt,
)


def test_extract_claude_result_collects_assistant_text_and_usage():
    raw_stdout = "\n".join(
        [
            json.dumps({"type": "system", "session_id": "abc"}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "Inspecting the repo."},
                        ],
                        "usage": {"input_tokens": 12, "output_tokens": 3},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "DONE"}],
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "result": "DONE",
                    "usage": {"cache_read_input_tokens": 40},
                    "total_cost_usd": 0.0042,
                    "duration_ms": 1200,
                    "session_id": "abc",
                }
            ),
        ]
    )

    text, usage = _extract_claude_result(raw_stdout)

    assert text == "Inspecting the repo.\nDONE\nDONE"
    assert usage["usage"] == {
        "input_tokens": 17,
        "output_tokens": 5,
        "cache_read_input_tokens": 40,
    }
    assert usage["total_cost_usd"] == 0.0042
    assert usage["duration_ms"] == 1200
    assert usage["session_id"] == "abc"


def test_extract_claude_tool_access_records_tool_uses_and_files():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "Read",
                                "input": {"file_path": "/repo/src/a.py"},
                            },
                            {
                                "type": "tool_use",
                                "id": "tool_2",
                                "name": "Edit",
                                "input": {
                                    "file_path": "/repo/src/b.py",
                                    "new_string": "one\ntwo\nthree",
                                },
                            },
                            {
                                "type": "tool_use",
                                "id": "tool_3",
                                "name": "Grep",
                                "input": {
                                    "pattern": "needle",
                                    "path": "/repo/src",
                                },
                            },
                        ],
                    },
                }
            ),
        ]
    )

    access = _extract_claude_tool_access(raw_stdout, cwd="/repo")

    assert access["tool_counts"] == {"Read": 1, "Edit": 1, "Grep": 1}
    assert access["files_read"] == {"src/a.py": {"reads": 1, "lines": 0}}
    assert access["files_written"] == {
        "src/b.py": {"writes": 1, "lines_written": 3}
    }
    assert access["grep_requests"] == [
        {"pattern": "needle", "path": "/repo/src", "glob": None}
    ]


def test_extract_claude_tool_access_walks_bash_commands():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "Bash",
                                "input": {
                                    "command": "sed -n '1,20p' /repo/src/a.py && rg -n 'needle' /repo/src",
                                },
                            },
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_1",
                                "content": [
                                    {"type": "text", "text": "line one\nline two\n"}
                                ],
                            }
                        ]
                    },
                }
            ),
        ]
    )

    access = _extract_claude_tool_access(raw_stdout, cwd="/repo")

    assert access["tool_counts"] == {"Bash": 1}
    assert access["files_read"] == {"src/a.py": {"reads": 1, "lines": 2}}
    assert access["grep_requests"] == [
        {"pattern": "needle", "path": "/repo/src", "glob": None}
    ]
    assert access["tool_uses"][0]["shell_files_read"] == ["src/a.py"]


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


def test_run_claude_prompt_constructs_command_and_records_output(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "Plan: write a small fix."}],
                    },
                }
            ),
            json.dumps({"type": "result", "result": "DONE"}),
        ]
    )
    calls = []

    def fake_which(name):
        return f"/bin/{name}" if name == "claude" else None

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=raw_stdout, stderr="")

    monkeypatch.setattr("optimizer1.claude_runner.shutil.which", fake_which)
    monkeypatch.setattr("optimizer1.claude_runner.subprocess.run", fake_run)

    result = run_claude_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
        model="claude-sonnet-4-6",
        base_url="https://example.test/anthropic",
        auth_token="sk-test",
    )

    assert result.command[:7] == (
        "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
    )
    assert result.command[-2:] == ("--model", "claude-sonnet-4-6")
    assert "--agent" not in result.command
    assert result.returncode == 0
    assert result.stdout.endswith("DONE")
    assert calls and calls[0][0] == result.command
    sent_kwargs = calls[0][1]
    assert sent_kwargs["input"] == "prompt"
    assert sent_kwargs["cwd"] == str(repo_dir.resolve())
    env = sent_kwargs["env"]
    assert env["ANTHROPIC_BASE_URL"] == "https://example.test/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-test"
    assert env["ANTHROPIC_MODEL"] == "claude-sonnet-4-6"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"

    meta = json.loads((tmp_path / "logs" / "iter_001" / "meta.json").read_text())
    assert meta["command"][0] == "claude"
    assert meta["returncode"] == 0


def test_run_claude_prompt_appends_agent_flag(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    monkeypatch.setattr(
        "optimizer1.claude_runner.shutil.which",
        lambda name: f"/bin/{name}" if name == "claude" else None,
    )
    monkeypatch.setattr(
        "optimizer1.claude_runner.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = run_claude_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
        model="",
        agent_name="proposer",
    )

    assert result.command[-2:] == ("--agent", "proposer")
    assert "--model" not in result.command


def test_run_claude_prompt_reports_missing_cli(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    monkeypatch.setattr("optimizer1.claude_runner.shutil.which", lambda name: None)

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be invoked when CLI is missing")

    monkeypatch.setattr("optimizer1.claude_runner.subprocess.run", fail_run)

    result = run_claude_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_missing",
    )

    assert result.returncode is None
    assert result.timed_out is False
    assert "claude CLI not found" in result.stderr
    assert result.command == (
        "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        DEFAULT_CLAUDE_MODEL,
    )
    meta = json.loads((tmp_path / "logs" / "iter_missing" / "meta.json").read_text())
    assert meta["returncode"] is None


def test_run_claude_prompt_can_run_inside_docker_sandbox(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    raw_stdout = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "Read",
                        "input": {"file_path": "/workspace/src/a.py"},
                    }
                ]
            },
        }
    )

    def fake_which(name):
        return "/bin/docker" if name == "docker" else None

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=raw_stdout, stderr="")

    monkeypatch.setattr("optimizer1.claude_runner.shutil.which", fake_which)
    monkeypatch.setattr("optimizer1.claude_runner.subprocess.run", fake_run)

    result = run_claude_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_docker",
        sandbox=ProposerSandboxConfig(
            kind="docker",
            docker_image="docker-claude:test",
        ),
    )

    assert result.command[:4] == ("docker", "run", "--rm", "-i")
    assert "docker-claude:test" in result.command
    image_index = result.command.index("docker-claude:test")
    assert result.command[image_index + 1] == "claude"
    assert result.command[image_index + 2] == "-p"
    # Files read paths are mapped relative to the docker workspace.
    assert result.tool_access["files_read"] == {"src/a.py": {"reads": 1, "lines": 0}}


def test_claude_env_falls_back_to_deepseek_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    env = _claude_env(base_url=None, auth_token=None, model="deepseek-v4-pro[1m]")

    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-deepseek"
    assert env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert env["ANTHROPIC_MODEL"] == "deepseek-v4-pro[1m]"


def test_run_code_agent_prompt_dispatches_to_claude(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    captured: dict = {}

    def fake_run_claude(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "optimizer1.claude_runner.run_claude_prompt", fake_run_claude
    )

    result = run_code_agent_prompt(
        "prompt",
        agent="claude",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
        model="claude-sonnet-4-6",
        claude_base_url="https://api.test/anthropic",
        claude_auth_token="sk-test",
        claude_agent_name="proposer",
    )

    assert result.stdout == "ok"
    assert captured["prompt"] == "prompt"
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["cwd"] == repo_dir
    assert captured["base_url"] == "https://api.test/anthropic"
    assert captured["auth_token"] == "sk-test"
    assert captured["agent_name"] == "proposer"
    assert captured["sandbox"] is None


@pytest.mark.parametrize("agent", ["opencode", "codex", "kimi", "anything-else"])
def test_run_code_agent_prompt_rejects_unsupported_agents(tmp_path, agent):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    with pytest.raises(ValueError, match="unsupported proposer agent"):
        run_code_agent_prompt(
            "prompt",
            agent=agent,
            cwd=repo_dir,
            log_dir=tmp_path / "logs",
            name="iter_001",
            model="some-model",
        )
