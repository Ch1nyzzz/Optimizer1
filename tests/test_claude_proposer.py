"""Tests for the Claude Code proposer runner branch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optimizer1.claude_runner import (
    _CLAUDE_THIRD_PARTY_ENV,
    _claude_command,
    _claude_env,
    _extract_claude_result,
    _extract_claude_tool_access,
    run_code_agent_prompt,
)


# ---- env composition ------------------------------------------------


def test_claude_env_sets_third_party_defaults(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")

    env = _claude_env(
        base_url="https://api.deepseek.com/anthropic",
        auth_token=None,
        model="deepseek-v4-pro[1m]",
    )
    assert env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-deepseek-test"
    assert env["ANTHROPIC_MODEL"] == "deepseek-v4-pro[1m]"
    for key, value in _CLAUDE_THIRD_PARTY_ENV.items():
        assert env[key] == value


def test_claude_env_explicit_token_takes_precedence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "wrong")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "also-wrong")
    env = _claude_env(
        base_url="https://api.deepseek.com/anthropic",
        auth_token="explicit-token",
        model="model-x",
    )
    assert env["ANTHROPIC_AUTH_TOKEN"] == "explicit-token"


def test_claude_env_skips_token_when_none_available(monkeypatch):
    for key in ("ANTHROPIC_AUTH_TOKEN", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    env = _claude_env(base_url=None, auth_token=None, model="m")
    assert "ANTHROPIC_AUTH_TOKEN" not in env


# ---- command shape --------------------------------------------------


def test_claude_command_contains_print_and_stream_json():
    cmd = _claude_command(model="deepseek-v4-pro[1m]")
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert cmd[cmd.index("--model") + 1] == "deepseek-v4-pro[1m]"


# ---- stream-json result extraction ---------------------------------


def _stream(events: list[dict]) -> str:
    return "\n".join(json.dumps(e) for e in events)


def test_extract_claude_result_aggregates_text_and_usage():
    events = [
        {"type": "system", "subtype": "init"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "first chunk"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "/abs/foo.py"},
                    },
                ],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
                ]
            },
        },
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "final answer"}],
                "usage": {"input_tokens": 80, "output_tokens": 20},
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "final answer",
            "total_cost_usd": 0.01,
            "session_id": "s-123",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    ]
    text, usage = _extract_claude_result(_stream(events))
    assert "first chunk" in text
    assert "final answer" in text
    assert usage is not None
    assert usage["usage"]["input_tokens"] == 180
    assert usage["usage"]["output_tokens"] == 70
    assert usage["total_cost_usd"] == 0.01
    assert usage["session_id"] == "s-123"


def test_extract_claude_tool_access_classifies_read_write_grep_bash():
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "u1",
                        "name": "Read",
                        "input": {"file_path": "/abs/wd/foo.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "u2",
                        "name": "Write",
                        "input": {
                            "file_path": "/abs/wd/bar.py",
                            "content": "line1\nline2\nline3\n",
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "u3",
                        "name": "Grep",
                        "input": {"pattern": "TODO", "path": "src/"},
                    },
                    {
                        "type": "tool_use",
                        "id": "u4",
                        "name": "Bash",
                        "input": {"command": "cat /abs/wd/baz.txt"},
                    },
                ]
            },
        }
    ]
    out = _extract_claude_tool_access(_stream(events), cwd=Path("/abs/wd"))
    assert "Read" in out["tool_counts"]
    assert out["tool_counts"]["Read"] == 1
    assert out["tool_counts"]["Write"] == 1
    assert out["tool_counts"]["Bash"] == 1
    assert "foo.py" in out["files_read"]
    assert out["files_written"]["bar.py"]["lines_written"] == 4  # 3 newlines + 1
    assert any(g["pattern"] == "TODO" for g in out["grep_requests"])
    assert "baz.txt" in out["files_read"]  # bash 'cat' parsed


# ---- dispatch -------------------------------------------------------


def test_run_code_agent_prompt_rejects_unknown_agent(tmp_path):
    with pytest.raises(ValueError, match="unsupported proposer agent"):
        run_code_agent_prompt(
            "prompt",
            agent="kimi",
            cwd=tmp_path,
            log_dir=tmp_path / "log",
            name="x",
            model="m",
        )


def test_run_code_agent_prompt_dispatches_to_claude(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_run_claude(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        # Build minimal ClaudeResult shape expected by callers
        from optimizer1.claude_runner import ClaudeResult, _empty_tool_access

        return ClaudeResult(
            returncode=0,
            timed_out=False,
            stdout="ok",
            stderr="",
            raw_stdout="",
            command=("claude", "-p"),
            usage=None,
            tool_access=_empty_tool_access(),
            duration_s=0.1,
            metrics={},
        )

    import optimizer1.claude_runner as runner

    monkeypatch.setattr(runner, "run_claude_prompt", fake_run_claude)
    out = run_code_agent_prompt(
        "hello",
        agent="claude",
        cwd=tmp_path,
        log_dir=tmp_path / "log",
        name="x",
        model="deepseek-v4-pro[1m]",
        claude_base_url="https://api.deepseek.com/anthropic",
        claude_auth_token="sk-x",
    )
    assert out.returncode == 0
    assert captured["prompt"] == "hello"
    assert captured["base_url"] == "https://api.deepseek.com/anthropic"
    assert captured["auth_token"] == "sk-x"
    assert captured["model"] == "deepseek-v4-pro[1m]"
