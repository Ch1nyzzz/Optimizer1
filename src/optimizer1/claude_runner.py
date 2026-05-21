"""Claude Code proposer runner.

This module wraps the ``claude -p`` CLI for non-interactive proposer
invocations. Optimizer1 only supports the Claude Code proposer; older
OpenCode / Codex / Kimi runners that previously lived in this file have
been removed. The historical names ``claude_runner`` and ``ClaudeResult``
are preserved as the canonical runner / result types.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


CLAUDE_EXECUTABLE = "claude"
DEFAULT_CLAUDE_MODEL = "deepseek-v4-pro[1m]"
DEFAULT_CLAUDE_BASE_URL = "https://api.deepseek.com/anthropic"
# Env vars set when routing Claude Code to a third-party endpoint.
# These are the cache-friendly defaults for DeepSeek/Kimi/etc.
_CLAUDE_THIRD_PARTY_ENV: dict[str, str] = {
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "DISABLE_TELEMETRY": "1",
}
DEFAULT_DOCKER_ENV_VARS = (
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "DIFF_EMBEDDING_MODEL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


@dataclass(frozen=True)
class ClaudeResult:
    """One Claude Code proposer invocation result."""

    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    raw_stdout: str
    command: tuple[str, ...]
    usage: dict[str, Any] | None
    tool_access: dict[str, Any]
    duration_s: float
    metrics: dict[str, Any] = field(default_factory=dict)
    # True when the stream-json transcript reported an Anthropic usage-limit
    # rejection (HTTP 429 / "you've hit your limit" / a rejected
    # ``rate_limit_event``). ``rate_limit_resets_at`` is the epoch seconds
    # the limit window resets, when the transcript carried it.
    rate_limited: bool = False
    rate_limit_resets_at: float | None = None


@dataclass(frozen=True)
class ProposerSandboxConfig:
    """Filesystem isolation settings for proposer code-agent invocations."""

    kind: str = "none"
    docker_image: str = ""
    docker_workspace: str = "/workspace"
    docker_env_vars: tuple[str, ...] = DEFAULT_DOCKER_ENV_VARS
    docker_mounts: tuple[str, ...] = ()
    docker_user: str = ""
    docker_home: str = ""


@dataclass(frozen=True)
class _PreparedAgentCommand:
    command: tuple[str, ...]
    run_cwd: Path
    extract_cwd: Path
    error: str = ""


def has_claude_cli() -> bool:
    return shutil.which(CLAUDE_EXECUTABLE) is not None


def _uses_docker_sandbox(sandbox: ProposerSandboxConfig | None) -> bool:
    return sandbox is not None and sandbox.kind.strip().lower() == "docker"


def _agent_visible_cwd(cwd: Path, *, sandbox: ProposerSandboxConfig | None) -> Path:
    if not _uses_docker_sandbox(sandbox):
        return cwd
    return Path(str(sandbox.docker_workspace or "/workspace"))


def _prepare_agent_command(
    command: tuple[str, ...],
    *,
    cwd: Path,
    sandbox: ProposerSandboxConfig | None,
    env: Mapping[str, str] | None = None,
) -> _PreparedAgentCommand:
    if sandbox is None or sandbox.kind.strip().lower() == "none":
        return _PreparedAgentCommand(command=command, run_cwd=cwd, extract_cwd=cwd)

    if not _uses_docker_sandbox(sandbox):
        return _PreparedAgentCommand(
            command=command,
            run_cwd=cwd,
            extract_cwd=cwd,
            error=f"unsupported proposer sandbox: {sandbox.kind!r}",
        )

    image = sandbox.docker_image.strip()
    if not image:
        return _PreparedAgentCommand(
            command=command,
            run_cwd=cwd,
            extract_cwd=cwd,
            error="--proposer-docker-image is required when --proposer-sandbox=docker",
        )
    if shutil.which("docker") is None:
        docker_command = ("docker", "run", "--rm", "-i", image, *command)
        return _PreparedAgentCommand(
            command=docker_command,
            run_cwd=cwd,
            extract_cwd=_agent_visible_cwd(cwd, sandbox=sandbox),
            error="docker CLI not found on PATH",
        )

    workspace = str(sandbox.docker_workspace or "/workspace")
    docker_parts: list[str] = [
        "docker",
        "run",
        "--rm",
        "-i",
        "-v",
        f"{cwd.resolve(strict=False)}:{workspace}:rw",
        "-w",
        workspace,
        "--entrypoint",
        "",
    ]
    if sandbox.docker_user.strip():
        docker_parts.extend(["--user", sandbox.docker_user.strip()])
    if sandbox.docker_home.strip():
        docker_parts.extend(["-e", f"HOME={sandbox.docker_home.strip()}"])
    env_lookup = env if env is not None else os.environ
    for env_name in _dedupe_strings(sandbox.docker_env_vars):
        if env_name in env_lookup:
            docker_parts.extend(["-e", env_name])
    for mount in sandbox.docker_mounts:
        mount_arg = _normalize_docker_mount(mount)
        if mount_arg:
            docker_parts.extend(["-v", mount_arg])
    docker_parts.append(image)
    docker_parts.extend(command)
    return _PreparedAgentCommand(
        command=tuple(docker_parts),
        run_cwd=cwd,
        extract_cwd=Path(workspace),
    )


def _dedupe_strings(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _normalize_docker_mount(spec: str) -> str:
    text = str(spec).strip()
    if not text:
        return ""
    parts = text.split(":", 1)
    if len(parts) != 2:
        return text
    host, rest = parts
    if host.startswith(("~", ".", "/")):
        host = str(Path(host).expanduser().resolve(strict=False))
    return f"{host}:{rest}"


def run_code_agent_prompt(
    prompt: str,
    *,
    agent: str,
    cwd: Path,
    log_dir: Path,
    name: str,
    model: str,
    effort: str = "",
    timeout_s: int = 2400,
    sandbox: ProposerSandboxConfig | None = None,
    claude_base_url: str | None = None,
    claude_auth_token: str | None = None,
    claude_append_system_prompt: str | None = None,
    claude_native_auth: bool = False,
    codex_model: str = "",
    codex_reasoning_effort: str = "",
    codex_home: str | None = None,
    codex_mcp_servers: dict[str, dict[str, Any]] | None = None,
) -> ClaudeResult:
    """Dispatch one proposer invocation.

    ``agent="claude"`` runs Claude Code via :func:`run_claude_prompt`.
    The ``claude_*`` keyword arguments are forwarded; ``codex_*`` are
    ignored.

    ``agent="codex"`` runs the Codex CLI via
    :func:`optimizer1.codex_runner.run_codex_prompt`. The Codex path
    uses ``codex_model`` / ``codex_reasoning_effort`` / ``codex_home``
    and registers per-invocation MCP servers from
    ``codex_mcp_servers``. Claude-only kwargs (base_url, auth_token,
    append_system_prompt, native_auth) are ignored because Codex
    authenticates via ``$CODEX_HOME/auth.json``, uses its own MCP
    override protocol, and loads its proposer contract from
    ``<workspace>/AGENTS.md`` rather than an appended system prompt.
    """

    normalized = agent.strip().lower()
    if normalized == "claude":
        return run_claude_prompt(
            prompt,
            cwd=cwd,
            log_dir=log_dir,
            name=name,
            model=model,
            effort=effort,
            timeout_s=timeout_s,
            sandbox=sandbox,
            base_url=claude_base_url,
            auth_token=claude_auth_token,
            append_system_prompt=claude_append_system_prompt,
            native_auth=claude_native_auth,
        )
    if normalized == "codex":
        from optimizer1.codex_runner import (
            DEFAULT_CODEX_MODEL,
            DEFAULT_CODEX_REASONING_EFFORT,
            run_codex_prompt,
        )

        return run_codex_prompt(
            prompt,
            cwd=cwd,
            log_dir=log_dir,
            name=name,
            model=codex_model or DEFAULT_CODEX_MODEL,
            reasoning_effort=codex_reasoning_effort or DEFAULT_CODEX_REASONING_EFFORT,
            timeout_s=timeout_s,
            sandbox=sandbox,
            codex_home=codex_home,
            mcp_servers=codex_mcp_servers,
        )
    raise ValueError(
        f"unsupported proposer agent: {agent!r}; expected 'claude' or 'codex'"
    )


def run_claude_prompt(
    prompt: str,
    *,
    cwd: Path,
    log_dir: Path,
    name: str,
    model: str = DEFAULT_CLAUDE_MODEL,
    effort: str = "",
    timeout_s: int = 2400,
    sandbox: ProposerSandboxConfig | None = None,
    base_url: str | None = None,
    auth_token: str | None = None,
    append_system_prompt: str | None = None,
    native_auth: bool = False,
) -> ClaudeResult:
    """Run ``claude -p`` non-interactively and persist logs.

    `claude -p --output-format stream-json` emits one JSON object per line
    with shapes ``{type:"system",...}`` / ``{type:"assistant", message:{...}}``
    / ``{type:"user", message:{tool_result}}`` / ``{type:"result",...}``.

    When ``append_system_prompt`` is given, ``--append-system-prompt
    <text>`` is appended so the proposer contract (the benchmark's
    self-contained skill) is delivered through the system prompt
    channel instead of the user message.

    When ``native_auth=True``, the Claude Code subscription OAuth path
    is used. Any ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` /
    ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_MODEL`` already set in the
    environment (including ``DEEPSEEK_API_KEY`` -> ``ANTHROPIC_AUTH_TOKEN``
    fallbacks) are removed from the forwarded env and never re-set. The
    ``claude`` CLI then reads its credentials from
    ``~/.claude/.credentials.json``.
    """

    cwd = cwd.resolve(strict=False)
    command = _claude_command(
        model=model,
        effort=effort,
        append_system_prompt=append_system_prompt,
        mcp_config=_claude_mcp_config_path(cwd, sandbox=sandbox),
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = _claude_env(
        base_url=base_url,
        auth_token=auth_token,
        model=model,
        native_auth=native_auth,
    )
    prepared = _prepare_agent_command(command, cwd=cwd, sandbox=sandbox, env=env)

    if prepared.error:
        result = ClaudeResult(
            returncode=None,
            timed_out=False,
            stdout="",
            stderr=prepared.error,
            raw_stdout="",
            command=prepared.command,
            usage=None,
            tool_access=_empty_tool_access(),
            duration_s=0.0,
            metrics={},
        )
        _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
        return result

    if not _uses_docker_sandbox(sandbox) and not has_claude_cli():
        result = ClaudeResult(
            returncode=None,
            timed_out=False,
            stdout="",
            stderr="claude CLI not found on PATH",
            raw_stdout="",
            command=command,
            usage=None,
            tool_access=_empty_tool_access(),
            duration_s=0.0,
            metrics={},
        )
        _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
        return result

    try:
        completed = subprocess.run(
            prepared.command,
            input=prompt,
            cwd=str(prepared.run_cwd),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )
        raw_stdout = completed.stdout or ""
        stdout, usage = _extract_claude_result(raw_stdout)
        tool_access = _extract_claude_tool_access(raw_stdout, cwd=prepared.extract_cwd)
        duration_s = time.time() - started
        metrics = _extract_session_metrics(
            usage=usage,
            tool_access=tool_access,
            duration_s=duration_s,
        )
        rate_limited, rate_limit_resets_at = _extract_rate_limit(raw_stdout)
        result = ClaudeResult(
            returncode=completed.returncode,
            timed_out=False,
            stdout=stdout,
            stderr=completed.stderr or "",
            raw_stdout=raw_stdout,
            command=prepared.command,
            usage=usage,
            tool_access=tool_access,
            duration_s=duration_s,
            metrics=metrics,
            rate_limited=rate_limited,
            rate_limit_resets_at=rate_limit_resets_at,
        )
    except subprocess.TimeoutExpired as exc:
        raw_stdout = _coerce(exc.stdout)
        tool_access = _extract_claude_tool_access(raw_stdout, cwd=prepared.extract_cwd)
        duration_s = time.time() - started
        rate_limited, rate_limit_resets_at = _extract_rate_limit(raw_stdout)
        result = ClaudeResult(
            returncode=None,
            timed_out=True,
            stdout=raw_stdout,
            stderr=_coerce(exc.stderr),
            raw_stdout=raw_stdout,
            command=prepared.command,
            usage=None,
            tool_access=tool_access,
            duration_s=duration_s,
            metrics=_extract_session_metrics(
                usage=None,
                tool_access=tool_access,
                duration_s=duration_s,
            ),
            rate_limited=rate_limited,
            rate_limit_resets_at=rate_limit_resets_at,
        )

    _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
    return result


def _claude_command(
    *,
    model: str,
    effort: str = "",
    append_system_prompt: str | None = None,
    mcp_config: Path | None = None,
) -> tuple[str, ...]:
    parts: list[str] = [
        CLAUDE_EXECUTABLE,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",  # required by some claude versions for stream-json
        "--permission-mode",
        "bypassPermissions",
    ]
    if mcp_config is not None:
        parts.extend(["--mcp-config", str(mcp_config)])
    if model:
        parts.extend(["--model", model])
    if effort:
        parts.extend(["--effort", effort])
    if append_system_prompt:
        parts.extend(["--append-system-prompt", append_system_prompt])
    return tuple(parts)


def _claude_mcp_config_path(
    cwd: Path,
    *,
    sandbox: ProposerSandboxConfig | None,
) -> Path | None:
    """Return the Claude-visible project MCP config path when present."""

    if not (cwd / ".mcp.json").is_file():
        return None
    return _agent_visible_cwd(cwd, sandbox=sandbox) / ".mcp.json"


def _claude_env(
    *,
    base_url: str | None,
    auth_token: str | None,
    model: str,
    native_auth: bool = False,
) -> dict[str, str]:
    env: dict[str, str] = dict(os.environ)
    if native_auth:
        # Strip every Anthropic override so the `claude` CLI falls back
        # to its own OAuth credentials in ~/.claude/.credentials.json.
        # In particular, drop ANTHROPIC_AUTH_TOKEN even if a project
        # .env has loaded DEEPSEEK_API_KEY into the process environment.
        for stale_key in (
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_MODEL",
        ):
            env.pop(stale_key, None)
        for key, value in _CLAUDE_THIRD_PARTY_ENV.items():
            env[key] = value
        return env

    resolved_base = (base_url or env.get("ANTHROPIC_BASE_URL") or DEFAULT_CLAUDE_BASE_URL).strip()
    env["ANTHROPIC_BASE_URL"] = resolved_base
    resolved_token = (
        auth_token
        or env.get("ANTHROPIC_AUTH_TOKEN")
        or env.get("DEEPSEEK_API_KEY")
        or env.get("ANTHROPIC_API_KEY")
        or ""
    ).strip()
    if resolved_token:
        env["ANTHROPIC_AUTH_TOKEN"] = resolved_token
    if model:
        env["ANTHROPIC_MODEL"] = model
    for key, value in _CLAUDE_THIRD_PARTY_ENV.items():
        env[key] = value
    return env


def _extract_claude_result(raw_stdout: str) -> tuple[str, dict[str, Any] | None]:
    """Parse Claude Code stream-json events.

    Returns the assistant's final text and a usage dict aggregated across
    assistant messages and the terminal `result` event.
    """

    text_chunks: list[str] = []
    usage: dict[str, Any] = {}
    for event in _jsonl_events(raw_stdout):
        et = str(event.get("type") or "")
        if et == "assistant":
            message = event.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and isinstance(block.get("text"), str)
                        ):
                            text_chunks.append(block["text"])
                msg_usage = message.get("usage")
                if isinstance(msg_usage, dict):
                    usage["usage"] = _merge_usage_dicts(
                        usage.get("usage")
                        if isinstance(usage.get("usage"), dict)
                        else {},
                        msg_usage,
                    )
        elif et == "result":
            for key in ("result", "message", "text"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    text_chunks.append(value)
                    break
            res_usage = event.get("usage")
            if isinstance(res_usage, dict):
                usage["usage"] = _merge_usage_dicts(
                    usage.get("usage") if isinstance(usage.get("usage"), dict) else {},
                    res_usage,
                )
            for key in (
                "total_cost_usd",
                "duration_ms",
                "duration_api_ms",
                "num_turns",
                "session_id",
            ):
                if key in event:
                    usage[key] = event[key]
    return "\n".join(text_chunks) or raw_stdout, usage or None


def _extract_claude_tool_access(
    raw_stdout: str, *, cwd: Path | str | None = None
) -> dict[str, Any]:
    """Walk Claude Code stream-json for tool_use blocks."""

    tool_uses: list[dict[str, Any]] = []
    files_read: dict[str, dict[str, int]] = {}
    files_written: dict[str, dict[str, int]] = {}
    grep_requests: list[dict[str, Any]] = []
    pending_outputs: dict[str, str] = {}

    # First pass: collect tool_result outputs by tool_use_id (Claude emits
    # them in subsequent {"type":"user", "message":{"content":[tool_result]}})
    for event in _jsonl_events(raw_stdout):
        if str(event.get("type")) != "user":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
            ):
                use_id = block.get("tool_use_id")
                output = block.get("content")
                if isinstance(output, list):
                    parts: list[str] = []
                    for part in output:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            parts.append(part["text"])
                    output = "\n".join(parts)
                if isinstance(use_id, str) and isinstance(output, str):
                    pending_outputs[use_id] = output

    # Second pass: walk assistant tool_use blocks
    for event in _jsonl_events(raw_stdout):
        if str(event.get("type")) != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = str(block.get("name") or "")
            tool_input = (
                block.get("input") if isinstance(block.get("input"), dict) else {}
            )
            use_id = block.get("id") or ""
            record: dict[str, Any] = {
                "id": use_id,
                "name": name,
                "input": tool_input,
            }
            output = pending_outputs.get(use_id, "")
            if output:
                record["_output"] = output
            tool_uses.append(record)
            path = _tool_path(tool_input)
            if path and name in {"Read", "read_file"}:
                rel = _make_relative(path, cwd)
                current = files_read.setdefault(rel, {"reads": 0, "lines": 0})
                current["reads"] += 1
            elif path and name in {"Write", "Edit", "apply_patch", "write_file"}:
                _add_written_lines(
                    files_written,
                    _make_relative(path, cwd),
                    _count_text_lines(
                        tool_input.get("content") or tool_input.get("new_string")
                    ),
                )
            elif name in {"Grep", "rg", "search"}:
                grep_requests.append(
                    {
                        "pattern": tool_input.get("pattern") or tool_input.get("query"),
                        "path": tool_input.get("path"),
                        "glob": tool_input.get("glob"),
                    }
                )
            elif _is_shell_tool_name(name):
                _add_shell_command_access(
                    record,
                    files_read=files_read,
                    files_written=files_written,
                    grep_requests=grep_requests,
                    cwd=cwd,
                )

    for record in tool_uses:
        record.pop("_output", None)

    return {
        "read_files": sorted(files_read),
        "grep_requests": _dedupe_dicts(grep_requests),
        "tool_uses": tool_uses,
        "tool_counts": dict(
            sorted(Counter(str(item.get("name") or "") for item in tool_uses).items())
        ),
        "files_read": dict(sorted(files_read.items())),
        "files_written": dict(sorted(files_written.items())),
        "evidence_usage": _summarize_evidence_usage(
            tool_uses=tool_uses,
            files_read=files_read,
        ),
    }


def _is_runstore_tool_name(name: str) -> bool:
    return (
        name.startswith("mcp__runstore-tools__runstore_")
        or name.startswith("runstore_")
    )


def _is_runstore_trace_tool_name(name: str) -> bool:
    return (
        name.startswith("mcp__runstore-tools__runstore_fact_state")
        or name.startswith("mcp__runstore-tools__runstore_fact_candidate_outcome")
        or name.startswith("mcp__runstore-tools__runstore_fact_compare_iterations")
        or name.startswith("mcp__runstore-tools__runstore_fact_task_history")
        or name.startswith("mcp__runstore-tools__runstore_fact_trace")
        or name.startswith("mcp__runstore-tools__runstore_link_")
        or name.startswith("mcp__runstore-tools__runstore_artifact_")
    )


def _is_runstore_mod_tool_name(name: str) -> bool:
    return (
        name.startswith("mcp__runstore-tools__runstore_fact_modification")
        or name.startswith("mcp__runstore-tools__runstore_fact_proposer_call")
        or name.startswith("mcp__runstore-tools__runstore_fact_file_history")
        or name.startswith("mcp__runstore-tools__runstore_fact_proposal")
    )


def _evidence_path_bucket(path: str) -> str | None:
    normalized = path.replace("\\", "/").lstrip("./")
    if "/workspace/" in normalized:
        normalized = normalized.split("/workspace/", 1)[1].lstrip("/")
    for marker in ("traces/", "reference_iterations/", "summaries/"):
        if normalized.startswith(marker) or f"/{marker}" in normalized:
            return marker.rstrip("/")
    return None


def _summarize_evidence_usage(
    *,
    tool_uses: list[dict[str, Any]],
    files_read: dict[str, dict[str, int]],
) -> dict[str, Any]:
    runstore_tool_calls = 0
    runstore_trace_tool_calls = 0
    runstore_mod_tool_calls = 0
    for item in tool_uses:
        name = str(item.get("name") or "")
        if _is_runstore_tool_name(name):
            runstore_tool_calls += 1
        if _is_runstore_trace_tool_name(name):
            runstore_trace_tool_calls += 1
        if _is_runstore_mod_tool_name(name):
            runstore_mod_tool_calls += 1

    raw_reads = {"traces": 0, "reference_iterations": 0, "summaries": 0}
    raw_unique = {"traces": set(), "reference_iterations": set(), "summaries": set()}
    for path, meta in files_read.items():
        bucket = _evidence_path_bucket(str(path))
        if bucket is None:
            continue
        details = meta if isinstance(meta, dict) else {}
        reads = _int_metric(details.get("reads", 0))
        if reads <= 0:
            reads = 1
        raw_reads[bucket] += reads
        raw_unique[bucket].add(str(path))

    raw_evidence_file_reads = sum(raw_reads.values())
    evidence_events = runstore_tool_calls + raw_evidence_file_reads
    return {
        "runstore_tool_calls": runstore_tool_calls,
        "runstore_trace_tool_calls": runstore_trace_tool_calls,
        "runstore_mod_tool_calls": runstore_mod_tool_calls,
        "raw_trace_file_reads": raw_reads["traces"],
        "raw_reference_file_reads": raw_reads["reference_iterations"],
        "raw_summary_file_reads": raw_reads["summaries"],
        "raw_evidence_file_reads": raw_evidence_file_reads,
        "raw_trace_unique_files": len(raw_unique["traces"]),
        "raw_reference_unique_files": len(raw_unique["reference_iterations"]),
        "raw_summary_unique_files": len(raw_unique["summaries"]),
        "evidence_usage_events": evidence_events,
        "evidence_usage_rate": (
            round(runstore_tool_calls / evidence_events, 4)
            if evidence_events
            else 0.0
        ),
    }


def _extract_rate_limit(raw_stdout: str) -> tuple[bool, float | None]:
    """Detect an Anthropic usage-limit rejection in a stream-json transcript.

    Returns ``(rate_limited, resets_at_epoch_or_None)``. Triggers on:
      * ``{"type":"result", "is_error":true, "api_error_status":429, ...}``
        (or a ``result`` text containing "hit your limit"), and/or
      * ``{"type":"rate_limit_event", "rate_limit_info":{"status":"rejected",
        "resetsAt":<epoch>, ...}}``.
    """

    rate_limited = False
    resets_at: float | None = None
    for event in _jsonl_events(raw_stdout):
        et = str(event.get("type") or "")
        if et == "result" and event.get("is_error"):
            status = event.get("api_error_status")
            text = event.get("result")
            if status == 429 or (
                isinstance(text, str) and "hit your limit" in text.lower()
            ):
                rate_limited = True
        elif et == "rate_limit_event":
            info = event.get("rate_limit_info")
            if isinstance(info, dict):
                status = str(info.get("status") or "").lower()
                overage = str(info.get("overageStatus") or "").lower()
                if status == "rejected" or overage == "rejected":
                    rate_limited = True
                raw_reset = info.get("resetsAt")
                if isinstance(raw_reset, (int, float)) and raw_reset > 0:
                    resets_at = (
                        float(raw_reset)
                        if resets_at is None
                        else max(resets_at, float(raw_reset))
                    )
    return rate_limited, resets_at


def _jsonl_events(raw_stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw_stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _merge_usage_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, (int, float)):
            merged[str(key)] = _int_metric(merged.get(str(key), 0)) + _int_metric(value)
    return merged


def _tool_path(tool_input: dict[str, Any]) -> str:
    for key in ("file_path", "path", "filename"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _empty_tool_access() -> dict[str, Any]:
    return {
        "read_files": [],
        "grep_requests": [],
        "tool_uses": [],
        "tool_counts": {},
        "files_read": {},
        "files_written": {},
    }


def _extract_session_metrics(
    *,
    usage: dict[str, Any] | None,
    tool_access: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    usage = usage or {}
    token_usage = usage.get("usage") if isinstance(usage.get("usage"), dict) else {}
    input_tokens = _int_metric(
        token_usage.get("input_tokens", token_usage.get("prompt_tokens", 0))
    )
    output_tokens = _int_metric(
        token_usage.get("output_tokens", token_usage.get("completion_tokens", 0))
    )
    cache_creation_tokens = _int_metric(token_usage.get("cache_creation_input_tokens", 0))
    cache_read_tokens = _int_metric(
        token_usage.get("cache_read_input_tokens", token_usage.get("cached_input_tokens", 0))
    )
    files_read = tool_access.get("files_read") if isinstance(tool_access, dict) else {}
    files_written = (
        tool_access.get("files_written") if isinstance(tool_access, dict) else {}
    )
    if not isinstance(files_read, dict):
        files_read = {}
    if not isinstance(files_written, dict):
        files_written = {}

    read_count = sum(_int_metric(item.get("reads", 0)) for item in files_read.values())
    read_lines = sum(_int_metric(item.get("lines", 0)) for item in files_read.values())
    write_count = sum(
        _int_metric(item.get("writes", 0)) for item in files_written.values()
    )
    written_lines = sum(
        _int_metric(item.get("lines_written", 0)) for item in files_written.values()
    )
    tool_uses = tool_access.get("tool_uses") if isinstance(tool_access, dict) else []
    if not isinstance(tool_uses, list):
        tool_uses = []

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "total_reported_tokens": (
            input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens
        ),
        "estimated_cost_usd": _float_metric(usage.get("total_cost_usd", 0.0)),
        "duration_s": round(float(duration_s), 3),
        "tool_calls": len(tool_uses),
        "tool_counts": (
            tool_access.get("tool_counts", {}) if isinstance(tool_access, dict) else {}
        ),
        "read_file_calls": read_count,
        "unique_files_read": len(files_read),
        "read_lines": read_lines,
        "write_file_calls": write_count,
        "written_lines": written_lines,
        "evidence_usage": (
            tool_access.get("evidence_usage", {})
            if isinstance(tool_access.get("evidence_usage"), dict)
            else {}
        ),
    }


def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _make_relative(filepath: str, cwd: Path | str | None) -> str:
    if not cwd:
        return filepath
    try:
        if not os.path.isabs(filepath):
            return filepath
        rel = os.path.relpath(filepath, str(cwd))
    except ValueError:
        return filepath
    if rel == ".." or rel.startswith(f"..{os.sep}"):
        return filepath
    return rel


def _is_shell_tool_name(name: object) -> bool:
    normalized = str(name or "").strip().lower()
    return normalized in {"bash", "shell", "execute_commands", "bash_command"}


def _add_shell_command_access(
    record: dict[str, Any],
    *,
    files_read: dict[str, dict[str, int]],
    files_written: dict[str, dict[str, int]],
    grep_requests: list[dict[str, Any]],
    cwd: Path | str | None,
) -> None:
    tool_input = record.get("input") if isinstance(record.get("input"), dict) else {}
    commands = _shell_commands_from_input(tool_input)
    if not commands:
        return

    output = str(record.get("_output") or "")
    shell_read_paths: list[str] = []
    shell_written_paths: list[str] = []
    for command in commands:
        parsed = _parse_shell_command_access(command)
        shell_read_paths.extend(parsed["read_paths"])
        shell_written_paths.extend(parsed["written_paths"])
        grep_requests.extend(parsed["grep_requests"])

    read_paths = _dedupe_strings_preserve_order(shell_read_paths)
    written_paths = _dedupe_strings_preserve_order(shell_written_paths)
    read_lines = _count_shell_output_lines(output) if len(read_paths) == 1 else 0
    for path in read_paths:
        current = files_read.setdefault(_make_relative(path, cwd), {"reads": 0, "lines": 0})
        current["reads"] += 1
        current["lines"] += read_lines
    for path in written_paths:
        _add_written_lines(files_written, _make_relative(path, cwd), 0)
    if read_paths:
        record["shell_files_read"] = [_make_relative(path, cwd) for path in read_paths]
    if written_paths:
        record["shell_files_written"] = [_make_relative(path, cwd) for path in written_paths]


def _shell_commands_from_input(tool_input: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    command = tool_input.get("command")
    if isinstance(command, str) and command.strip():
        commands.append(command)
    raw_commands = tool_input.get("commands")
    if isinstance(raw_commands, list):
        for item in raw_commands:
            if isinstance(item, str) and item.strip():
                commands.append(item)
            elif isinstance(item, dict):
                for key in ("command", "keystrokes", "cmd"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        commands.append(value)
                        break
    return commands


def _parse_shell_command_access(command: str) -> dict[str, Any]:
    unwrapped = _unwrap_shell_command(command)
    read_paths: list[str] = []
    written_paths: list[str] = []
    grep_requests: list[dict[str, Any]] = []

    read_paths.extend(_extract_python_read_paths(unwrapped))
    written_paths.extend(_extract_python_write_paths(unwrapped))

    for segment in _shell_command_segments(unwrapped):
        if not segment:
            continue
        tokens = _strip_env_assignments(segment)
        if not tokens:
            continue
        cmd = Path(tokens[0]).name
        if cmd in {"sudo", "env", "timeout", "time", "command"} and len(tokens) > 1:
            tokens = _strip_env_assignments(tokens[1:])
            if not tokens:
                continue
            cmd = Path(tokens[0]).name

        if cmd in {"cat", "sed", "head", "tail", "nl", "wc"}:
            read_paths.extend(_path_args(tokens[1:]))
        elif cmd == "jq":
            read_paths.extend(_jq_path_args(tokens[1:]))
        elif cmd in {"grep", "egrep", "fgrep", "rg"}:
            grep_requests.append(_grep_request_from_tokens(cmd, tokens[1:]))
        elif cmd in {"tee"}:
            written_paths.extend(_path_args(tokens[1:]))

        written_paths.extend(_redirect_paths(tokens))

    return {
        "read_paths": _dedupe_strings_preserve_order(read_paths),
        "written_paths": _dedupe_strings_preserve_order(written_paths),
        "grep_requests": _dedupe_dicts(grep_requests),
    }


def _unwrap_shell_command(command: str) -> str:
    current = command.strip()
    for _ in range(3):
        try:
            tokens = shlex.split(current)
        except ValueError:
            return current
        if len(tokens) < 3:
            return current
        exe = Path(tokens[0]).name
        if exe not in {"bash", "sh", "zsh"}:
            return current
        for idx, token in enumerate(tokens[1:], start=1):
            if token.startswith("-") and "c" in token and idx + 1 < len(tokens):
                current = tokens[idx + 1]
                break
        else:
            return current
    return current


def _shell_command_segments(command: str) -> list[list[str]]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;()<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except (TypeError, ValueError):
        try:
            tokens = shlex.split(command)
        except ValueError:
            return []

    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in {"|", "||", "&&", ";", "(", ")"}:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _strip_env_assignments(tokens: list[str]) -> list[str]:
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "env":
            idx += 1
            continue
        if "=" in token and not token.startswith("-") and token.split("=", 1)[0].isidentifier():
            idx += 1
            continue
        break
    return tokens[idx:]


def _path_args(tokens: list[str]) -> list[str]:
    paths: list[str] = []
    skip_next = False
    options_with_values = {
        "-e",
        "-f",
        "-m",
        "-n",
        "-C",
        "-A",
        "-B",
        "--context",
        "--after-context",
        "--before-context",
        "--max-count",
        "--lines",
        "--bytes",
    }
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {">", ">>", "2>", "2>>", "<", "<<"}:
            skip_next = token in {">", ">>", "2>", "2>>", "<", "<<"}
            continue
        if token in options_with_values:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        if _looks_like_path(token):
            paths.append(_clean_shell_path_token(token))
    return paths


def _jq_path_args(tokens: list[str]) -> list[str]:
    paths: list[str] = []
    filter_seen = False
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {"-f", "--from-file", "-L"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        if _looks_like_path(token):
            paths.append(_clean_shell_path_token(token))
            continue
        if not filter_seen:
            filter_seen = True
    return paths


def _grep_request_from_tokens(command_name: str, tokens: list[str]) -> dict[str, Any]:
    pattern: str | None = None
    paths: list[str] = []
    skip_next = False
    expect_pattern = False
    files_only = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {"-e", "--regexp"}:
            expect_pattern = True
            continue
        if token in {"-f", "--file", "-C", "-A", "-B", "--context", "--after-context", "--before-context"}:
            skip_next = True
            continue
        if token == "--files" and command_name == "rg":
            files_only = True
            continue
        if token.startswith("-"):
            continue
        if expect_pattern:
            pattern = token
            expect_pattern = False
            continue
        if pattern is None and not files_only:
            pattern = token
            continue
        if _looks_like_path(token):
            paths.append(_clean_shell_path_token(token))
    return {
        "pattern": pattern,
        "path": ", ".join(paths) if paths else None,
        "glob": None,
    }


def _redirect_paths(tokens: list[str]) -> list[str]:
    paths: list[str] = []
    for idx, token in enumerate(tokens[:-1]):
        if token in {">", ">>", "1>", "1>>", "2>", "2>>"} and _looks_like_path(tokens[idx + 1]):
            paths.append(_clean_shell_path_token(tokens[idx + 1]))
    for token in tokens:
        match = re.match(r"^(?:[12])?>>(.+)$", token)
        if match and _looks_like_path(match.group(1)):
            paths.append(_clean_shell_path_token(match.group(1)))
    return paths


def _extract_python_read_paths(command: str) -> list[str]:
    paths: list[str] = []
    for pattern in (
        r"(?:Path|pathlib\.Path)\(\s*['\"]([^'\"]+)['\"]\s*\)\.read_text\s*\(",
        r"\bopen\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*['\"]([^'\"]*)['\"])?",
    ):
        for match in re.finditer(pattern, command):
            path = match.group(1)
            mode = match.group(2) if len(match.groups()) > 1 else None
            if mode and any(flag in mode for flag in ("w", "a", "+")):
                continue
            if _looks_like_path(path):
                paths.append(_clean_shell_path_token(path))
    return paths


def _extract_python_write_paths(command: str) -> list[str]:
    paths: list[str] = []
    for pattern in (
        r"(?:Path|pathlib\.Path)\(\s*['\"]([^'\"]+)['\"]\s*\)\.write_text\s*\(",
        r"\bopen\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]*[wa][^'\"]*)['\"]",
    ):
        for match in re.finditer(pattern, command):
            path = match.group(1)
            if _looks_like_path(path):
                paths.append(_clean_shell_path_token(path))
    return paths


def _looks_like_path(token: str) -> bool:
    value = _clean_shell_path_token(token)
    if not value or value.startswith("$") or value in {"-", "/dev/null"}:
        return False
    if value.startswith(("/", "./", "../", "~")):
        return True
    if "/" in value or "*" in value or "?" in value:
        return True
    suffix = Path(value).suffix.lower()
    return suffix in {
        ".py",
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".toml",
        ".lock",
        ".patch",
        ".log",
        ".csv",
        ".tsv",
        ".db",
        ".pkl",
        ".npy",
    }


def _clean_shell_path_token(token: str) -> str:
    return token.strip().strip("'\"").rstrip(",:")


def _dedupe_strings_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _clean_shell_path_token(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _count_shell_output_lines(output: str) -> int:
    return len([line for line in output.splitlines() if line.strip()])


def _count_text_lines(value: object) -> int:
    if not isinstance(value, str) or not value:
        return 0
    return value.count("\n") + 1


def _add_written_lines(
    files_written: dict[str, dict[str, int]],
    path: str,
    lines: int,
) -> None:
    current = files_written.setdefault(path, {"writes": 0, "lines_written": 0})
    current["writes"] += 1
    current["lines_written"] += lines


def _int_metric(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_metric(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _write_logs(result: ClaudeResult, *, log_dir: Path, name: str, prompt: str) -> None:
    prefix = log_dir / name
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "prompt.md").write_text(prompt, encoding="utf-8")
    (prefix / "stdout.md").write_text(result.stdout or "", encoding="utf-8")
    (prefix / "stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    (prefix / "stream.jsonl").write_text(result.raw_stdout or "", encoding="utf-8")
    meta = {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "rate_limited": result.rate_limited,
        "rate_limit_resets_at": result.rate_limit_resets_at,
        "command": list(result.command),
        "usage": result.usage,
        "tool_access": result.tool_access,
        "metrics": result.metrics,
        "duration_s": result.duration_s,
    }
    (prefix / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (prefix / "tool_access.json").write_text(
        json.dumps(result.tool_access, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (prefix / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _coerce(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
