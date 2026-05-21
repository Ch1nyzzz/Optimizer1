"""Codex CLI proposer runner.

Wraps ``codex exec --json`` for non-interactive proposer invocations,
producing a :class:`~optimizer1.claude_runner.ClaudeResult` so the
optimizer dispatch can treat Claude Code and Codex uniformly.

Codex emits one JSON object per line on stdout. The shapes we rely on
(observed from ``codex-cli 0.130.0`` against ``gpt-5.5``):

  - ``{"type":"thread.started", "thread_id": ...}``
  - ``{"type":"turn.started"}``
  - ``{"type":"item.started"|"item.completed", "item": {...}}`` where
    ``item.type`` is one of:
      * ``agent_message`` — ``{id, type, text}``
      * ``command_execution`` — ``{id, type, command, aggregated_output,
        exit_code, status}``
      * ``file_change`` — ``{id, type, changes:[{path, kind}], status}``
      * ``mcp_tool_call`` — ``{id, type, server, tool, arguments,
        result:{content:[{type,text}], structured_content}, error, status}``
      * ``reasoning`` — passed through but ignored
  - ``{"type":"turn.completed", "usage":{input_tokens, cached_input_tokens,
    output_tokens, reasoning_output_tokens}}``
  - ``{"type":"turn.failed", "error":{"message": "..."}}``
  - ``{"type":"error", "message": "..."}``

Authentication: the Codex CLI reads ``$CODEX_HOME/auth.json`` (default
``~/.codex/auth.json``) for the ChatGPT OAuth token. We never inject
``OPENAI_API_KEY``; if the user wants a different identity they pass
``--codex-home`` and we forward it as ``CODEX_HOME``.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from optimizer1.claude_runner import (
    ClaudeResult,
    ProposerSandboxConfig,
    _add_shell_command_access,
    _add_written_lines,
    _dedupe_dicts,
    _empty_tool_access,
    _evidence_path_bucket,
    _extract_session_metrics,
    _float_metric,
    _int_metric,
    _is_runstore_mod_tool_name,
    _is_runstore_tool_name,
    _is_runstore_trace_tool_name,
    _make_relative,
    _prepare_agent_command,
    _uses_docker_sandbox,
    _write_logs,
)


CODEX_EXECUTABLE = "codex"
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING_EFFORT = "high"

# Pseudo tool name used when Codex emits a ``command_execution`` item.
# ``Bash`` matches what Claude Code calls its shell tool, which lets the
# shared ``_add_shell_command_access`` parser harvest files_read /
# grep_requests / files_written without modification.
_CODEX_SHELL_TOOL_NAME = "Bash"


def has_codex_cli() -> bool:
    return shutil.which(CODEX_EXECUTABLE) is not None


def run_codex_prompt(
    prompt: str,
    *,
    cwd: Path,
    log_dir: Path,
    name: str,
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
    timeout_s: int = 2400,
    sandbox: ProposerSandboxConfig | None = None,
    codex_home: str | None = None,
    mcp_servers: dict[str, dict[str, Any]] | None = None,
) -> ClaudeResult:
    """Run ``codex exec --json`` non-interactively and persist logs.

    ``mcp_servers`` is a mapping of MCP server name → ``{"command": ...,
    "args": [...], "env": {KEY: VAL, ...}}``. Each entry is injected via
    ``-c mcp_servers.<name>.command="..."`` so the registration is
    per-invocation (no pollution of ``~/.codex/config.toml``).

    Authentication is taken from ``$CODEX_HOME/auth.json``. If
    ``codex_home`` is provided, it is forwarded as ``CODEX_HOME`` in the
    subprocess environment; otherwise Codex falls back to ``~/.codex``.
    """

    cwd = cwd.resolve(strict=False)
    command = _codex_command(
        cwd=cwd,
        model=model,
        reasoning_effort=reasoning_effort,
        mcp_servers=mcp_servers or {},
        sandbox=sandbox,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = _codex_env(codex_home=codex_home)
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

    if not _uses_docker_sandbox(sandbox) and not has_codex_cli():
        result = ClaudeResult(
            returncode=None,
            timed_out=False,
            stdout="",
            stderr="codex CLI not found on PATH",
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
        stdout, usage = _extract_codex_result(raw_stdout)
        tool_access = _extract_codex_tool_access(raw_stdout, cwd=prepared.extract_cwd)
        duration_s = time.time() - started
        metrics = _extract_session_metrics(
            usage=usage,
            tool_access=tool_access,
            duration_s=duration_s,
        )
        rate_limited, rate_limit_resets_at = _extract_codex_rate_limit(raw_stdout)
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
        tool_access = _extract_codex_tool_access(raw_stdout, cwd=prepared.extract_cwd)
        duration_s = time.time() - started
        rate_limited, rate_limit_resets_at = _extract_codex_rate_limit(raw_stdout)
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


def _codex_command(
    *,
    cwd: Path,
    model: str,
    reasoning_effort: str,
    mcp_servers: dict[str, dict[str, Any]],
    sandbox: ProposerSandboxConfig | None,
) -> tuple[str, ...]:
    visible_cwd = (
        Path(str(sandbox.docker_workspace or "/workspace"))
        if _uses_docker_sandbox(sandbox)
        else cwd
    )
    parts: list[str] = [
        CODEX_EXECUTABLE,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "danger-full-access",
        "-C",
        str(visible_cwd),
    ]
    if model:
        parts.extend(["-m", model])
    if reasoning_effort:
        parts.extend(["-c", f"model_reasoning_effort={_toml_literal(reasoning_effort)}"])
    for server_name, spec in mcp_servers.items():
        parts.extend(_mcp_server_overrides(server_name, spec))
    return tuple(parts)


def _mcp_server_overrides(
    server_name: str, spec: dict[str, Any]
) -> list[str]:
    """Translate one MCP server spec into ``-c mcp_servers.<name>.*`` flags."""

    overrides: list[str] = []
    command = spec.get("command")
    if isinstance(command, str) and command:
        overrides.extend(
            [
                "-c",
                f"mcp_servers.{server_name}.command={_toml_literal(command)}",
            ]
        )
    args = spec.get("args")
    if isinstance(args, (list, tuple)):
        overrides.extend(
            [
                "-c",
                f"mcp_servers.{server_name}.args={_toml_array(args)}",
            ]
        )
    env = spec.get("env")
    if isinstance(env, dict):
        for env_key, env_val in env.items():
            if not isinstance(env_key, str) or not env_key:
                continue
            overrides.extend(
                [
                    "-c",
                    f"mcp_servers.{server_name}.env.{env_key}={_toml_literal(str(env_val))}",
                ]
            )
    # Conservative startup / per-tool timeouts so an MCP server that
    # takes a few seconds to import (we ship a fat sqlite + evidence
    # store) does not race the default 10s startup window.
    overrides.extend(
        [
            "-c",
            f"mcp_servers.{server_name}.startup_timeout_sec=30",
            "-c",
            f"mcp_servers.{server_name}.tool_timeout_sec=120",
        ]
    )
    return overrides


def _toml_literal(value: str) -> str:
    """Render a string as a TOML basic-string literal (double-quoted, escaped)."""

    escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
    return f'"{escaped}"'


def _toml_array(values: list[Any] | tuple[Any, ...]) -> str:
    rendered = ",".join(_toml_literal(str(item)) for item in values)
    return f"[{rendered}]"


def _codex_env(*, codex_home: str | None) -> dict[str, str]:
    env: dict[str, str] = dict(os.environ)
    home = (codex_home or "").strip()
    if home:
        env["CODEX_HOME"] = home
    # Codex reads ChatGPT OAuth from $CODEX_HOME/auth.json. Do not
    # forward an OPENAI_API_KEY — that would silently switch the
    # auth mode away from chatgpt and reject gpt-5.5.
    env.pop("OPENAI_API_KEY", None)
    return env


# ---- JSONL event parsing ----------------------------------------


def _jsonl_events(raw_stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _extract_codex_result(raw_stdout: str) -> tuple[str, dict[str, Any] | None]:
    """Return the agent's final text and an aggregated usage dict.

    Final text = concatenation of every ``agent_message`` ``text``
    payload in completion order. Usage is taken from the terminal
    ``turn.completed`` event and normalized into the shape the shared
    ``_extract_session_metrics`` expects (``{"usage": {...}}``).
    """

    text_chunks: list[str] = []
    usage_raw: dict[str, Any] | None = None
    cost_total: float = 0.0
    cost_seen = False
    for event in _jsonl_events(raw_stdout):
        et = str(event.get("type") or "")
        if et == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text:
                    text_chunks.append(text)
        elif et == "turn.completed":
            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict):
                usage_raw = raw_usage
            cost = event.get("total_cost_usd")
            if isinstance(cost, (int, float)):
                cost_total += float(cost)
                cost_seen = True

    if usage_raw is None and not cost_seen:
        return "\n".join(text_chunks) or raw_stdout, None

    normalized: dict[str, int] = {}
    if usage_raw:
        normalized = {
            "input_tokens": _int_metric(usage_raw.get("input_tokens", 0)),
            "output_tokens": _int_metric(usage_raw.get("output_tokens", 0)),
            "cache_read_input_tokens": _int_metric(
                usage_raw.get("cached_input_tokens", 0)
            ),
            "reasoning_output_tokens": _int_metric(
                usage_raw.get("reasoning_output_tokens", 0)
            ),
        }
    payload: dict[str, Any] = {"usage": normalized}
    if cost_seen:
        payload["total_cost_usd"] = round(cost_total, 6)
    return "\n".join(text_chunks) or raw_stdout, payload


def _extract_codex_tool_access(
    raw_stdout: str, *, cwd: Path | str | None = None
) -> dict[str, Any]:
    """Walk Codex JSONL events and build the same tool_access dict shape
    Claude Code produces.

    Codex calls everything through three item types:

      * ``mcp_tool_call`` → mapped to ``mcp__<server>__<tool>`` so the
        existing evidence-usage classifiers match unchanged.
      * ``command_execution`` → mapped to the pseudo tool name ``Bash``;
        the shared shell-command parser extracts files_read /
        files_written / grep_requests from the command string.
      * ``file_change`` → contributes directly to files_written, with
        the pseudo tool name ``FileChange``.
    """

    tool_uses: list[dict[str, Any]] = []
    files_read: dict[str, dict[str, int]] = {}
    files_written: dict[str, dict[str, int]] = {}
    grep_requests: list[dict[str, Any]] = []

    for event in _jsonl_events(raw_stdout):
        if str(event.get("type")) != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")

        if item_type == "mcp_tool_call":
            server = str(item.get("server") or "")
            tool = str(item.get("tool") or "")
            qualified = f"mcp__{server}__{tool}" if server and tool else (tool or server)
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            record: dict[str, Any] = {
                "id": item.get("id") or "",
                "name": qualified,
                "input": arguments,
                "status": item.get("status"),
            }
            output = _flatten_mcp_result(item.get("result"))
            if output:
                record["output"] = output
            error = item.get("error")
            if error:
                record["error"] = error
            tool_uses.append(record)
            continue

        if item_type == "command_execution":
            command = item.get("command")
            if not isinstance(command, str) or not command:
                continue
            record = {
                "id": item.get("id") or "",
                "name": _CODEX_SHELL_TOOL_NAME,
                "input": {"command": command},
                "status": item.get("status"),
                "exit_code": item.get("exit_code"),
                "_output": item.get("aggregated_output") or "",
            }
            tool_uses.append(record)
            _add_shell_command_access(
                record,
                files_read=files_read,
                files_written=files_written,
                grep_requests=grep_requests,
                cwd=cwd,
            )
            record.pop("_output", None)
            continue

        if item_type == "file_change":
            changes = item.get("changes")
            if not isinstance(changes, list):
                continue
            change_paths: list[dict[str, str]] = []
            for change in changes:
                if not isinstance(change, dict):
                    continue
                path = change.get("path")
                kind = change.get("kind") or "update"
                if not isinstance(path, str) or not path:
                    continue
                rel = _make_relative(path, cwd)
                _add_written_lines(files_written, rel, 0)
                change_paths.append({"path": rel, "kind": str(kind)})
            if change_paths:
                tool_uses.append(
                    {
                        "id": item.get("id") or "",
                        "name": "FileChange",
                        "input": {"changes": change_paths},
                        "status": item.get("status"),
                    }
                )
            continue

        # agent_message / reasoning / unknown — not a tool use.
        if item_type in {"agent_message", "reasoning", ""}:
            continue

    return {
        "read_files": sorted(files_read),
        "grep_requests": _dedupe_dicts(grep_requests),
        "tool_uses": tool_uses,
        "tool_counts": dict(
            sorted(Counter(str(item.get("name") or "") for item in tool_uses).items())
        ),
        "files_read": dict(sorted(files_read.items())),
        "files_written": dict(sorted(files_written.items())),
        "evidence_usage": _summarize_codex_evidence_usage(
            tool_uses=tool_uses,
            files_read=files_read,
        ),
    }


def _flatten_mcp_result(result: Any) -> str:
    """Pull the text payload out of an MCP CallToolResult-like dict."""

    if not isinstance(result, dict):
        return ""
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _summarize_codex_evidence_usage(
    *,
    tool_uses: list[dict[str, Any]],
    files_read: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Mirror :func:`claude_runner._summarize_evidence_usage` for Codex.

    Kept inline (not imported) because Claude's helper inspects the
    Claude-flavored tool_uses list ; we want the exact same metric
    surface but driven by Codex's ``mcp__<server>__<tool>`` names.
    """

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
    raw_unique: dict[str, set[str]] = {
        "traces": set(),
        "reference_iterations": set(),
        "summaries": set(),
    }
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


def _extract_codex_rate_limit(raw_stdout: str) -> tuple[bool, float | None]:
    """Best-effort rate-limit / usage-cap detection on a Codex stream.

    Codex surfaces upstream throttling as a ``turn.failed`` with one of
    these error message shapes:

      * Provider 429: ``"status":429`` / ``Too Many Requests`` /
        ``rate_limit`` / ``rate limit``.
      * ChatGPT account usage cap (what gpt-5.5 hits when the daily
        quota for the logged-in ChatGPT account is exhausted):
        ``"You've hit your usage limit. Visit ... or try again at 3:52 AM."``.
        Specifically, the substrings ``"usage limit"``, ``"hit your
        limit"``, and ``"purchase more credits"`` all appear in this
        family; we match any.

    When a ChatGPT cap message carries a ``try again at HH:MM (AM|PM)``
    clause, we parse the wall-clock and convert it to the next-future
    epoch (today if still upcoming, otherwise tomorrow) so the
    optimizer's wait-on-rate-limit loop sleeps until the actual reset
    instead of using the default 30-minute fallback.
    """

    rate_limited = False
    resets_at: float | None = None
    for event in _jsonl_events(raw_stdout):
        et = str(event.get("type") or "")
        msg = ""
        if et == "error":
            msg = str(event.get("message") or "")
        elif et == "turn.failed":
            err = event.get("error")
            if isinstance(err, dict):
                msg = str(err.get("message") or "")
        if not msg:
            continue
        lowered = msg.lower()
        if (
            '"status":429' in msg
            or "status: 429" in lowered
            or "too many requests" in lowered
            or "rate_limit" in lowered
            or "rate limit" in lowered
            or "usage limit" in lowered
            or "hit your limit" in lowered
            or "purchase more credits" in lowered
        ):
            rate_limited = True
            parsed = _parse_try_again_time(msg)
            if parsed is not None:
                resets_at = parsed if resets_at is None else max(resets_at, parsed)
    return rate_limited, resets_at


def _parse_try_again_time(msg: str) -> float | None:
    """Parse ``try again at HH:MM (AM|PM)`` out of a Codex usage-cap message.

    Returns the next-future epoch matching that wall-clock in the local
    timezone, or ``None`` when no clock is present. Today if the clock
    is still upcoming, otherwise tomorrow (the reset is always within
    the next 24h for ChatGPT usage caps).
    """

    import re
    import time as _time
    from datetime import datetime, timedelta

    match = re.search(
        r"try again at\s+(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?",
        msg,
    )
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    meridiem = (match.group(3) or "").upper()
    if meridiem == "PM" and hour != 12:
        hour += 12
    elif meridiem == "AM" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    now = datetime.now()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return _time.mktime(candidate.timetuple())


def _coerce(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


__all__ = [
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_REASONING_EFFORT",
    "has_codex_cli",
    "run_codex_prompt",
]
