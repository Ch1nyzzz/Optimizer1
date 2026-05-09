# Optimizer1 OpenCode Proposer Migration

This directory is an independent working copy derived from `/data/home/yuhan/MemoMemo`.
Do not edit `/data/home/yuhan/MemoMemo` or any files outside `/data/home/yuhan/Optimizer1`.

Goal: add an OpenCode-backed proposer implementation for the Memo optimization loop.

Scope for this pass:
- Support `--proposer-agent opencode`.
- Only support the existing `selection_policy=default` path for now.
- Keep Claude, Codex, and Kimi proposer behavior unchanged.
- Preserve the existing proposer artifact contract:
  - prompt logs under `proposer_calls/iter_XXX/`
  - `pending_eval.json` written by the proposer
  - `meta.json`, raw stdout/stderr logs, usage/tool-access metrics where available
- Use the existing `ClaudeResult`/`run_code_agent_prompt` abstraction rather than adding a parallel optimizer loop.

Expected OpenCode runner behavior:
- Use a non-interactive OpenCode CLI invocation.
- Prefer an interface compatible with `opencode run`.
- Run in the proposer workspace, pass the prompt on stdin or as a positional prompt if that is the locally supported OpenCode CLI contract.
- If `opencode` is not installed, return a clear `ClaudeResult` error like the Codex/Kimi runners do.
- Docker sandbox support should mirror the Codex runner structure where practical.
- Parse JSONL/JSON/text output robustly. It is acceptable to reuse the Codex-style extractor for OpenCode if OpenCode output events are similar, with a plain-text fallback.

Required code touch points:
- `src/optimizer1/claude_runner.py`
- `src/optimizer1/optimizer.py`
- `src/optimizer1/cli.py`
- task-specific optimizer config subclasses if they duplicate config fields
- tests under `tests/test_claude_runner.py` and `tests/test_optimizer.py`

Validation:
- Run focused tests for proposer dispatch and runner behavior.
- At minimum:
  - `python -m pytest tests/test_claude_runner.py tests/test_optimizer.py -q`
- If the full focused suite is too slow, run the specific tests touched and document what was not run.

Completion criteria:
- `--proposer-agent opencode` is accepted by the CLI.
- default-mode optimizer dispatches to `agent="opencode"` with the configured OpenCode model.
- non-default/adaptive policies either continue to work for existing agents or fail clearly for OpenCode if not implemented.
- tests pass for the implemented surface.
- End the Ralph response with `EXIT_SIGNAL: true` only when the task is complete.
