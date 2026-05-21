#!/usr/bin/env bash
# Launch a Codex proposer run (gpt-5.5, reasoning_effort=high) against the
# LoCoMo memory benchmark.
#
# Requirements (one-time):
#   1. `codex login` on this host. Confirms with `codex login status` ->
#      "Logged in using ChatGPT". Credentials live in ~/.codex/auth.json.
#   2. The Codex proposer reads its identity from $CODEX_HOME/auth.json. We
#      do NOT export OPENAI_API_KEY — that would silently switch Codex to
#      api-key auth and reject gpt-5.5 ("not supported when using Codex with
#      a ChatGPT account").
#   3. An OpenAI-compatible eval server for the scaffolds, e.g. vLLM
#      Qwen3-8B on http://127.0.0.1:8002/v1 (override with EVAL_BASE_URL).
#   4. TOGETHER_API_KEY in .env (used by trace_similar MCP for embeddings).
#
# Usage:
#   bash scripts/launch_codex_gpt55_locomo.sh [TIMESTAMP]
#
# Optional env overrides:
#   ITERATIONS=30 LIMIT=80 EVAL_WORKERS=128 \
#   CODEX_REASONING_EFFORT=high SELECTION_POLICY=default \
#   bash scripts/launch_codex_gpt55_locomo.sh

set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${TOGETHER_API_KEY:-}" ]; then
  echo "error: TOGETHER_API_KEY not set in .env (needed by trace_similar MCP for embeddings)" >&2
  exit 1
fi

# trace_similar MCP tool uses an OpenAI-compatible embedding endpoint; we
# point it at Together AI. These vars only affect the MCP subprocess, NOT
# Codex itself (codex_runner._codex_env strips OPENAI_API_KEY).
export OPENAI_API_KEY="${TOGETHER_API_KEY}"
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-30}"
LIMIT="${LIMIT:-80}"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
EVAL_MODEL="${EVAL_MODEL:-Qwen3-8B}"
EVAL_BASE_URL="${EVAL_BASE_URL:-http://127.0.0.1:8002/v1}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.5}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-high}"
SELECTION_POLICY="${SELECTION_POLICY:-default}"

run_id="locomo_memgpt_codex_${CODEX_MODEL//./_}_${CODEX_REASONING_EFFORT}_${SELECTION_POLICY}_iter${ITERATIONS}_train${LIMIT}_${TS}"
mkdir -p logs runs
log_path="logs/${run_id}.log"
status_file="logs/launch_${run_id}.status"

if [ -d "runs/${run_id}" ]; then
  echo "runs/${run_id} already exists; refusing to clobber" >&2
  exit 1
fi
: > "$status_file"
printf '[%s] START %s\n[%s] LOG   %s\n' \
  "$(date -Is)" "$run_id" "$(date -Is)" "$log_path" >> "$status_file"

setsid nohup python -m optimizer1.cli optimize \
  --locomo \
  --run-id "$run_id" \
  --iterations "$ITERATIONS" \
  --split train \
  --limit "$LIMIT" \
  --eval-workers "$EVAL_WORKERS" \
  --model "$EVAL_MODEL" \
  --base-url "$EVAL_BASE_URL" \
  --scaffolds memgpt_source \
  --proposer-agent codex \
  --codex-model "$CODEX_MODEL" \
  --codex-reasoning-effort "$CODEX_REASONING_EFFORT" \
  --selection-policy "$SELECTION_POLICY" \
  > "$log_path" 2>&1 < /dev/null &

pid=$!
printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
printf '%s %s %s\n' "$pid" "$run_id" "$log_path"
printf 'status: %s\n' "$status_file"
