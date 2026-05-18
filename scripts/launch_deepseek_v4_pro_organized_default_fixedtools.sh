#!/usr/bin/env bash
# Launch organized default fixedtools runs across LoCoMo and LongMemEval-S.
# This arm keeps the default selection policy and clean patch base, but
# enables organized state.md plus EvidenceStore MCP tools.
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  printf 'fatal: DEEPSEEK_API_KEY is not set; populate .env or export it.\n' >&2
  exit 2
fi

unset ANTHROPIC_API_KEY
export ANTHROPIC_AUTH_TOKEN="${DEEPSEEK_API_KEY}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.deepseek.com/anthropic}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-deepseek-v4-pro[1m]}"

if [ -n "${TOGETHER_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${TOGETHER_API_KEY}"
  export OPENAI_BASE_URL="https://api.together.xyz/v1"
fi
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-50}"
TASKS="${TASKS:-locomo,longmemeval}"
CLAUDE_MODEL="${CLAUDE_MODEL:-deepseek-v4-pro[1m]}"
CLAUDE_BASE_URL="${CLAUDE_BASE_URL:-https://api.deepseek.com/anthropic}"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
EVAL_MODEL="${EVAL_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}"
EVAL_BASE_URL="${EVAL_BASE_URL:-http://127.0.0.1:8002/v1}"
DOCKER_USER_SPEC="${DOCKER_USER_SPEC:-$(id -u):$(id -g)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-docker-claude:latest}"

mkdir -p logs runs
status_file="logs/launch_deepseek_v4_pro_organized_default_fixedtools_${TS}.status"
: > "$status_file"
printf 'timestamp=%s\n' "$TS" >> "$status_file"
printf 'mode=organized_default_fixedtools\n' >> "$status_file"

contains() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }

start_one() {
  local task="$1"
  local task_label task_args=() run_id log_path

  if [ "$task" = "locomo" ]; then
    task_label="locomo"
    task_args=(--locomo)
  elif [ "$task" = "longmemeval" ]; then
    task_label="longmemeval_s"
    task_args=(--longmemeval --longmemeval-variant s)
  else
    printf '[%s] SKIP unknown_task=%s\n' "$(date -Is)" "$task" >> "$status_file"
    return 0
  fi

  run_id="${task_label}_deepseek_v4_pro_organized_fixedtools_iter${ITERATIONS}_${TS}"
  if [ -d "runs/${run_id}" ]; then
    printf '[%s] SKIP %s existing_run_dir\n' "$(date -Is)" "$run_id" >> "$status_file"
    return 0
  fi

  log_path="logs/${run_id}.log"
  printf '[%s] START %s\n' "$(date -Is)" "$run_id" >> "$status_file"
  printf '[%s] LOG   %s\n' "$(date -Is)" "$log_path" >> "$status_file"

  setsid python -m optimizer1.cli optimize \
    "${task_args[@]}" \
    --selection-policy default \
    --organized \
    --run-id "$run_id" \
    --iterations "$ITERATIONS" \
    --split train \
    --scaffolds memgpt_source \
    --scaffold-extra-json @configs/source_memory.example.json \
    --eval-workers "$EVAL_WORKERS" \
    --model "$EVAL_MODEL" \
    --base-url "$EVAL_BASE_URL" \
    --proposer-agent claude \
    --claude-model "$CLAUDE_MODEL" \
    --claude-base-url "$CLAUDE_BASE_URL" \
    --claude-effort max \
    --force-budget high \
    --proposer-sandbox docker \
    --proposer-docker-image "$DOCKER_IMAGE" \
    --proposer-docker-user "$DOCKER_USER_SPEC" \
    --proposer-docker-home /tmp \
    --proposer-docker-env DEEPSEEK_API_KEY \
    --proposer-docker-env ANTHROPIC_AUTH_TOKEN \
    --proposer-docker-env ANTHROPIC_BASE_URL \
    --proposer-docker-env ANTHROPIC_MODEL \
    --proposer-docker-env OPENAI_API_KEY \
    --proposer-docker-env OPENAI_BASE_URL \
    --proposer-docker-env DIFF_EMBEDDING_MODEL \
    > "$log_path" 2>&1 < /dev/null &

  local pid=$!
  printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
  printf '%s %s %s\n' "$pid" "$run_id" "$log_path"
}

for task in locomo longmemeval; do
  contains "$TASKS" "$task" || continue
  start_one "$task"
done

printf '\nstatus: %s\n' "$status_file"
