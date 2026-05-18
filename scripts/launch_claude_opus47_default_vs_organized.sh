#!/usr/bin/env bash
# Launch default-vs-organized runs across LoCoMo and LongMemEval with
# Claude Opus 4.7 via Claude Code native OAuth.
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

unset DEEPSEEK_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY \
      ANTHROPIC_BASE_URL ANTHROPIC_MODEL

if [ -n "${TOGETHER_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${TOGETHER_API_KEY}"
  export OPENAI_BASE_URL="https://api.together.xyz/v1"
fi
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-15}"
TASKS="${TASKS:-locomo,longmemeval}"
ARMS="${ARMS:-default,organized}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-7}"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
EVAL_MODEL="${EVAL_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}"
EVAL_BASE_URL="${EVAL_BASE_URL:-http://127.0.0.1:8002/v1}"
DOCKER_USER_SPEC="${DOCKER_USER_SPEC:-$(id -u):$(id -g)}"

mkdir -p logs runs
status_file="logs/launch_claude_opus47_default_vs_organized_${TS}.status"
: > "$status_file"

contains() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }

prepare_proposer_home() {
  # Claude native OAuth needs a writable HOME in the container. Mounting the
  # host ~/.claude read-only fails because Claude writes session/plugin state;
  # mounting it read-write pollutes the host login. Use an isolated per-run
  # writable copy of the minimum OAuth files instead.
  local run_id="$1"
  local stage="/tmp/optimizer1_native_proposer_${run_id}"
  rm -rf "$stage"
  mkdir -p "$stage/.claude"
  cp /data/home/yuhan/.claude.json "$stage/.claude.json"
  cp /data/home/yuhan/.claude/.credentials.json "$stage/.claude/.credentials.json"
  chmod 600 "$stage/.claude.json" "$stage/.claude/.credentials.json"
  printf '%s' "$stage"
}

start_one() {
  local task="$1"
  local arm="$2"
  local task_label task_args=() arm_args=() run_id log_path stage_home

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

  if [ "$arm" = "default" ]; then
    arm_args=(--selection-policy default)
    run_id="${task_label}_claude_opus47_default_iter${ITERATIONS}_${TS}"
  elif [ "$arm" = "organized" ]; then
    arm_args=(--selection-policy default --organized)
    run_id="${task_label}_claude_opus47_organized_iter${ITERATIONS}_${TS}"
  else
    printf '[%s] SKIP unknown_arm=%s\n' "$(date -Is)" "$arm" >> "$status_file"
    return 0
  fi

  if [ -d "runs/${run_id}" ]; then
    printf '[%s] SKIP %s existing_run_dir\n' "$(date -Is)" "$run_id" >> "$status_file"
    return 0
  fi

  stage_home="$(prepare_proposer_home "$run_id")"
  log_path="logs/${run_id}.log"
  printf '[%s] START %s\n' "$(date -Is)" "$run_id" >> "$status_file"
  printf '[%s] LOG   %s\n' "$(date -Is)" "$log_path" >> "$status_file"

  setsid nohup python -m optimizer1.cli optimize \
    "${task_args[@]}" \
    "${arm_args[@]}" \
    --run-id "$run_id" \
    --iterations "$ITERATIONS" \
    --split train \
    --scaffolds memgpt_source \
    --scaffold-extra-json @configs/source_memory.example.json \
    --eval-workers "$EVAL_WORKERS" \
    --model "$EVAL_MODEL" \
    --base-url "$EVAL_BASE_URL" \
    --proposer-agent claude \
    --claude-native-auth \
    --claude-model "$CLAUDE_MODEL" \
    --proposer-sandbox docker \
    --proposer-docker-image docker-claude:latest \
    --proposer-docker-user "$DOCKER_USER_SPEC" \
    --proposer-docker-home /home/yuhan \
    --proposer-docker-mount "${stage_home}:/home/yuhan:rw" \
    > "$log_path" 2>&1 < /dev/null &

  local pid=$!
  printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
  printf '%s %s %s\n' "$pid" "$run_id" "$log_path"
}

for task in locomo longmemeval; do
  contains "$TASKS" "$task" || continue
  for arm in default organized; do
    contains "$ARMS" "$arm" || continue
    start_one "$task" "$arm"
  done
done

printf '\nstatus: %s\n' "$status_file"
