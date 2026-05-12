#!/usr/bin/env bash
# Launch the claudekimi proposer on LoCoMo + LongMemEval, optimal1 and
# optimal1_no_diagnose arms (4 runs total, all in parallel).
#
# This mirrors the existing claudecode optimal1 runs
# (locomo_claudecode_deepseek_v4_pro_maxeffort_optimal1_* and the
# longmemeval_s_* equivalents) one-for-one, swapping ONLY the proposer:
#   * proposer code agent: claude CLI inside docker-claude-kimi:latest,
#     routed to api.kimi.com/coding with model kimi-k2.6 at --effort max.
# Everything else is identical to the claudecode optimal1 setup:
#   * scaffold seed:        memgpt_source + configs/source_memory.example.json
#   * iterations:           50, split train
#   * eval judge (LoCoMo + LongMemEval recall): Qwen3-8B vLLM at :8002
#   * LongMemEval LLM judge: Together AI openai/gpt-oss-120b
#   * selection policy:     pareto
#   * trace harness:        on (MCP trace_similar tool; lazy diff embedding
#                           via Together AI + BAAI/bge-large-en-v1.5)
#   * diagnoser subagent:   on for the optimal1 arm, off for optimal1_no_diagnose
#
# Each run is detached with `setsid nohup` so it survives the parent shell.
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${KIMI_API_KEY:-}" ]; then
  echo "error: KIMI_API_KEY not set (expected in .env); the claudekimi proposer cannot authenticate" >&2
  exit 1
fi
if [ -z "${TOGETHER_API_KEY:-}" ]; then
  echo "error: TOGETHER_API_KEY not set (expected in .env); needed for the LongMemEval LLM judge and trace embeddings" >&2
  exit 1
fi

# Kimi "coding" subscription keys are sk-kimi-*; the anthropic-compat
# coding endpoint is api.kimi.com/coding. (A bare sk-* Moonshot key would
# use api.moonshot.ai/anthropic instead.)
if [[ "$KIMI_API_KEY" == sk-kimi-* ]]; then
  KIMI_BASE_URL="https://api.kimi.com/coding"
else
  KIMI_BASE_URL="https://api.moonshot.ai/anthropic"
fi
KIMI_MODEL="${KIMI_MODEL:-kimi-k2.6}"

# trace_similar MCP tool lazily embeds historical diffs + the proposer's
# query at call time. Point OPENAI_API_KEY / OPENAI_BASE_URL at Together AI
# so the forked MCP server (inside the docker proposer container) hits an
# OpenAI-compatible endpoint; DIFF_EMBEDDING_MODEL picks the model. These
# are forwarded into the container because they are in claude_runner's
# DEFAULT_DOCKER_ENV_VARS.
export OPENAI_API_KEY="${TOGETHER_API_KEY}"
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"
# claude 2.1.85 inside docker-claude-kimi doesn't play well with the
# experimental tool-search feature against a third-party model endpoint;
# the image's own claude-kimi wrapper disables it, so do the same here.
export ENABLE_TOOL_SEARCH=false
# Make sure the diagnoser/proposer subagents also run on the Kimi model
# rather than falling back to a sonnet/haiku alias the endpoint can't serve.
export ANTHROPIC_DEFAULT_OPUS_MODEL="${KIMI_MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${KIMI_MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${KIMI_MODEL}"
export CLAUDE_CODE_SUBAGENT_MODEL="${KIMI_MODEL}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
EVAL_MODEL="${EVAL_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}"
EVAL_BASE_URL="${EVAL_BASE_URL:-http://127.0.0.1:8002/v1}"
LME_JUDGE_MODEL="${LME_JUDGE_MODEL:-openai/gpt-oss-120b}"
LME_JUDGE_BASE_URL="${LME_JUDGE_BASE_URL:-https://api.together.xyz/v1}"
DOCKER_USER_SPEC="${DOCKER_USER_SPEC:-$(id -u):$(id -g)}"

mkdir -p logs runs
status_file="logs/launch_claudekimi_memory_optimal1_ablation_${TS}.status"
: > "$status_file"

# Shared base args: identical to the claudecode optimal1 runs except the
# proposer. memgpt_source scaffold + the recall/archival seed config; the
# Qwen3-8B vLLM judge; pareto selection; trace harness left on (no
# --proposer-no-trace-harness-section). --diagnose is appended per-arm.
common_args=(
  python -m optimizer1.cli optimize
  --iterations "$ITERATIONS"
  --split train
  --scaffolds memgpt_source
  --scaffold-extra-json @configs/source_memory.example.json
  --eval-workers "$EVAL_WORKERS"
  --model "$EVAL_MODEL"
  --base-url "$EVAL_BASE_URL"
  --proposer-agent claude
  --claude-base-url "$KIMI_BASE_URL"
  --claude-auth-token "$KIMI_API_KEY"
  --claude-model "$KIMI_MODEL"
  --claude-effort max
  --proposer-sandbox docker
  --proposer-docker-image docker-claude-kimi:latest
  --proposer-docker-user "$DOCKER_USER_SPEC"
  --proposer-docker-home /tmp
  --proposer-docker-env KIMI_API_KEY
  --proposer-docker-env ENABLE_TOOL_SEARCH
  --proposer-docker-env CLAUDE_CODE_SUBAGENT_MODEL
  --proposer-docker-env ANTHROPIC_DEFAULT_OPUS_MODEL
  --proposer-docker-env ANTHROPIC_DEFAULT_SONNET_MODEL
  --proposer-docker-env ANTHROPIC_DEFAULT_HAIKU_MODEL
  --selection-policy pareto
)

start_one() {
  local task="$1"   # locomo | longmemeval
  local arm="$2"    # optimal1 | optimal1_no_diagnose
  local task_label task_args=() arm_args=() run_id log_path

  if [ "$task" = "locomo" ]; then
    task_label="locomo"
    task_args=(--locomo)
  elif [ "$task" = "longmemeval" ]; then
    task_label="longmemeval_s"
    task_args=(
      --longmemeval --longmemeval-variant s
      --longmemeval-judge-model "$LME_JUDGE_MODEL"
      --longmemeval-judge-base-url "$LME_JUDGE_BASE_URL"
      --longmemeval-judge-api-key "$TOGETHER_API_KEY"
    )
  else
    printf '[%s] SKIP unknown_task=%s\n' "$(date -Is)" "$task" >> "$status_file"
    return 0
  fi

  if [ "$arm" = "optimal1" ]; then
    arm_args=(--diagnose)
  else
    arm_args=()
  fi

  run_id="${task_label}_claudekimi_k26_maxeffort_${arm}_${TS}"
  if [ -d "runs/${run_id}" ]; then
    printf '[%s] SKIP %s existing_run_dir\n' "$(date -Is)" "$run_id" >> "$status_file"
    return 0
  fi

  log_path="logs/${run_id}.log"
  printf '[%s] START %s\n' "$(date -Is)" "$run_id" >> "$status_file"
  printf '[%s] LOG   %s\n' "$(date -Is)" "$log_path" >> "$status_file"

  setsid nohup "${common_args[@]}" \
    "${task_args[@]}" \
    "${arm_args[@]}" \
    --run-id "$run_id" \
    > "$log_path" 2>&1 < /dev/null &

  local pid=$!
  printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
  printf '%s %s %s\n' "$pid" "$run_id" "$log_path"
}

for task in locomo longmemeval; do
  for arm in optimal1 optimal1_no_diagnose; do
    start_one "$task" "$arm"
  done
done

printf '\nstatus: %s\n' "$status_file"
