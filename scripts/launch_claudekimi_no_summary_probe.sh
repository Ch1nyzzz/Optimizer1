#!/usr/bin/env bash
# Probe: `default` with the cumulative cross-session summary withheld
# (--no-summary), claudekimi proposer, LoCoMo + LongMemEval, 15 iters.
#
# Two runs (one per benchmark). The matching WITH-summary baseline is the
# `baseline` arm of launch_claudekimi_default_vs_rebase_probe.sh
# (`*_claudekimi_k26_maxeffort_baseline_iter15_*` -- same setup, just with
# the `summaries/` directory). The only difference vs that baseline here is
# --no-summary: the workspace `summaries/` dir is not created and the prompt's
# summary section is replaced with a note pointing at the raw
# reference_iterations/iter_NNN/ bundles.
#
# Same proposer/scaffold/judge/docker stack as
# launch_claudekimi_memory_optimal1_ablation.sh; only --no-summary and the
# iteration count differ.
#
# Knobs: ITERATIONS=15  TASKS=locomo,longmemeval  TS=<timestamp>
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
  echo "error: KIMI_API_KEY not set (expected in .env)" >&2; exit 1
fi
if [ -z "${TOGETHER_API_KEY:-}" ]; then
  echo "error: TOGETHER_API_KEY not set (expected in .env)" >&2; exit 1
fi

if [[ "$KIMI_API_KEY" == sk-kimi-* ]]; then
  KIMI_BASE_URL="https://api.kimi.com/coding"
else
  KIMI_BASE_URL="https://api.moonshot.ai/anthropic"
fi
KIMI_MODEL="${KIMI_MODEL:-kimi-k2.6}"

export OPENAI_API_KEY="${TOGETHER_API_KEY}"
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"
export ENABLE_TOOL_SEARCH=false
export ANTHROPIC_DEFAULT_OPUS_MODEL="${KIMI_MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${KIMI_MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${KIMI_MODEL}"
export CLAUDE_CODE_SUBAGENT_MODEL="${KIMI_MODEL}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-15}"
TASKS="${TASKS:-locomo,longmemeval}"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
EVAL_MODEL="${EVAL_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}"
EVAL_BASE_URL="${EVAL_BASE_URL:-http://127.0.0.1:8002/v1}"
LME_JUDGE_MODEL="${LME_JUDGE_MODEL:-openai/gpt-oss-120b}"
LME_JUDGE_BASE_URL="${LME_JUDGE_BASE_URL:-https://api.together.xyz/v1}"
DOCKER_USER_SPEC="${DOCKER_USER_SPEC:-$(id -u):$(id -g)}"

mkdir -p logs runs
status_file="logs/launch_claudekimi_no_summary_probe_${TS}.status"
: > "$status_file"

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
  --selection-policy default
  --no-summary
)

contains() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }

start_one() {
  local task="$1" task_label task_args=() run_id log_path
  if [ "$task" = "locomo" ]; then
    task_label="locomo"; task_args=(--locomo)
  elif [ "$task" = "longmemeval" ]; then
    task_label="longmemeval_s"
    task_args=(--longmemeval --longmemeval-variant s
      --longmemeval-judge-model "$LME_JUDGE_MODEL"
      --longmemeval-judge-base-url "$LME_JUDGE_BASE_URL"
      --longmemeval-judge-api-key "$TOGETHER_API_KEY")
  else
    printf '[%s] SKIP unknown_task=%s\n' "$(date -Is)" "$task" >> "$status_file"; return 0
  fi
  run_id="${task_label}_claudekimi_k26_maxeffort_nosummary_iter${ITERATIONS}_${TS}"
  if [ -d "runs/${run_id}" ]; then
    printf '[%s] SKIP %s existing_run_dir\n' "$(date -Is)" "$run_id" >> "$status_file"; return 0
  fi
  log_path="logs/${run_id}.log"
  printf '[%s] START %s\n' "$(date -Is)" "$run_id" >> "$status_file"
  printf '[%s] LOG   %s\n' "$(date -Is)" "$log_path" >> "$status_file"
  setsid nohup "${common_args[@]}" "${task_args[@]}" --run-id "$run_id" \
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
