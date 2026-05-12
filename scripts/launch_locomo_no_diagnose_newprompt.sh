#!/usr/bin/env bash
# Re-run the LoCoMo no_diagnose arm with the rewritten anti-overfitting
# proposer.md / workspace.md prompts, DeepSeek v4 Pro proposer at max thinking
# effort. Mirrors the prior run
#   locomo_claudecode_deepseek_v4_pro_maxeffort_optimal1_no_diagnose_20260511_153958
# one-for-one (memgpt_source seed, Qwen3-8B :8002 judge, pareto selection,
# trace harness on, budget forced high, no --diagnose) — the only intended
# change is the prompt text, which is read from src/optimizer1/prompts/ at run
# time. That earlier run overfit (train passrate 0.45 -> held-out test 0.32,
# below the default arm's 0.36; winning candidates were a qwen14b model swap
# and a two-model "verify" pipeline — exactly the patterns the new prompt
# disqualifies).
#
# Set TASK=longmemeval to run the LongMemEval no_diagnose arm instead (adds the
# Together gpt-oss-120b LLM judge).
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "error: DEEPSEEK_API_KEY not set (expected in .env)" >&2; exit 1
fi
if [ -z "${TOGETHER_API_KEY:-}" ]; then
  echo "error: TOGETHER_API_KEY not set (expected in .env; needed for trace_similar embeddings"\
       "and, for longmemeval, the LLM judge)" >&2; exit 1
fi

# trace_similar MCP tool's lazy diff embedding -> Together AI OpenAI-compat endpoint.
export OPENAI_API_KEY="${TOGETHER_API_KEY}"
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
TASK="${TASK:-locomo}"           # locomo | longmemeval
ITERATIONS="${ITERATIONS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
EVAL_MODEL="${EVAL_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}"
EVAL_BASE_URL="${EVAL_BASE_URL:-http://127.0.0.1:8002/v1}"
DOCKER_USER_SPEC="${DOCKER_USER_SPEC:-$(id -u):$(id -g)}"

case "$TASK" in
  locomo)
    task_label="locomo"
    task_args=(--locomo) ;;
  longmemeval)
    task_label="longmemeval_s"
    task_args=(
      --longmemeval --longmemeval-variant s
      --longmemeval-judge-model openai/gpt-oss-120b
      --longmemeval-judge-base-url https://api.together.xyz/v1
      --longmemeval-judge-api-key "$TOGETHER_API_KEY"
    ) ;;
  *) echo "error: unknown TASK=$TASK (expected locomo|longmemeval)" >&2; exit 1 ;;
esac

run_id="${task_label}_claudecode_deepseek_v4_pro_maxeffort_optimal1_no_diagnose_newprompt_${TS}"
mkdir -p logs runs
log_path="logs/${run_id}.log"
status_file="logs/launch_${run_id}.status"
if [ -d "runs/${run_id}" ]; then echo "runs/${run_id} already exists; refusing to clobber" >&2; exit 1; fi
: > "$status_file"
printf '[%s] START %s\n[%s] LOG   %s\n' "$(date -Is)" "$run_id" "$(date -Is)" "$log_path" >> "$status_file"

setsid nohup python -m optimizer1.cli optimize \
  "${task_args[@]}" \
  --run-id "$run_id" \
  --iterations "$ITERATIONS" \
  --split train \
  --eval-workers "$EVAL_WORKERS" \
  --model "$EVAL_MODEL" \
  --base-url "$EVAL_BASE_URL" \
  --scaffolds memgpt_source \
  --proposer-agent claude \
  --claude-model 'deepseek-v4-pro[1m]' \
  --claude-effort max \
  --force-budget high \
  --proposer-sandbox docker \
  --proposer-docker-image docker-claude:latest \
  --proposer-docker-user "$DOCKER_USER_SPEC" \
  --proposer-docker-home /tmp \
  --selection-policy pareto \
  > "$log_path" 2>&1 < /dev/null &

pid=$!
printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
printf '%s %s %s\n' "$pid" "$run_id" "$log_path"
printf 'status: %s\n' "$status_file"
