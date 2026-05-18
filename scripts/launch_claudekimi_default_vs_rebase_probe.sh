#!/usr/bin/env bash
# Probe: default vs default+rebase on the claudekimi proposer, LoCoMo +
# LongMemEval, short horizon (15 iters by default) -- a clean rebase-on/off
# ablation to see whether rotating the patch base changes the *trend* of the
# search (does the patch start compounding from iter ~2, vs the fixed-base
# early-churn pattern), and what that does to the early breakthrough curve.
#
# Four runs, all detached + in parallel:
#   * baseline arm: --selection-policy default                 (fixed base x0)
#   * rebase arm:   --selection-policy pareto --no-historian   (base resampled
#                   from the current Pareto frontier; everything else -- the
#                   fixed-high full-history context, the prompt, the trace
#                   harness -- identical to the baseline arm, and the
#                   stagnation-forensics historian is OFF so the *only*
#                   difference vs baseline is the patch base)
#   neither arm uses --diagnose.
#
# This is the same proposer/scaffold/judge/docker setup as
# launch_claudekimi_memory_optimal1_ablation.sh; only the policy arms and the
# iteration count differ.
#
# Knobs via env vars:
#   ITERATIONS=15            propose rounds per run
#   TASKS=locomo,longmemeval which benchmarks (default: both)
#   ARMS=baseline,rebase     which arms (default: both)
#   TS=<timestamp>           run-id suffix (default: now)
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

# Kimi "coding" subscription keys are sk-kimi-*; the anthropic-compat coding
# endpoint is api.kimi.com/coding. A bare sk-* Moonshot key uses
# api.moonshot.ai/anthropic instead.
if [[ "$KIMI_API_KEY" == sk-kimi-* ]]; then
  KIMI_BASE_URL="https://api.kimi.com/coding"
else
  KIMI_BASE_URL="https://api.moonshot.ai/anthropic"
fi
KIMI_MODEL="${KIMI_MODEL:-kimi-k2.6}"

# trace_similar MCP tool embeds historical diffs + the proposer's query at
# call time; point OpenAI-compat env at Together AI so the forked MCP server
# (inside the docker proposer container) has an endpoint to hit.
export OPENAI_API_KEY="${TOGETHER_API_KEY}"
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"
# claude 2.1.85 inside docker-claude-kimi doesn't play well with tool-search
# against a third-party endpoint; disable it (the image's own wrapper does too).
export ENABLE_TOOL_SEARCH=false
# Keep proposer/diagnoser subagents on the Kimi model, not a sonnet/haiku alias.
export ANTHROPIC_DEFAULT_OPUS_MODEL="${KIMI_MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${KIMI_MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${KIMI_MODEL}"
export CLAUDE_CODE_SUBAGENT_MODEL="${KIMI_MODEL}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-15}"
TASKS="${TASKS:-locomo,longmemeval}"
ARMS="${ARMS:-baseline,rebase}"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
EVAL_MODEL="${EVAL_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}"
EVAL_BASE_URL="${EVAL_BASE_URL:-http://127.0.0.1:8002/v1}"
LME_JUDGE_MODEL="${LME_JUDGE_MODEL:-openai/gpt-oss-120b}"
LME_JUDGE_BASE_URL="${LME_JUDGE_BASE_URL:-https://api.together.xyz/v1}"
DOCKER_USER_SPEC="${DOCKER_USER_SPEC:-$(id -u):$(id -g)}"

mkdir -p logs runs
status_file="logs/launch_claudekimi_default_vs_rebase_probe_${TS}.status"
: > "$status_file"

# Shared base args -- identical across both arms (the policy + --no-historian
# are the only per-arm difference).
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
)

contains() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }

start_one() {
  local task="$1"   # locomo | longmemeval
  local arm="$2"    # baseline | rebase
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

  if [ "$arm" = "baseline" ]; then
    arm_args=(--selection-policy default)
    run_id="${task_label}_claudekimi_k26_maxeffort_baseline_iter${ITERATIONS}_${TS}"
  elif [ "$arm" = "rebase" ]; then
    arm_args=(--selection-policy pareto --no-historian)
    run_id="${task_label}_claudekimi_k26_maxeffort_rebase_nohistorian_iter${ITERATIONS}_${TS}"
  else
    printf '[%s] SKIP unknown_arm=%s\n' "$(date -Is)" "$arm" >> "$status_file"
    return 0
  fi

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
  contains "$TASKS" "$task" || continue
  for arm in baseline rebase; do
    contains "$ARMS" "$arm" || continue
    start_one "$task" "$arm"
  done
done

printf '\nstatus: %s\n' "$status_file"
