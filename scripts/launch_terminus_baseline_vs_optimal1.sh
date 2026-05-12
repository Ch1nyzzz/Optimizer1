#!/usr/bin/env bash
# Launch baseline-vs-optimal1 Terminal-Bench 2.0 (Terminus-KIRA) runs.
#
# Everything except the algorithm is aligned with the meta-harness TB2
# reference experiment: dataset terminal-bench@2.0, primary baseline
# Terminus-KIRA (agents.baseline_kira), secondary baseline Terminus-2
# (agents.baseline_terminus2), 2 attempts/task, smoke task extract-elf, the
# SKILL.md proposer prior carried via optimizer1's proposer subagent +
# optimization cells. The three deliberate deviations from meta-harness are:
#   * algorithm  = Optimal1 (pareto base resampling + --diagnose + trace MCP
#                  + historian) instead of the plain propose→eval loop;
#   * solver     = DeepSeek v4 Pro, high ("max") reasoning effort, instead of
#                  Claude Opus 4.6;
#   * rollout sandbox = Harbor's local `docker` environment instead of runloop.
#
# Arms:
#   * baseline arm: --selection-policy default + --proposer-no-trace-harness-section
#                   (no MCP tools, no --diagnose) — the meta-harness baseline.
#   * optimal1 arm: --selection-policy pareto + --diagnose + full trace MCP tools
#                   (+ historian subagent on stagnation).
#
# Knobs (env vars):
#   ARMS=baseline,optimal1               which arms (default: both)
#   ITERATIONS=10                        evolution iterations per arm
#   SPLIT=train                          train = 30 "hard" tasks; test = the other 59
#                                        (test needs TERMINUS_TEST_TASKS or a tasks file)
#   TERMINUS_TEST_TASKS=                 comma-separated TB2 task ids for the test split
#   ROLLOUT_TRIALS=2                     attempts per task
#   ROLLOUT_CONCURRENCY=12               max concurrent harbor docker trials
#   AGENT_TIMEOUT_MULT=2.0               --agent-timeout-multiplier (DeepSeek v4 Pro max effort is slow; 0 = harbor default)
#   TERMINUS_ENV=docker                  harbor environment (set to `runloop` for the cloud path)
#   SOLVER_MODEL=openai/deepseek-v4-pro  litellm model id for the Terminus base model
#   SOLVER_BASE_URL=https://api.deepseek.com/v1
#   SOLVER_API_KEY_ENV=DEEPSEEK_API_KEY
#   REASONING_EFFORT=high                ("max" == high; "" to omit)
#   JOB_TIMEOUT_S=21600                  wall-clock timeout for one harbor run
#   TERMINUS_SOURCE_PATH=references/vendor/meta-harness/reference_examples/terminal_bench_2
#   TS=<timestamp>                       optional
#
# Each run is detached with `setsid` so it survives the parent shell.
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# trace_similar (Optimal1's MCP tool) lazily embeds historical diffs at query
# time against an OpenAI-compatible endpoint. Point OPENAI_* at Together AI.
if [ -n "${TOGETHER_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${TOGETHER_API_KEY}"
  export OPENAI_BASE_URL="https://api.together.xyz/v1"
fi
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"

if [ -z "${KIMI_API_KEY:-}" ] && [ -z "${MOONSHOT_API_KEY:-}" ]; then
  echo "warning: KIMI_API_KEY / MOONSHOT_API_KEY not set; the claudekimi proposer will fail to authenticate" >&2
fi
SOLVER_API_KEY_ENV="${SOLVER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
if [ -z "${!SOLVER_API_KEY_ENV:-}" ]; then
  echo "warning: ${SOLVER_API_KEY_ENV} not set; the Terminus solver (DeepSeek v4 Pro) will fail unless --dry-run" >&2
fi

DOCKER_USER_SPEC="${DOCKER_USER_SPEC:-$(id -u):$(id -g)}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ARMS="${ARMS:-baseline,optimal1}"
ITERATIONS="${ITERATIONS:-10}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-0}"
ROLLOUT_TRIALS="${ROLLOUT_TRIALS:-2}"
ROLLOUT_CONCURRENCY="${ROLLOUT_CONCURRENCY:-12}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-2.0}"
TERMINUS_ENV="${TERMINUS_ENV:-docker}"
SOLVER_MODEL="${SOLVER_MODEL:-openai/deepseek-v4-pro}"
SOLVER_BASE_URL="${SOLVER_BASE_URL:-https://api.deepseek.com/v1}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
JOB_TIMEOUT_S="${JOB_TIMEOUT_S:-21600}"
TERMINUS_SOURCE_PATH="${TERMINUS_SOURCE_PATH:-references/vendor/meta-harness/reference_examples/terminal_bench_2}"
TERMINUS_TEST_TASKS="${TERMINUS_TEST_TASKS:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p logs runs
status_file="logs/launch_terminus_${TS}.status"
: > "$status_file"

start_one() {
  local arm="$1"     # baseline | optimal1
  local run_id log_path
  run_id="terminus_kira_deepseek_v4_pro_claudekimi_${arm}_${SPLIT}_${TS}"
  if [ -d "runs/${run_id}" ]; then
    printf '[%s] SKIP %s existing_run_dir\n' "$(date -Is)" "$run_id" >> "$status_file"
    return 0
  fi

  local arm_args
  if [ "$arm" = "baseline" ]; then
    arm_args=(--selection-policy default --proposer-no-trace-harness-section)
  else
    arm_args=(--selection-policy pareto --diagnose)
  fi

  local test_args=()
  [ -n "$TERMINUS_TEST_TASKS" ] && test_args+=(--terminus-test-tasks "$TERMINUS_TEST_TASKS")
  local dry_args=()
  [ "$DRY_RUN" = "1" ] && dry_args+=(--dry-run)
  local effort_args=()
  [ -n "$REASONING_EFFORT" ] && effort_args+=(--terminus-reasoning-effort "$REASONING_EFFORT")

  log_path="logs/${run_id}.log"
  printf '[%s] START %s\n' "$(date -Is)" "$run_id" >> "$status_file"
  printf '[%s] LOG   %s\n' "$(date -Is)" "$log_path" >> "$status_file"

  setsid nohup python -m optimizer1.cli optimize \
    --terminus \
    --run-id "$run_id" \
    --iterations "$ITERATIONS" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --terminus-source-path "$TERMINUS_SOURCE_PATH" \
    --terminus-env "$TERMINUS_ENV" \
    --terminus-solver-model "$SOLVER_MODEL" \
    --terminus-solver-base-url "$SOLVER_BASE_URL" \
    --terminus-solver-api-key-env "$SOLVER_API_KEY_ENV" \
    "${effort_args[@]}" \
    --terminus-rollout-trials "$ROLLOUT_TRIALS" \
    --terminus-rollout-concurrency "$ROLLOUT_CONCURRENCY" \
    --terminus-agent-timeout-multiplier "$AGENT_TIMEOUT_MULT" \
    --terminus-job-timeout-s "$JOB_TIMEOUT_S" \
    "${test_args[@]}" \
    --proposer-agent kimi \
    --proposer-sandbox docker \
    --proposer-docker-image docker-claude-kimi:latest \
    --proposer-docker-user "$DOCKER_USER_SPEC" \
    --proposer-docker-home /tmp \
    --proposer-docker-env KIMI_API_KEY \
    "${arm_args[@]}" \
    "${dry_args[@]}" \
    > "$log_path" 2>&1 < /dev/null &

  local pid=$!
  printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
  printf '%s %s %s\n' "$pid" "$run_id" "$log_path"
}

IFS=','
for arm in $ARMS; do
  start_one "$arm"
done
unset IFS

printf '\nstatus: %s\n' "$status_file"
