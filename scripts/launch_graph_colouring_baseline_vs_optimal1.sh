#!/usr/bin/env bash
# Launch baseline-vs-optimal1 runs on the graph-colouring DIMACS C++ heuristic
# benchmark with the Claude OAuth proposer.
#
#   * baseline arm: --proposer-no-trace-harness-section (no MCP tools, raw trace
#                   only), no --diagnose, default selection policy.
#   * optimal1 arm: full trace tools (MCP server registered), --diagnose,
#                    --selection-policy pareto, historian subagent triggered on
#                    stagnation streak.
#
# Knobs via env vars:
#   ARMS=baseline,optimal1                     which arms to launch (default: both)
#   AUTH_MODE=deepseek|native                  proposer auth path.
#                                              deepseek (default): claude CLI
#                                              talks to api.deepseek.com via the
#                                              anthropic-compat endpoint, keyed
#                                              by DEEPSEEK_API_KEY.
#                                              native: claude OAuth using the
#                                              host's ~/.claude credentials.
#   CLAUDE_MODEL=                              override --claude-model. Empty =
#                                              the optimizer1 default
#                                              (deepseek-v4-pro[1m] for deepseek
#                                              mode, claude CLI default for
#                                              native mode).
#   TS=<timestamp>                             optional; defaults to now.
#   ITERATIONS=20                              proposer iterations per arm.
#   GRAPH_COLOURING_TARGET_ALGORITHM=evolved   the algorithm slot the proposer
#                                              mutates and the eval invokes.
#   GRAPH_COLOURING_TIME_BUDGET_S=30           per-instance runner timeout.
#   GRAPH_COLOURING_COMPILE_TIMEOUT_S=120      `make all` timeout per candidate.
#   GRAPH_COLOURING_TRAIN_INSTANCES=...        comma-separated DIMACS .col list.
#                                              empty = curated default subset.
#   GRAPH_COLOURING_SOURCE_PATH=               override the upstream checkout
#                                              (default references/vendor/graph-colouring).
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

AUTH_MODE="${AUTH_MODE:-deepseek}"
CLAUDE_MODEL="${CLAUDE_MODEL:-}"

if [ "$AUTH_MODE" = "native" ]; then
  # Native Claude OAuth: strip every Anthropic override the project .env or the
  # parent shell may have leaked, so claude-runner's native_auth path actually
  # falls back to ~/.claude/.credentials.json instead of DEEPSEEK_API_KEY.
  unset DEEPSEEK_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY \
        ANTHROPIC_BASE_URL ANTHROPIC_MODEL
elif [ "$AUTH_MODE" = "deepseek" ]; then
  if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    printf 'fatal: AUTH_MODE=deepseek but DEEPSEEK_API_KEY is not set; populate .env or export it.\n' >&2
    exit 2
  fi
  # claude-runner builds the docker -e flags from os.environ at command-prep
  # time (see claude_runner._prepare_agent_command), so any env var that
  # should reach the proposer container must already be exported in this
  # parent shell. Set the anthropic-compat trio explicitly from
  # DEEPSEEK_API_KEY and the deepseek anthropic endpoint defaults; without
  # this the in-container claude CLI prints "Not logged in · Please run
  # /login" and the iter dies with returncode 1 before any tool call.
  unset ANTHROPIC_API_KEY
  export ANTHROPIC_AUTH_TOKEN="${DEEPSEEK_API_KEY}"
  export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.deepseek.com/anthropic}"
  export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-deepseek-v4-pro[1m]}"
else
  printf 'fatal: AUTH_MODE must be deepseek or native (got %s).\n' "$AUTH_MODE" >&2
  exit 2
fi

# trace_similar MCP tool lazily embeds historical diffs and the proposer's
# query at call time. Point OPENAI_* at Together AI so the MCP server (forked
# inside the docker proposer container) hits an OpenAI-compatible endpoint.
if [ -n "${TOGETHER_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${TOGETHER_API_KEY}"
  export OPENAI_BASE_URL="https://api.together.xyz/v1"
fi
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"

DOCKER_USER_SPEC="${DOCKER_USER_SPEC:-$(id -u):$(id -g)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-docker-claude-cpp:latest}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ARMS="${ARMS:-baseline,optimal1}"
ITERATIONS="${ITERATIONS:-20}"

GRAPH_COLOURING_TARGET_ALGORITHM="${GRAPH_COLOURING_TARGET_ALGORITHM:-evolved}"
GRAPH_COLOURING_TIME_BUDGET_S="${GRAPH_COLOURING_TIME_BUDGET_S:-30}"
GRAPH_COLOURING_COMPILE_TIMEOUT_S="${GRAPH_COLOURING_COMPILE_TIMEOUT_S:-120}"
GRAPH_COLOURING_TRAIN_INSTANCES="${GRAPH_COLOURING_TRAIN_INSTANCES:-}"
GRAPH_COLOURING_SOURCE_PATH="${GRAPH_COLOURING_SOURCE_PATH:-references/vendor/graph-colouring}"

if [ ! -d "$GRAPH_COLOURING_SOURCE_PATH" ]; then
  printf 'fatal: graph-colouring source path %s does not exist; run scripts/fetch_reference_repos.sh first.\n' \
    "$GRAPH_COLOURING_SOURCE_PATH" >&2
  exit 2
fi

mkdir -p logs runs
status_file="logs/launch_graph_colouring_${TS}.status"
: > "$status_file"

prepare_proposer_home() {
  # Per-run isolated, writable copy of the host's Claude OAuth credentials.
  # Only used in AUTH_MODE=native; see
  # launch_claude_native_baseline_vs_optimal1.sh for the rationale (host
  # ~/.claude :ro breaks claude in container; :rw pollutes the host login).
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
  local arm="$1"   # baseline | optimal1
  local run_id common_arm_args=()
  run_id="graph_colouring_${AUTH_MODE}_${arm}_${TS}"
  if [ -d "runs/${run_id}" ]; then
    printf '[%s] SKIP %s existing_run_dir\n' "$(date -Is)" "$run_id" >> "$status_file"
    return 0
  fi

  if [ "$AUTH_MODE" = "native" ]; then
    local stage_home
    stage_home="$(prepare_proposer_home "$run_id")"
    common_arm_args=(
      --proposer-sandbox docker
      --proposer-docker-image "$DOCKER_IMAGE"
      --proposer-docker-user "$DOCKER_USER_SPEC"
      --proposer-docker-home /home/yuhan
      --proposer-docker-mount "${stage_home}:/home/yuhan:rw"
    )
  else
    # DeepSeek (anthropic-compat endpoint): no .claude credentials needed,
    # the key flows in via DEEPSEEK_API_KEY → ANTHROPIC_AUTH_TOKEN inside
    # the container. HOME=/tmp keeps claude CLI from trying to write
    # session state into a non-existent home dir. The extra
    # --proposer-docker-env entries are NOT in DEFAULT_DOCKER_ENV_VARS;
    # claude-runner only forwards env vars that are both (a) on the
    # docker_env_vars list and (b) present in os.environ when the docker
    # command is built. ANTHROPIC_BASE_URL/MODEL plus the OPENAI_*/
    # DIFF_EMBEDDING_MODEL trio are needed for trace_similar's lazy
    # embedding via Together AI in optimal1 runs.
    common_arm_args=(
      --proposer-sandbox docker
      --proposer-docker-image "$DOCKER_IMAGE"
      --proposer-docker-user "$DOCKER_USER_SPEC"
      --proposer-docker-home /tmp
      --proposer-docker-env DEEPSEEK_API_KEY
      --proposer-docker-env ANTHROPIC_AUTH_TOKEN
      --proposer-docker-env ANTHROPIC_BASE_URL
      --proposer-docker-env ANTHROPIC_MODEL
      --proposer-docker-env OPENAI_API_KEY
      --proposer-docker-env OPENAI_BASE_URL
      --proposer-docker-env DIFF_EMBEDDING_MODEL
    )
  fi

  local arm_args
  if [ "$arm" = "baseline" ]; then
    # default policy + raw trace on disk only. Proposer still gets the
    # subagent system prompt and the workspace CLAUDE.md, identical to
    # optimal1; that channel is not a variable.
    arm_args=(
      "${common_arm_args[@]}"
      --selection-policy default
      --proposer-no-trace-harness-section
    )
  else
    # Optimal1: pareto base resampling, diagnoser subagent, MCP trace
    # tools (trace_similar with lazy embedding via the OPENAI_* env).
    arm_args=(
      "${common_arm_args[@]}"
      --selection-policy pareto
      --diagnose
    )
  fi

  local task_args=(
    --graph-colouring
    --graph-colouring-source-path "$GRAPH_COLOURING_SOURCE_PATH"
    --graph-colouring-target-algorithm "$GRAPH_COLOURING_TARGET_ALGORITHM"
    --graph-colouring-time-budget-s "$GRAPH_COLOURING_TIME_BUDGET_S"
    --graph-colouring-compile-timeout-s "$GRAPH_COLOURING_COMPILE_TIMEOUT_S"
  )
  if [ -n "$GRAPH_COLOURING_TRAIN_INSTANCES" ]; then
    task_args+=(--graph-colouring-train-instances "$GRAPH_COLOURING_TRAIN_INSTANCES")
  fi

  local auth_args=()
  if [ "$AUTH_MODE" = "native" ]; then
    auth_args=(--claude-native-auth)
  fi
  if [ -n "$CLAUDE_MODEL" ]; then
    auth_args+=(--claude-model "$CLAUDE_MODEL")
  fi

  local log_path="logs/${run_id}.log"
  printf '[%s] START %s\n' "$(date -Is)" "$run_id" >> "$status_file"
  printf '[%s] LOG   %s\n' "$(date -Is)" "$log_path" >> "$status_file"

  setsid nohup python -m optimizer1.cli optimize \
    "${task_args[@]}" \
    --run-id "$run_id" \
    --iterations "$ITERATIONS" \
    --split train \
    --proposer-agent claude \
    "${auth_args[@]}" \
    "${arm_args[@]}" \
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
