#!/usr/bin/env bash
# Launch baseline-vs-optimal1 runs across LoCoMo / LongMemEval / SWE-bench mini
# with the Claude OAuth proposer:
#   * baseline arm: --proposer-no-trace-harness-section (no MCP tools, raw trace only),
#                   no --diagnose, default selection policy.
#   * optimal1 arm: full trace tools (MCP server registered), --diagnose,
#                    --selection-policy pareto, historian subagent triggered
#                    on stagnation streak.
#
# Knobs via env vars:
#   ARMS=baseline,optimal1                       which arms to launch (default: both)
#   TASKS=locomo,longmemeval,swebench            which tasks to launch (default: locomo+longmemeval)
#   TS=<timestamp>                               optional; defaults to now.
#   ITERATIONS=20                                LoCoMo + LongMemEval iters
#   SWEBENCH_ITERATIONS=20                       SWE-bench mini iters (independent of ITERATIONS so the same
#                                                invocation can run 50 iters on memory benchmarks and 30 on swe)
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

# Native Claude OAuth: strip every Anthropic override the project .env or the
# parent shell may have leaked, so claude-runner's native_auth path actually
# falls back to ~/.claude/.credentials.json instead of DEEPSEEK_API_KEY.
unset DEEPSEEK_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY \
      ANTHROPIC_BASE_URL ANTHROPIC_MODEL

# trace_similar MCP tool lazily embeds historical diffs and the
# proposer's query at call time. Point OPENAI_API_KEY / OPENAI_BASE_URL
# at Together AI so the MCP server (forked inside the docker proposer
# container) hits an OpenAI-compatible endpoint. DIFF_EMBEDDING_MODEL
# overrides the default model the MCP server picks.
if [ -n "${TOGETHER_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${TOGETHER_API_KEY}"
  export OPENAI_BASE_URL="https://api.together.xyz/v1"
fi
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"

DOCKER_USER_SPEC="${DOCKER_USER_SPEC:-$(id -u):$(id -g)}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ARMS="${ARMS:-baseline,optimal1}"
TASKS="${TASKS:-locomo,longmemeval}"
ITERATIONS="${ITERATIONS:-20}"
SWEBENCH_ITERATIONS="${SWEBENCH_ITERATIONS:-${ITERATIONS}}"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
EVAL_MODEL="${EVAL_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}"
EVAL_BASE_URL="${EVAL_BASE_URL:-http://127.0.0.1:8002/v1}"

# SWE-bench mini uses an independent solver stack (DeepSeek v4 Flash via the
# anthropic-compat endpoint) and the mini-swe-agent runner. EVAL_MODEL /
# EVAL_BASE_URL above are the Qwen3-8B vLLM judge for LoCoMo + LongMemEval
# and are NOT used by the SWE-bench arm — that one passes
# --mini-swe-agent-command / --mini-swe-agent-eval-command instead.
SWEBENCH_DATA_PATH="${SWEBENCH_DATA_PATH:-data/swebench_verified_full.json}"
SWEBENCH_LIMIT="${SWEBENCH_LIMIT:-30}"
SWEBENCH_EVAL_TIMEOUT_S="${SWEBENCH_EVAL_TIMEOUT_S:-900}"
SWEBENCH_EVAL_WORKERS="${SWEBENCH_EVAL_WORKERS:-10}"
SWEBENCH_SOLVER_MODEL="${SWEBENCH_SOLVER_MODEL:-openai/deepseek-v4-flash}"
SWEBENCH_SOLVER_BASE_URL="${SWEBENCH_SOLVER_BASE_URL:-https://api.deepseek.com/v1}"
SWEBENCH_SOLVER_API_KEY_ENV="${SWEBENCH_SOLVER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
SWEBENCH_SOLVER_MAX_TOKENS="${SWEBENCH_SOLVER_MAX_TOKENS:-4096}"
SWEBENCH_MINISWE_RUNNER="${SWEBENCH_MINISWE_RUNNER:-/data/home/yuhan/MemoMemo/scripts/run_miniswe_swebench_single.py}"
SWEBENCH_MINISWE_RUN_CMD="python ${SWEBENCH_MINISWE_RUNNER} run --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir} --model ${SWEBENCH_SOLVER_MODEL} --base-url ${SWEBENCH_SOLVER_BASE_URL} --max-tokens ${SWEBENCH_SOLVER_MAX_TOKENS} --api-key-env ${SWEBENCH_SOLVER_API_KEY_ENV}"
SWEBENCH_MINISWE_EVAL_CMD="python ${SWEBENCH_MINISWE_RUNNER} eval --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir}"

mkdir -p logs runs
status_file="logs/launch_claude_native_${TS}.status"
: > "$status_file"

prepare_proposer_home() {
  # Per-run isolated, writable copy of the host's Claude OAuth credentials.
  # Mounting the host's ~/.claude directly :ro breaks claude in container
  # (EROFS on plugins/installed_plugins.json rename, ENOENT on
  # projects/-workspace/ mkdir, etc.). Mounting :rw would let the
  # container's session state pollute the host login. The clean fix is to
  # copy just .credentials.json + .claude.json into a fresh dir per run
  # and bind that dir to /home/yuhan:rw. The container then has its own
  # writable HOME, while the host login is untouched. Refresh-token
  # updates inside the container persist for subsequent iterations
  # because every iter sees the same STAGE_HOME.
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
  local task="$1"      # locomo | longmemeval | swebench
  local arm="$2"       # baseline | optimal1
  local run_id args task_label task_iters task_eval_args
  task_iters="$ITERATIONS"
  task_eval_args=(
    --eval-workers "$EVAL_WORKERS"
    --model "$EVAL_MODEL"
    --base-url "$EVAL_BASE_URL"
  )
  if [ "$task" = "locomo" ]; then
    task_label="locomo"
    args=(--locomo)
  elif [ "$task" = "longmemeval" ]; then
    task_label="longmemeval_s"
    args=(--longmemeval --longmemeval-variant s)
  elif [ "$task" = "swebench" ]; then
    task_label="swebench_miniswe_deepseek_v4_flash"
    task_iters="$SWEBENCH_ITERATIONS"
    args=(
      --swebench
      --swebench-data-path "$SWEBENCH_DATA_PATH"
      --limit "$SWEBENCH_LIMIT"
      --eval-timeout-s "$SWEBENCH_EVAL_TIMEOUT_S"
      --mini-swe-agent-command "$SWEBENCH_MINISWE_RUN_CMD"
      --mini-swe-agent-eval-command "$SWEBENCH_MINISWE_EVAL_CMD"
    )
    # SWE-bench mini bypasses the LoCoMo/LongMemEval Qwen judge stack;
    # eval workers are capped low because each task spawns a separate
    # docker-style mini-swe-agent run via the DeepSeek API.
    task_eval_args=(--eval-workers "$SWEBENCH_EVAL_WORKERS")
  else
    printf '[%s] SKIP unknown_task=%s\n' "$(date -Is)" "$task" >> "$status_file"
    return 0
  fi
  run_id="${task_label}_claude_native_${arm}_${TS}"
  if [ -d "runs/${run_id}" ]; then
    printf '[%s] SKIP %s existing_run_dir\n' "$(date -Is)" "$run_id" >> "$status_file"
    return 0
  fi

  # Both arms now share the docker sandbox + the .claude/agents/proposer.md
  # subagent system-prompt channel + Claude OAuth via per-run STAGE_HOME.
  # The remaining variables vs baseline are the three "Optimal1 additions":
  #
  #     1. selection_policy:   default → pareto   (patch base resampled
  #                                                from the Pareto frontier;
  #                                                historian subagent
  #                                                triggers on stagnation)
  #     2. diagnoser subagent: off     → on       (--diagnose)
  #     3. MCP trace tools:    off     → on       (no --proposer-no-trace-
  #                                                harness-section). The MCP
  #                                                server's trace_similar
  #                                                tool lazily embeds diffs
  #                                                using the env-supplied
  #                                                DIFF_EMBEDDING_MODEL +
  #                                                OPENAI_* endpoint.
  #
  # All three flip together for optimal1. Anything outside this block —
  # docker image, user spec, docker mount of STAGE_HOME, claude version,
  # eval stack, iter count, split, scaffold seed — must stay identical.
  local stage_home
  stage_home="$(prepare_proposer_home "$run_id")"

  local common_arm_args=(
    --proposer-sandbox docker
    --proposer-docker-image docker-claude:latest
    --proposer-docker-user "$DOCKER_USER_SPEC"
    --proposer-docker-home /home/yuhan
    --proposer-docker-mount "${stage_home}:/home/yuhan:rw"
  )

  local arm_args
  if [ "$arm" = "baseline" ]; then
    # default policy + raw trace on disk only (no MCP tools, no
    # --diagnose, no embedding). The proposer still gets the
    # subagent system prompt (.claude/agents/proposer.md) and the
    # workspace's CLAUDE.md, identical to optimal1 — that channel
    # is no longer a variable.
    arm_args=(
      "${common_arm_args[@]}"
      --selection-policy default
      --proposer-no-trace-harness-section
    )
  else
    # Optimal1: pareto policy + diagnoser + MCP trace tools (which
    # bring trace_similar with lazy embedding via the exported
    # OPENAI_* + DIFF_EMBEDDING_MODEL env vars).
    arm_args=(
      "${common_arm_args[@]}"
      --selection-policy pareto
      --diagnose
    )
  fi

  local lme_extra=()
  if [ "$task" = "longmemeval" ]; then
    if [ -n "${TOGETHER_API_KEY:-}" ]; then
      lme_extra+=(--longmemeval-judge-api-key "$TOGETHER_API_KEY")
    else
      lme_extra+=(--longmemeval-no-llm-judge)
    fi
  fi

  local log_path="logs/${run_id}.log"
  printf '[%s] START %s\n' "$(date -Is)" "$run_id" >> "$status_file"
  printf '[%s] LOG   %s\n' "$(date -Is)" "$log_path" >> "$status_file"

  setsid nohup python -m optimizer1.cli optimize \
    "${args[@]}" \
    --run-id "$run_id" \
    --iterations "$task_iters" \
    --split train \
    "${task_eval_args[@]}" \
    --proposer-agent claude \
    --claude-native-auth \
    "${arm_args[@]}" \
    "${lme_extra[@]}" \
    > "$log_path" 2>&1 < /dev/null &

  local pid=$!
  printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
  printf '%s %s %s\n' "$pid" "$run_id" "$log_path"
}

IFS=','
for arm in $ARMS; do
  for task in $TASKS; do
    start_one "$task" "$arm"
  done
done
unset IFS

printf '\nstatus: %s\n' "$status_file"
