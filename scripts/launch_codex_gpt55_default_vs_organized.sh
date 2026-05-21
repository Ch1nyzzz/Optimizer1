#!/usr/bin/env bash
# Codex (gpt-5.5, reasoning_effort=high) proposer, default-vs-organized arms,
# across LoCoMo + LongMemEval-S. Base eval model is Together AI
# `openai/gpt-oss-120b` (NOT a local vLLM). 30 iters per run, all 4 launched
# in parallel.
#
# Fairness: default vs organized arms of the SAME benchmark share one iter-0
# baseline (via --baseline-dir). If a baseline dir for either benchmark is
# missing, the script primes it first by running the optimizer with
# --iterations 0 against that benchmark. Cross-benchmark baselines are
# distinct because the underlying eval sets differ.
#
# Auth: `codex login status` must report ChatGPT login (~/.codex/auth.json).
#       TOGETHER_API_KEY must be in .env.
#
# Outputs: runs/<run_id>/, logs/<run_id>.log, logs/launch_<TS>.status
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${TOGETHER_API_KEY:-}" ]; then
  echo "error: TOGETHER_API_KEY not set in .env" >&2
  exit 1
fi

# Together OpenAI-compat endpoint for the eval scaffold's LLM calls and the
# trace_similar MCP embeddings. _codex_env in codex_runner strips
# OPENAI_API_KEY from the env it forwards to the codex CLI, so the proposer's
# ChatGPT login is unaffected; these vars only reach the eval / MCP processes.
export OPENAI_API_KEY="${TOGETHER_API_KEY}"
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-30}"
TASKS="${TASKS:-locomo,longmemeval}"
ARMS="${ARMS:-default,organized}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.5}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-high}"
EVAL_MODEL="${EVAL_MODEL:-openai/gpt-oss-120b}"
EVAL_BASE_URL="${EVAL_BASE_URL:-https://api.together.xyz/v1}"
EVAL_WORKERS="${EVAL_WORKERS:-32}"
EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-300}"
LME_JUDGE_MODEL="${LME_JUDGE_MODEL:-openai/gpt-oss-120b}"
LME_JUDGE_BASE_URL="${LME_JUDGE_BASE_URL:-https://api.together.xyz/v1}"
# Stable baseline-dir name across timestamps so re-runs of this script keep
# reusing the same iter-0 across the day. Override BASELINE_LOCOMO_DIR /
# BASELINE_LME_DIR to point at any prior iter-0 directory.
BASELINE_LOCOMO_DIR="${BASELINE_LOCOMO_DIR:-runs/baseline_locomo_codex_gpt_oss_120b_20260518}"
BASELINE_LME_DIR="${BASELINE_LME_DIR:-runs/baseline_longmemeval_s_codex_gpt_oss_120b_20260518}"

mkdir -p logs runs
status_file="logs/launch_codex_gpt55_default_vs_organized_${TS}.status"
: > "$status_file"
printf '[%s] LAUNCHER start ts=%s iter=%s tasks=%s arms=%s eval_model=%s\n' \
  "$(date -Is)" "$TS" "$ITERATIONS" "$TASKS" "$ARMS" "$EVAL_MODEL" \
  >> "$status_file"
printf '[%s] BASELINE_LOCOMO_DIR=%s\n' "$(date -Is)" "$BASELINE_LOCOMO_DIR" >> "$status_file"
printf '[%s] BASELINE_LME_DIR=%s\n'    "$(date -Is)" "$BASELINE_LME_DIR"    >> "$status_file"

contains() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }

# Prime a baseline directory by running the optimizer with --iterations 0.
# Synchronous because the 4 main runs depend on it via --baseline-dir.
prime_baseline() {
  local benchmark="$1"
  local target_dir="$2"
  if [ -f "${target_dir}/run_summary.json" ]; then
    printf '[%s] BASELINE_REUSE %s=%s\n' "$(date -Is)" "$benchmark" "$target_dir" >> "$status_file"
    return 0
  fi
  printf '[%s] BASELINE_PRIME %s -> %s\n' "$(date -Is)" "$benchmark" "$target_dir" >> "$status_file"
  local benchmark_args=()
  if [ "$benchmark" = "locomo" ]; then
    benchmark_args=(--locomo)
  else
    benchmark_args=(
      --longmemeval --longmemeval-variant s
      --longmemeval-judge-model "$LME_JUDGE_MODEL"
      --longmemeval-judge-base-url "$LME_JUDGE_BASE_URL"
      --longmemeval-judge-api-key "$TOGETHER_API_KEY"
    )
  fi
  local prime_log="logs/baseline_${benchmark}_${TS}.log"
  python -m optimizer1.cli optimize \
    "${benchmark_args[@]}" \
    --run-id "$(basename "$target_dir")" \
    --out "$target_dir" \
    --iterations 0 \
    --split train \
    --eval-workers "$EVAL_WORKERS" \
    --eval-timeout-s "$EVAL_TIMEOUT_S" \
    --model "$EVAL_MODEL" \
    --base-url "$EVAL_BASE_URL" \
    --api-key "$TOGETHER_API_KEY" \
    --scaffolds memgpt_source \
    --proposer-agent codex \
    --codex-model "$CODEX_MODEL" \
    --codex-reasoning-effort "$CODEX_REASONING_EFFORT" \
    --no-test-frontier \
    > "$prime_log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ] || [ ! -f "${target_dir}/run_summary.json" ]; then
    printf '[%s] BASELINE_PRIME_FAIL %s rc=%s log=%s\n' "$(date -Is)" "$benchmark" "$rc" "$prime_log" >> "$status_file"
    return 1
  fi
  printf '[%s] BASELINE_PRIME_DONE %s\n' "$(date -Is)" "$benchmark" >> "$status_file"
}

# Prime baselines in parallel where applicable.
declare -A prime_pids=()
for t in locomo longmemeval; do
  contains "$TASKS" "$t" || continue
  if [ "$t" = "locomo" ]; then
    target="$BASELINE_LOCOMO_DIR"
  else
    target="$BASELINE_LME_DIR"
  fi
  if [ ! -f "${target}/run_summary.json" ]; then
    prime_baseline "$t" "$target" &
    prime_pids[$t]=$!
  fi
done
for t in "${!prime_pids[@]}"; do
  if ! wait "${prime_pids[$t]}"; then
    printf '[%s] LAUNCHER abort because %s baseline failed\n' "$(date -Is)" "$t" >> "$status_file"
    exit 1
  fi
done

start_one() {
  local task="$1"
  local arm="$2"
  local task_label task_args=() arm_args=() baseline_dir run_id log_path

  if [ "$task" = "locomo" ]; then
    task_label="locomo"
    task_args=(--locomo)
    baseline_dir="$BASELINE_LOCOMO_DIR"
  elif [ "$task" = "longmemeval" ]; then
    task_label="longmemeval_s"
    task_args=(
      --longmemeval --longmemeval-variant s
      --longmemeval-judge-model "$LME_JUDGE_MODEL"
      --longmemeval-judge-base-url "$LME_JUDGE_BASE_URL"
      --longmemeval-judge-api-key "$TOGETHER_API_KEY"
    )
    baseline_dir="$BASELINE_LME_DIR"
  else
    printf '[%s] SKIP unknown_task=%s\n' "$(date -Is)" "$task" >> "$status_file"
    return 0
  fi

  if [ "$arm" = "default" ]; then
    arm_args=(--selection-policy default)
    run_id="${task_label}_codex_gpt55_default_iter${ITERATIONS}_${TS}"
  elif [ "$arm" = "organized" ]; then
    arm_args=(--selection-policy default --organized)
    run_id="${task_label}_codex_gpt55_organized_iter${ITERATIONS}_${TS}"
  else
    printf '[%s] SKIP unknown_arm=%s\n' "$(date -Is)" "$arm" >> "$status_file"
    return 0
  fi

  if [ -d "runs/${run_id}" ]; then
    printf '[%s] SKIP %s existing_run_dir\n' "$(date -Is)" "$run_id" >> "$status_file"
    return 0
  fi

  log_path="logs/${run_id}.log"
  printf '[%s] START %s baseline=%s\n[%s] LOG   %s\n' \
    "$(date -Is)" "$run_id" "$baseline_dir" "$(date -Is)" "$log_path" >> "$status_file"

  setsid nohup python -m optimizer1.cli optimize \
    "${task_args[@]}" \
    "${arm_args[@]}" \
    --run-id "$run_id" \
    --iterations "$ITERATIONS" \
    --split train \
    --baseline-dir "$baseline_dir" \
    --eval-workers "$EVAL_WORKERS" \
    --eval-timeout-s "$EVAL_TIMEOUT_S" \
    --model "$EVAL_MODEL" \
    --base-url "$EVAL_BASE_URL" \
    --api-key "$TOGETHER_API_KEY" \
    --scaffolds memgpt_source \
    --proposer-agent codex \
    --codex-model "$CODEX_MODEL" \
    --codex-reasoning-effort "$CODEX_REASONING_EFFORT" \
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

printf '[%s] LAUNCHER done — see %s\n' "$(date -Is)" "$status_file" >> "$status_file"
printf '\nstatus: %s\n' "$status_file"
