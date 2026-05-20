#!/usr/bin/env bash
# Codex (gpt-5.5, reasoning_effort=high) proposer × SWE-bench mini-swe-agent
# (MiniMax-M2.7 via Together AI), default-vs-organized on volatile30,
# 30 iters each, both arms parallel and sharing a primed iter-0 baseline.
#
# Mirrors the configuration of the in-flight runs:
#   swebench_miniswe_minimax_m27_claudekimi_pure_default_volatile30_*
#   swebench_miniswe_minimax_m27_claudekimi_organized_volatile30_*
# The only intentional difference is the proposer: claude(kimi) → codex(gpt-5.5).
# No docker sandbox is used for the codex proposer — the codex CLI runs on
# the host against the per-iteration workspace just like the locomo/lme
# codex runs (auth from ~/.codex/auth.json).
#
# Auth: `codex login status` must report ChatGPT login.
#       TOGETHER_API_KEY must be in .env (for both the eval scaffold and the
#       trace_similar MCP embeddings).
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

# Together OpenAI-compat endpoint for the trace_similar MCP embeddings and
# (indirectly via api-key-env=TOGETHER_API_KEY) the mini-swe-agent itself.
# _codex_env in codex_runner strips OPENAI_API_KEY before forwarding to the
# codex CLI, so this does not affect the ChatGPT login.
export OPENAI_API_KEY="${TOGETHER_API_KEY}"
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-30}"
ARMS="${ARMS:-default,organized}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.5}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-high}"
SWE_LIMIT="${SWE_LIMIT:-30}"
SWE_DATA_PATH="${SWE_DATA_PATH:-data/swebench_train_volatile30.json}"
EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-900}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"
EVAL_MODEL="${EVAL_MODEL:-openai/MiniMaxAI/MiniMax-M2.7}"
EVAL_BASE_URL="${EVAL_BASE_URL:-https://api.together.xyz/v1}"
MINISWE_MAX_TOKENS="${MINISWE_MAX_TOKENS:-4096}"
BASELINE_SWE_DIR="${BASELINE_SWE_DIR:-runs/baseline_swebench_volatile30_miniswe_minimax_m27_20260518}"

mkdir -p logs runs
status_file="logs/launch_codex_gpt55_swebench_default_vs_organized_${TS}.status"
: > "$status_file"
printf '[%s] LAUNCHER start ts=%s iter=%s arms=%s eval_model=%s\n' \
  "$(date -Is)" "$TS" "$ITERATIONS" "$ARMS" "$EVAL_MODEL" >> "$status_file"
printf '[%s] BASELINE_SWE_DIR=%s\n' "$(date -Is)" "$BASELINE_SWE_DIR" >> "$status_file"

contains() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }

# Shared scaffold-eval CLI fragment. Repeated verbatim across the prime call
# and the two arm calls so the baseline measurement and the optimization
# loop see the same scaffold contract.
miniswe_args=(
  --mini-swe-agent-command "python scripts/run_miniswe_swebench_single.py run --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir} --model $EVAL_MODEL --base-url $EVAL_BASE_URL --max-tokens $MINISWE_MAX_TOKENS --api-key-env TOGETHER_API_KEY"
  --mini-swe-agent-eval-command "python scripts/run_miniswe_swebench_single.py eval --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir}"
)

prime_swebench_baseline() {
  if [ -f "${BASELINE_SWE_DIR}/run_summary.json" ]; then
    printf '[%s] BASELINE_REUSE swebench=%s\n' "$(date -Is)" "$BASELINE_SWE_DIR" >> "$status_file"
    return 0
  fi
  printf '[%s] BASELINE_PRIME swebench -> %s\n' "$(date -Is)" "$BASELINE_SWE_DIR" >> "$status_file"
  local prime_log="logs/baseline_swebench_${TS}.log"
  python -m optimizer1.cli optimize \
    --swebench \
    --run-id "$(basename "$BASELINE_SWE_DIR")" \
    --out "$BASELINE_SWE_DIR" \
    --iterations 0 \
    --split train \
    --limit "$SWE_LIMIT" \
    --swebench-data-path "$SWE_DATA_PATH" \
    --eval-timeout-s "$EVAL_TIMEOUT_S" \
    --eval-workers "$EVAL_WORKERS" \
    --selection-policy default \
    "${miniswe_args[@]}" \
    --proposer-agent codex \
    --codex-model "$CODEX_MODEL" \
    --codex-reasoning-effort "$CODEX_REASONING_EFFORT" \
    --no-test-frontier \
    > "$prime_log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ] || [ ! -f "${BASELINE_SWE_DIR}/run_summary.json" ]; then
    printf '[%s] BASELINE_PRIME_FAIL swebench rc=%s log=%s\n' "$(date -Is)" "$rc" "$prime_log" >> "$status_file"
    return 1
  fi
  printf '[%s] BASELINE_PRIME_DONE swebench\n' "$(date -Is)" >> "$status_file"
}

prime_swebench_baseline || exit 1

start_one() {
  local arm="$1"
  local arm_args=() run_id log_path
  if [ "$arm" = "default" ]; then
    arm_args=(--selection-policy default)
    run_id="swebench_miniswe_minimax_m27_codex_gpt55_pure_default_volatile30_${TS}"
  elif [ "$arm" = "organized" ]; then
    arm_args=(--selection-policy default --organized)
    run_id="swebench_miniswe_minimax_m27_codex_gpt55_organized_volatile30_${TS}"
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
    "$(date -Is)" "$run_id" "$BASELINE_SWE_DIR" "$(date -Is)" "$log_path" >> "$status_file"

  setsid nohup python -m optimizer1.cli optimize \
    --swebench \
    --run-id "$run_id" \
    --iterations "$ITERATIONS" \
    --split train \
    --limit "$SWE_LIMIT" \
    --swebench-data-path "$SWE_DATA_PATH" \
    --baseline-dir "$BASELINE_SWE_DIR" \
    --eval-timeout-s "$EVAL_TIMEOUT_S" \
    --eval-workers "$EVAL_WORKERS" \
    --test-frontier-candidate-limit 1 \
    "${arm_args[@]}" \
    "${miniswe_args[@]}" \
    --proposer-agent codex \
    --codex-model "$CODEX_MODEL" \
    --codex-reasoning-effort "$CODEX_REASONING_EFFORT" \
    > "$log_path" 2>&1 < /dev/null &

  local pid=$!
  printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
}

for arm in default organized; do
  contains "$ARMS" "$arm" || continue
  start_one "$arm"
done

printf '[%s] LAUNCHER done — see %s\n' "$(date -Is)" "$status_file" >> "$status_file"
printf '\nstatus: %s\n' "$status_file"
