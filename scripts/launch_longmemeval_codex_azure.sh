#!/usr/bin/env bash
# LongMemEval memory-benchmark comparison experiment —
# Codex proposer (Azure OpenAI) × a remote OpenAI-compatible eval model.
#
# This is the experiment a teammate runs after pulling the repo. See
# docs/LOCOMO_LONGMEMEVAL_CODEX_AZURE.md for the full step-by-step setup.
#
#   * proposer   = Codex CLI authenticated against Azure OpenAI. Auth is an API
#                  key in ~/.codex/config.toml (template: docs/codex_config.azure.toml);
#                  there is no interactive Codex "login" for Azure.
#   * eval model = the model the memgpt_source scaffold queries while answering
#                  LongMemEval questions. It runs on YOUR OWN remote
#                  OpenAI-compatible endpoint — set EVAL_BASE_URL / EVAL_MODEL /
#                  EVAL_API_KEY in .env.
#   * judge      = LongMemEval scores answers with an LLM-as-judge. It runs on
#                  YOUR OWN endpoint too — set JUDGE_BASE_URL / JUDGE_MODEL /
#                  JUDGE_API_KEY. Run with NO_LLM_JUDGE=1 to skip the LLM judge
#                  and use local token/F1 scoring instead.
#
# Two arms — identical except for the RunStore tool surface. Both arms expose
# the same upstream-2 summary files (evolution_summary.jsonl +
# best_candidates.json), so the only variable across arms is the tools:
#   * default   arm: --selection-policy default
#                    (upstream-2 summaries, skill mode "default", no RunStore tools)
#   * organized arm: --organized --selection-policy default
#                    (upstream-2 summaries, skill mode "organized-summaries",
#                     generates state.md and registers RunStore tools)
# Both arms share ONE primed seed frontier (prime_longmemeval_baseline), reused
# via --baseline-dir so the memgpt_source seed eval is not paid for twice.
#
# After the iterations each arm automatically evaluates its best train
# Pareto-frontier candidate on the held-out LongMemEval test split.
#
# The LongMemEval dataset is git-ignored, so a fresh clone has none. The script
# auto-runs `optimizer1.cli longmemeval prepare --allow-download` on first use;
# it downloads the cleaned `s` variant (~277 MB) from Hugging Face and writes the
# deterministic warmup/train/test splits (seed 13).
#
# Secrets are read from .env (git-ignored). Copy .env.example to .env and fill
# it in — see docs/LOCOMO_LONGMEMEVAL_CODEX_AZURE.md. NEVER commit the real .env.
#
# Each arm is detached with `setsid` so it survives the parent shell. The
# dataset prepare + baseline prime run in the foreground (they must finish
# before the arms start), so launch the whole script under `nohup`/`tmux` if
# your SSH session is flaky.
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# ---- knobs (env vars) ---------------------------------------------------
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ARMS="${ARMS:-default,organized}"
ITERATIONS="${ITERATIONS:-30}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-0}"                 # 0 = the whole split (train = 100 LongMemEval questions)
VARIANT="${VARIANT:-s}"             # LongMemEval variant: s | m | oracle
EVAL_WORKERS="${EVAL_WORKERS:-32}"  # concurrent eval calls; raise/lower to your endpoint
EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-300}"
SCAFFOLDS="${SCAFFOLDS:-memgpt_source}"
NO_LLM_JUDGE="${NO_LLM_JUDGE:-0}"   # 1 => local token/F1 scoring instead of the LLM judge
DRY_RUN="${DRY_RUN:-0}"

# Codex proposer (Azure OpenAI). CODEX_MODEL MUST be your Azure *deployment*
# name — it is forwarded as `codex exec -m`, and config.toml's model_provider
# routes it to Azure. Keep it in sync with `model` in ~/.codex/config.toml.
CODEX_MODEL="${CODEX_MODEL:-gpt-5.1-codex}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-high}"
CODEX_HOME="${CODEX_HOME:-}"   # empty => ~/.codex (must hold the Azure config.toml)

# Scaffold eval model — your own remote OpenAI-compatible endpoint. EVAL_API_KEY_ENV
# names the .env variable that holds the key (default EVAL_API_KEY).
EVAL_BASE_URL="${EVAL_BASE_URL:-}"
EVAL_MODEL="${EVAL_MODEL:-}"
EVAL_API_KEY_ENV="${EVAL_API_KEY_ENV:-EVAL_API_KEY}"
eval_api_key="${!EVAL_API_KEY_ENV:-}"

# LLM judge — your own OpenAI-compatible endpoint. JUDGE_API_KEY_ENV names the
# .env variable that holds the key (default JUDGE_API_KEY). Ignored when
# NO_LLM_JUDGE=1.
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-JUDGE_API_KEY}"
judge_api_key="${!JUDGE_API_KEY_ENV:-}"

# ---- preflight: secrets + endpoints -------------------------------------
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "error: AZURE_OPENAI_API_KEY not set. The Codex proposer authenticates" >&2
  echo "       against Azure OpenAI via it — see docs/LOCOMO_LONGMEMEVAL_CODEX_AZURE.md." >&2
  exit 1
fi
if [ -z "$EVAL_BASE_URL" ] || [ -z "$EVAL_MODEL" ]; then
  echo "error: EVAL_BASE_URL / EVAL_MODEL not set. The LongMemEval scaffold queries" >&2
  echo "       a remote OpenAI-compatible endpoint — set both in .env." >&2
  exit 1
fi
case "${EVAL_BASE_URL}${EVAL_MODEL}" in
  *your-eval-endpoint*|*your-eval-model*|*replace_me*)
    echo "error: EVAL_BASE_URL / EVAL_MODEL still hold .env.example placeholders." >&2
    echo "       Edit .env with your real endpoint and model name." >&2
    exit 1 ;;
esac
if [ -z "$eval_api_key" ]; then
  echo "warning: \$$EVAL_API_KEY_ENV is empty; --api-key falls back to EMPTY." >&2
  echo "         Fine for an unauthenticated endpoint, otherwise set it in .env." >&2
fi
if [ "$NO_LLM_JUDGE" != "1" ]; then
  if [ -z "$JUDGE_BASE_URL" ] || [ -z "$JUDGE_MODEL" ]; then
    echo "error: JUDGE_BASE_URL / JUDGE_MODEL not set. LongMemEval scores answers" >&2
    echo "       with an LLM judge — set both in .env, or run with NO_LLM_JUDGE=1" >&2
    echo "       to fall back to local token/F1 scoring." >&2
    exit 1
  fi
  case "${JUDGE_BASE_URL}${JUDGE_MODEL}" in
    *your-judge-endpoint*|*your-judge-model*|*replace_me*)
      echo "error: JUDGE_BASE_URL / JUDGE_MODEL still hold .env.example placeholders." >&2
      echo "       Edit .env, or run with NO_LLM_JUDGE=1 to skip the LLM judge." >&2
      exit 1 ;;
  esac
  if [ -z "$judge_api_key" ]; then
    echo "warning: \$$JUDGE_API_KEY_ENV is empty; the LLM judge call may be rejected." >&2
  fi
fi

mkdir -p logs runs
status_file="logs/launch_longmemeval_codex_azure_${TS}.status"
: > "$status_file"
printf '[%s] LAUNCHER start ts=%s iter=%s arms=%s variant=%s split=%s limit=%s workers=%s judge=%s\n' \
  "$(date -Is)" "$TS" "$ITERATIONS" "$ARMS" "$VARIANT" "$SPLIT" "$LIMIT" "$EVAL_WORKERS" \
  "$([ "$NO_LLM_JUDGE" = "1" ] && echo local-f1 || echo llm)" >> "$status_file"

contains() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }

# ---- dataset ------------------------------------------------------------
# data/ is git-ignored, so a fresh clone has no LongMemEval data. `prepare`
# downloads the cleaned variant (~277 MB for `s`) from Hugging Face and writes
# the deterministic warmup/train/test splits.
ensure_longmemeval_dataset() {
  if [ -f "data/longmemeval/longmemeval_${VARIANT}_cleaned.json" ] \
     && [ -f "data/longmemeval/splits_${VARIANT}.json" ]; then
    return 0
  fi
  printf '[%s] DATASET_PREPARE longmemeval variant=%s (download ~277MB)\n' \
    "$(date -Is)" "$VARIANT" >> "$status_file"
  if ! python -m optimizer1.cli longmemeval prepare \
       --variant "$VARIANT" --allow-download >> "$status_file" 2>&1; then
    echo "error: 'longmemeval prepare' failed — see $status_file" >&2
    exit 1
  fi
  printf '[%s] DATASET_READY longmemeval variant=%s\n' "$(date -Is)" "$VARIANT" >> "$status_file"
}

# ---- shared CLI fragment ------------------------------------------------
# Everything common to the baseline prime and both arms: LongMemEval dataset +
# remote eval model + LLM judge + Codex(Azure) proposer. The per-run bits
# (--run-id, --iterations, arm flags, test-frontier) are added by the callers.
common_args=(
  --longmemeval
  --longmemeval-variant "$VARIANT"
  --split "$SPLIT"
  --limit "$LIMIT"
  --scaffolds "$SCAFFOLDS"
  --model "$EVAL_MODEL"
  --base-url "$EVAL_BASE_URL"
  --api-key "${eval_api_key:-EMPTY}"
  --eval-workers "$EVAL_WORKERS"
  --eval-timeout-s "$EVAL_TIMEOUT_S"
  --proposer-agent codex
  --codex-model "$CODEX_MODEL"
  --codex-reasoning-effort "$CODEX_REASONING_EFFORT"
)
if [ "$NO_LLM_JUDGE" = "1" ]; then
  common_args+=(--longmemeval-no-llm-judge)
else
  common_args+=(
    --longmemeval-judge-model "$JUDGE_MODEL"
    --longmemeval-judge-base-url "$JUDGE_BASE_URL"
  )
  [ -n "$judge_api_key" ] && common_args+=(--longmemeval-judge-api-key "$judge_api_key")
fi
[ -n "$CODEX_HOME" ] && common_args+=(--codex-home "$CODEX_HOME")
[ "$DRY_RUN" = "1" ] && common_args+=(--dry-run)

# ---- shared primed baseline --------------------------------------------
# One --iterations 0 run evaluates the memgpt_source seed scaffold and writes
# the seed frontier; both arms reuse it via --baseline-dir. The count check in
# the optimizer requires the prime to use the SAME --split/--limit as the arms,
# which it does (common_args).
baseline_run_id="longmemeval_${VARIANT}_codex_azure_baseline_${SPLIT}_limit${LIMIT}_${TS}"
BASELINE_DIR="${BASELINE_DIR:-runs/${baseline_run_id}}"

prime_longmemeval_baseline() {
  if [ -f "${BASELINE_DIR}/optimizer_summary.json" ]; then
    printf '[%s] BASELINE_REUSE %s\n' "$(date -Is)" "$BASELINE_DIR" >> "$status_file"
    return 0
  fi
  local prime_log="logs/${baseline_run_id}.log"
  printf '[%s] BASELINE_PRIME -> %s\n[%s] BASELINE_LOG %s\n' \
    "$(date -Is)" "$BASELINE_DIR" "$(date -Is)" "$prime_log" >> "$status_file"
  python -m optimizer1.cli optimize \
    "${common_args[@]}" \
    --run-id "$baseline_run_id" \
    --iterations 0 \
    --selection-policy default \
    --no-test-frontier \
    > "$prime_log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ] || [ ! -f "${BASELINE_DIR}/optimizer_summary.json" ]; then
    printf '[%s] BASELINE_PRIME_FAIL rc=%s log=%s\n' \
      "$(date -Is)" "$rc" "$prime_log" >> "$status_file"
    return 1
  fi
  printf '[%s] BASELINE_PRIME_DONE %s\n' "$(date -Is)" "$BASELINE_DIR" >> "$status_file"
}

# ---- one optimization arm ----------------------------------------------
start_one() {
  local arm="$1"
  local arm_args=() run_id log_path
  if [ "$arm" = "default" ]; then
    # No summary flag: the proposer gets the upstream-2 summary files and
    # skill mode "default" (no RunStore tools). --no-summary would instead
    # withhold the summaries entirely, breaking parity with the organized arm.
    arm_args=(--selection-policy default)
    run_id="longmemeval_${VARIANT}_codex_azure_default_${SPLIT}_${TS}"
  elif [ "$arm" = "organized" ]; then
    arm_args=(--organized --selection-policy default)
    run_id="longmemeval_${VARIANT}_codex_azure_organized_${SPLIT}_${TS}"
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
    "$(date -Is)" "$run_id" "$BASELINE_DIR" "$(date -Is)" "$log_path" >> "$status_file"

  # No --no-test-frontier: after the iterations the optimizer automatically
  # evaluates the best train-frontier candidate (--test-frontier-candidate-limit 1)
  # on the held-out LongMemEval test split (--test-frontier-limit 0 = all test tasks).
  setsid nohup python -m optimizer1.cli optimize \
    "${common_args[@]}" \
    --run-id "$run_id" \
    --iterations "$ITERATIONS" \
    --baseline-dir "$BASELINE_DIR" \
    --test-frontier-candidate-limit 1 \
    "${arm_args[@]}" \
    > "$log_path" 2>&1 < /dev/null &

  local pid=$!
  printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
  printf '%s %s %s\n' "$pid" "$run_id" "$log_path"
}

# Prepare the dataset, prime the shared baseline (foreground), then launch
# both arms in parallel.
ensure_longmemeval_dataset

prime_longmemeval_baseline || {
  echo "error: baseline prime failed — see $status_file" >&2
  exit 1
}

for arm in default organized; do
  contains "$ARMS" "$arm" || continue
  start_one "$arm"
done

printf '[%s] LAUNCHER done\n' "$(date -Is)" >> "$status_file"
printf '\nstatus: %s\n' "$status_file"
