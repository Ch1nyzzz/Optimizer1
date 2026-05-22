#!/usr/bin/env bash
# Terminal-Bench 2.0 (Terminus-KIRA) comparison experiment —
# Codex proposer (Azure OpenAI) × DeepSeek-V4 solver, every Harbor trial in a
# remote Daytona cloud sandbox.
#
# This is the experiment a teammate runs after pulling the repo. See
# docs/TERMINUS_CODEX_AZURE.md for the full step-by-step setup.
#
#   * proposer = Codex CLI authenticated against Azure OpenAI (default), or the
#                Claude Code CLI routed at a provider's anthropic-compatible
#                endpoint. Pick with PROPOSER_AGENT=codex|claude — see the
#                proposer knobs below and docs/TERMINUS_CODEX_AZURE.md.
#   * solver   = DeepSeek-V4 on YOUR OWN endpoint — override SOLVER_MODEL /
#                SOLVER_BASE_URL / SOLVER_API_KEY_ENV. Defaults point at the
#                official DeepSeek API, not Together AI.
#   * rollout  = Daytona cloud sandbox (--terminus-env daytona). Concurrency is
#                bounded by your Daytona account's concurrent-sandbox quota.
#
# Scale: 20 evolution iterations, 20 tasks/iter (first 20 of the 30-task hard
# split), 2 attempts/task, all 40 trials per iteration parallel on Daytona.
#
# Two arms — identical except for the RunStore tool surface. Both arms expose
# the same upstream-2 summary files (evolution_summary.jsonl +
# best_candidates.json), so the only variable across arms is the tools:
#   * default   arm: --selection-policy default
#                    (upstream-2 summaries, skill mode "default", no RunStore tools)
#   * organized arm: --organized --selection-policy default
#                    (upstream-2 summaries, skill mode "organized-summaries",
#                     generates state.md and registers RunStore tools)
# Both arms share ONE primed iter-0 KIRA baseline (prime_terminus_baseline),
# reused via --baseline-dir so the seed frontier is not paid for twice.
#
# After the 20 iterations each arm automatically evaluates its best train
# Pareto-frontier candidate on the full TB2 test split (the 59 held-out tasks).
#
# Secrets are read from .env (git-ignored). Copy .env.example to .env and fill
# it in — see docs/TERMINUS_CODEX_AZURE.md. NEVER commit the real .env.
#
# Each arm is detached with `setsid` so it survives the parent shell. The
# baseline prime runs in the foreground (it must finish before the arms start),
# so launch the whole script under `nohup`/`tmux` if your SSH session is flaky.
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# ---- preflight: secrets -------------------------------------------------
# DAYTONA_API_KEY is always required (every rollout is a Daytona sandbox). The
# proposer's own secret check is PROPOSER_AGENT-dependent and runs after the
# knobs block below.
if [ -z "${DAYTONA_API_KEY:-}" ]; then
  echo "error: DAYTONA_API_KEY not set. Copy .env.example to .env and fill in" >&2
  echo "       a managed-cloud key from https://app.daytona.io." >&2
  exit 1
fi

# ---- knobs (env vars) ---------------------------------------------------
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ARMS="${ARMS:-default,organized}"
ITERATIONS="${ITERATIONS:-20}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-20}"
ROLLOUT_TRIALS="${ROLLOUT_TRIALS:-2}"
ROLLOUT_CONCURRENCY="${ROLLOUT_CONCURRENCY:-40}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-2.0}"
ENV_KWARGS="${ENV_KWARGS-auto_stop_interval_mins=20,auto_delete_interval_mins=120}"
JOB_TIMEOUT_S="${JOB_TIMEOUT_S:-21600}"
TERMINUS_SOURCE_PATH="${TERMINUS_SOURCE_PATH:-references/vendor/meta-harness/reference_examples/terminal_bench_2}"
TERMINUS_TEST_TASKS="${TERMINUS_TEST_TASKS:-}"
DRY_RUN="${DRY_RUN:-0}"

# ---- proposer (PROPOSER_AGENT=codex|claude) -----------------------------
PROPOSER_AGENT="${PROPOSER_AGENT:-codex}"

# Codex proposer (Azure OpenAI) — used when PROPOSER_AGENT=codex. CODEX_MODEL
# MUST be your Azure *deployment* name — it is forwarded as `codex exec -m`,
# and config.toml's model_provider routes it to Azure. Keep it in sync with
# `model` in ~/.codex/config.toml.
CODEX_MODEL="${CODEX_MODEL:-gpt-5.1-codex}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-high}"
CODEX_HOME="${CODEX_HOME:-}"   # empty => ~/.codex (must hold the Azure config.toml)

# Claude Code proposer — used when PROPOSER_AGENT=claude. Azure OpenAI is NOT
# anthropic-compatible, so the Claude proposer is routed at a provider's
# anthropic-compatible endpoint instead. Configure your provider:
#   CLAUDE_BASE_URL    the provider's anthropic-compatible base URL
#   CLAUDE_MODEL       the model id that provider expects
#   CLAUDE_API_KEY_ENV names the .env variable holding the provider API key
#   CLAUDE_EFFORT      optional thinking effort (low|medium|high|xhigh|max)
CLAUDE_MODEL="${CLAUDE_MODEL:-}"
CLAUDE_BASE_URL="${CLAUDE_BASE_URL:-}"
CLAUDE_API_KEY_ENV="${CLAUDE_API_KEY_ENV:-ANTHROPIC_AUTH_TOKEN}"
CLAUDE_EFFORT="${CLAUDE_EFFORT:-}"

# Terminus solver — DeepSeek-V4 on your own endpoint. Override all three to
# point at the endpoint you have access to. SOLVER_MODEL is a litellm
# <provider>/<model> id — set it to match how your DeepSeek-V4-Pro
# provider/endpoint is configured.
SOLVER_MODEL="${SOLVER_MODEL:-openai/deepseek-v4-pro}"
SOLVER_BASE_URL="${SOLVER_BASE_URL:-https://api.deepseek.com/v1}"
SOLVER_API_KEY_ENV="${SOLVER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"

if [ -z "${!SOLVER_API_KEY_ENV:-}" ]; then
  echo "warning: ${SOLVER_API_KEY_ENV} not set; the DeepSeek-V4 solver will fail" >&2
  echo "         unless DRY_RUN=1. Set it in .env or export it." >&2
fi

# ---- preflight: proposer ------------------------------------------------
# Build the proposer CLI fragment and check the secret the chosen agent needs.
proposer_args=()
if [ "$PROPOSER_AGENT" = "codex" ]; then
  if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
    echo "error: AZURE_OPENAI_API_KEY not set. The Codex proposer authenticates" >&2
    echo "       against Azure OpenAI via it — see docs/TERMINUS_CODEX_AZURE.md." >&2
    exit 1
  fi
  proposer_args=(
    --proposer-agent codex
    --codex-model "$CODEX_MODEL"
    --codex-reasoning-effort "$CODEX_REASONING_EFFORT"
  )
  [ -n "$CODEX_HOME" ] && proposer_args+=(--codex-home "$CODEX_HOME")
elif [ "$PROPOSER_AGENT" = "claude" ]; then
  claude_key="${!CLAUDE_API_KEY_ENV:-}"
  if [ -z "$CLAUDE_BASE_URL" ] || [ -z "$CLAUDE_MODEL" ]; then
    echo "error: PROPOSER_AGENT=claude needs CLAUDE_BASE_URL and CLAUDE_MODEL set" >&2
    echo "       to your provider's anthropic-compatible endpoint and model id —" >&2
    echo "       see docs/TERMINUS_CODEX_AZURE.md." >&2
    exit 1
  fi
  if [ -z "$claude_key" ]; then
    echo "error: \$$CLAUDE_API_KEY_ENV is empty. PROPOSER_AGENT=claude reads the" >&2
    echo "       provider API key from the .env variable named by CLAUDE_API_KEY_ENV." >&2
    exit 1
  fi
  if ! command -v claude >/dev/null 2>&1; then
    echo "error: the 'claude' CLI is not on PATH. PROPOSER_AGENT=claude runs the" >&2
    echo "       Claude Code CLI — install it: npm install -g @anthropic-ai/claude-code" >&2
    exit 1
  fi
  proposer_args=(
    --proposer-agent claude
    --claude-base-url "$CLAUDE_BASE_URL"
    --claude-model "$CLAUDE_MODEL"
    --claude-auth-token "$claude_key"
  )
  [ -n "$CLAUDE_EFFORT" ] && proposer_args+=(--claude-effort "$CLAUDE_EFFORT")
else
  echo "error: PROPOSER_AGENT must be 'codex' or 'claude'; got '$PROPOSER_AGENT'" >&2
  exit 1
fi

mkdir -p logs runs
status_file="logs/launch_terminus_codex_azure_daytona_${TS}.status"
: > "$status_file"
printf '[%s] LAUNCHER start ts=%s proposer=%s iter=%s arms=%s split=%s limit=%s concurrency=%s\n' \
  "$(date -Is)" "$TS" "$PROPOSER_AGENT" "$ITERATIONS" "$ARMS" "$SPLIT" "$LIMIT" "$ROLLOUT_CONCURRENCY" \
  >> "$status_file"

contains() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }

# ---- shared CLI fragment ------------------------------------------------
# Everything common to the baseline prime and both arms: terminus dataset +
# Daytona rollout + DeepSeek solver + the chosen proposer (proposer_args, built
# in the preflight above). The per-run bits (--run-id, --iterations, arm flags,
# test-frontier) are added by the callers.
common_args=(
  --terminus
  --split "$SPLIT"
  --limit "$LIMIT"
  --terminus-source-path "$TERMINUS_SOURCE_PATH"
  --terminus-env daytona
  --terminus-solver-model "$SOLVER_MODEL"
  --terminus-solver-base-url "$SOLVER_BASE_URL"
  --terminus-solver-api-key-env "$SOLVER_API_KEY_ENV"
  --terminus-rollout-trials "$ROLLOUT_TRIALS"
  --terminus-rollout-concurrency "$ROLLOUT_CONCURRENCY"
  --terminus-agent-timeout-multiplier "$AGENT_TIMEOUT_MULT"
  --terminus-job-timeout-s "$JOB_TIMEOUT_S"
  "${proposer_args[@]}"
)
[ -n "$ENV_KWARGS" ] && common_args+=(--terminus-env-kwargs "$ENV_KWARGS")
[ -n "$REASONING_EFFORT" ] && common_args+=(--terminus-reasoning-effort "$REASONING_EFFORT")
[ -n "$TERMINUS_TEST_TASKS" ] && common_args+=(--terminus-test-tasks "$TERMINUS_TEST_TASKS")
[ "$DRY_RUN" = "1" ] && common_args+=(--dry-run)

# ---- shared primed baseline --------------------------------------------
# One iter-0 run produces the KIRA seed frontier; both arms reuse it via
# --baseline-dir. The count check in the optimizer requires the prime to use
# the SAME --split/--limit as the arms, which it does (common_args).
baseline_run_id="terminus_kira_codex_azure_daytona_baseline_${SPLIT}_limit${LIMIT}_${TS}"
BASELINE_DIR="${BASELINE_DIR:-runs/${baseline_run_id}}"

prime_terminus_baseline() {
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
    run_id="terminus_kira_codex_azure_daytona_default_${SPLIT}_${TS}"
  elif [ "$arm" = "organized" ]; then
    arm_args=(--organized --selection-policy default)
    run_id="terminus_kira_codex_azure_daytona_organized_${SPLIT}_${TS}"
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

  # No --no-test-frontier: after the 20 iterations the optimizer automatically
  # evaluates the best train-frontier candidate (--test-frontier-candidate-limit 1)
  # on the full TB2 test split (--test-frontier-limit 0 = all 59 held-out tasks).
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

# Prime the shared baseline first (foreground), then launch both arms parallel.
prime_terminus_baseline || {
  echo "error: baseline prime failed — see $status_file" >&2
  exit 1
}

for arm in default organized; do
  contains "$ARMS" "$arm" || continue
  start_one "$arm"
done

printf '[%s] LAUNCHER done\n' "$(date -Is)" >> "$status_file"
printf '\nstatus: %s\n' "$status_file"
