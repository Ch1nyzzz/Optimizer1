#!/usr/bin/env bash
# SWE-bench (mini-SWE-agent) comparison experiment —
# Codex proposer (Azure OpenAI) × a remote OpenAI-compatible eval model.
#
# This is the experiment a teammate runs after pulling the repo. See
# docs/SWEBENCH_CODEX_AZURE.zh.md for the full step-by-step setup.
#
#   * proposer   = Codex CLI authenticated against Azure OpenAI (default), or the
#                  Claude Code CLI routed at a provider's anthropic-compatible
#                  endpoint. Pick with PROPOSER_AGENT=codex|claude — see the
#                  proposer knobs below and docs/SWEBENCH_CODEX_AZURE.zh.md.
#   * solver     = the DeepSeek-V4-Pro model mini-SWE-agent drives while solving
#                  each SWE-bench task — the same base model as the Terminal-Bench
#                  experiment. It runs on YOUR OWN endpoint; override
#                  SOLVER_MODEL / SOLVER_BASE_URL / SOLVER_API_KEY_ENV. Defaults
#                  point at the official DeepSeek API.
#   * eval       = the OFFICIAL SWE-bench harness runs every candidate patch in
#                  a per-instance Docker container ON THIS MACHINE. You need a
#                  working Docker daemon and plenty of free disk — there is no
#                  cloud offload for SWE-bench (unlike the Terminal-Bench run).
#
# Two arms — identical except for the RunStore tool surface. Both arms expose
# the same upstream-2 summary files (evolution_summary.jsonl +
# best_candidates.json), so the only variable across arms is the tools:
#   * default   arm: --selection-policy default
#                    (upstream-2 summaries, skill mode "default", no RunStore tools)
#   * organized arm: --organized --selection-policy default
#                    (upstream-2 summaries, skill mode "organized-summaries",
#                     generates state.md and registers RunStore tools)
# Both arms share ONE primed iter-0 baseline (prime_swebench_baseline), reused
# via --baseline-dir so the seed scaffold eval is not paid for twice.
#
# Dataset: data/swebench_train_volatile30.json (committed) carries 30 `train`
# instances + 470 `test` instances. The arms optimize on the 30 train tasks;
# after the iterations the best train-frontier candidate is evaluated on the
# held-out test set (TEST_FRONTIER_LIMIT caps it — 0 = all 470).
#
# Secrets are read from .env (git-ignored). Copy .env.example to .env and fill
# it in — see docs/SWEBENCH_CODEX_AZURE.zh.md. NEVER commit the real .env.
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

# ---- knobs (env vars) ---------------------------------------------------
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ARMS="${ARMS:-default,organized}"
ITERATIONS="${ITERATIONS:-30}"
SWE_LIMIT="${SWE_LIMIT:-30}"                 # train tasks per iteration
SWE_DATA_PATH="${SWE_DATA_PATH:-data/swebench_train_volatile30.json}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"           # concurrent Docker eval containers
EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-900}"
MINISWE_MAX_TOKENS="${MINISWE_MAX_TOKENS:-4096}"
TEST_FRONTIER_LIMIT="${TEST_FRONTIER_LIMIT:-0}"   # 0 = all 470 held-out test instances
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

# SWE-bench solver — DeepSeek-V4-Pro on your own endpoint, the same base model
# the Terminal-Bench experiment uses. Override all three to point mini-SWE-agent
# at an endpoint you have access to. SOLVER_API_KEY_ENV names the .env variable
# holding the key (default DEEPSEEK_API_KEY, shared with the Terminal-Bench run).
# SOLVER_MODEL is a litellm <provider>/<model> id — set it to match how your
# DeepSeek-V4-Pro provider/endpoint is configured.
SOLVER_MODEL="${SOLVER_MODEL:-openai/deepseek-v4-pro}"
SOLVER_BASE_URL="${SOLVER_BASE_URL:-https://api.deepseek.com/v1}"
SOLVER_API_KEY_ENV="${SOLVER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
solver_api_key="${!SOLVER_API_KEY_ENV:-}"

# ---- preflight: proposer ------------------------------------------------
# Build the proposer CLI fragment and check the secret the chosen agent needs.
proposer_args=()
if [ "$PROPOSER_AGENT" = "codex" ]; then
  if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
    echo "error: AZURE_OPENAI_API_KEY not set. The Codex proposer authenticates" >&2
    echo "       against Azure OpenAI via it — see docs/SWEBENCH_CODEX_AZURE.zh.md." >&2
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
    echo "       see docs/SWEBENCH_CODEX_AZURE.zh.md." >&2
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

# ---- preflight: solver endpoint + tooling -------------------------------
if [ -z "$solver_api_key" ]; then
  echo "warning: \$$SOLVER_API_KEY_ENV is empty; the DeepSeek-V4-Pro solver will" >&2
  echo "         fail unless DRY_RUN=1. Set it in .env, or override SOLVER_* to" >&2
  echo "         point mini-SWE-agent at an endpoint you have access to." >&2
fi
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "error: a working Docker daemon is required. The SWE-bench harness runs" >&2
  echo "       every candidate patch in a per-instance Docker container on this" >&2
  echo "       machine. Install Docker and make sure 'docker info' succeeds." >&2
  exit 1
fi
if ! command -v uvx >/dev/null 2>&1; then
  echo "error: 'uvx' not found. Both the mini-SWE-agent rollout and the SWE-bench" >&2
  echo "       evaluation are invoked through uvx. Install uv:" >&2
  echo "         curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
if [ ! -f "$SWE_DATA_PATH" ]; then
  echo "error: SWE-bench dataset not found at $SWE_DATA_PATH" >&2
  echo "       It ships with the repo — make sure you pulled the latest main." >&2
  exit 1
fi

mkdir -p logs runs
status_file="logs/launch_swebench_codex_azure_${TS}.status"
: > "$status_file"
printf '[%s] LAUNCHER start ts=%s proposer=%s iter=%s arms=%s limit=%s workers=%s test_limit=%s\n' \
  "$(date -Is)" "$TS" "$PROPOSER_AGENT" "$ITERATIONS" "$ARMS" "$SWE_LIMIT" "$EVAL_WORKERS" "$TEST_FRONTIER_LIMIT" \
  >> "$status_file"

contains() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }

# ---- shared CLI fragment ------------------------------------------------
# Everything common to the baseline prime and both arms: SWE-bench dataset +
# mini-SWE-agent run/eval commands (remote eval model) + the chosen proposer
# (proposer_args, built in the preflight above). The per-run bits (--run-id,
# --iterations, arm flags, test-frontier) are added by the callers. swebench.py
# rewrites the relative scripts/... path in the command templates to the
# trusted absolute repo-root copy before invoking it.
miniswe_run_cmd="python scripts/run_miniswe_swebench_single.py run --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir} --model $SOLVER_MODEL --base-url $SOLVER_BASE_URL --max-tokens $MINISWE_MAX_TOKENS"
if [ -n "$solver_api_key" ]; then
  miniswe_run_cmd="$miniswe_run_cmd --api-key-env $SOLVER_API_KEY_ENV"
fi

common_args=(
  --swebench
  --split train
  --limit "$SWE_LIMIT"
  --swebench-data-path "$SWE_DATA_PATH"
  --eval-timeout-s "$EVAL_TIMEOUT_S"
  --eval-workers "$EVAL_WORKERS"
  --mini-swe-agent-command "$miniswe_run_cmd"
  --mini-swe-agent-eval-command "python scripts/run_miniswe_swebench_single.py eval --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir}"
  "${proposer_args[@]}"
)
[ "$DRY_RUN" = "1" ] && common_args+=(--dry-run)

# ---- shared primed baseline --------------------------------------------
# One --iterations 0 run evaluates the seed scaffold and writes the seed
# frontier; both arms reuse it via --baseline-dir. The count check in the
# optimizer requires the prime to use the SAME --split/--limit as the arms,
# which it does (common_args).
baseline_run_id="swebench_codex_azure_baseline_train_limit${SWE_LIMIT}_${TS}"
BASELINE_DIR="${BASELINE_DIR:-runs/${baseline_run_id}}"

prime_swebench_baseline() {
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
    run_id="swebench_codex_azure_default_train_${TS}"
  elif [ "$arm" = "organized" ]; then
    arm_args=(--organized --selection-policy default)
    run_id="swebench_codex_azure_organized_train_${TS}"
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
  # on the held-out SWE-bench test split (--test-frontier-limit 0 = all 470).
  setsid nohup python -m optimizer1.cli optimize \
    "${common_args[@]}" \
    --run-id "$run_id" \
    --iterations "$ITERATIONS" \
    --baseline-dir "$BASELINE_DIR" \
    --test-frontier-candidate-limit 1 \
    --test-frontier-limit "$TEST_FRONTIER_LIMIT" \
    "${arm_args[@]}" \
    > "$log_path" 2>&1 < /dev/null &

  local pid=$!
  printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
  printf '%s %s %s\n' "$pid" "$run_id" "$log_path"
}

# Prime the shared baseline first (foreground), then launch both arms parallel.
prime_swebench_baseline || {
  echo "error: baseline prime failed — see $status_file" >&2
  exit 1
}

for arm in default organized; do
  contains "$ARMS" "$arm" || continue
  start_one "$arm"
done

printf '[%s] LAUNCHER done\n' "$(date -Is)" >> "$status_file"
printf '\nstatus: %s\n' "$status_file"
