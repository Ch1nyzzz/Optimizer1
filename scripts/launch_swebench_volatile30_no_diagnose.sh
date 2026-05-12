#!/usr/bin/env bash
# Re-run the SWE-bench mini optimizer with a freshly-picked, repo-diverse,
# "volatile" 30-instance training set (data/swebench_train_volatile30.json),
# no_diagnose arm, DeepSeek v4 Pro proposer at max thinking effort.
#
# Why a new train set: the old --limit 30 slice of swebench_verified_full.json
# was the alphabetically-first 30 instances (22 astropy + 8 django). Over 6
# full-set (500-instance) evals, 15 of those 30 were never solved by the
# deepseek-v4-flash solver and 9 were always solved, leaving only ~6 movable
# instances -> a flat 0.46-0.50 ceiling with almost no optimization signal.
# The new set is 30 instances drawn from the 153 "volatile" instances
# (1..5 of 6 evals pass), balanced across django/sympy/scikit-learn/sphinx/
# pydata/matplotlib/pytest/astropy/psf and weighted toward ~3/6 (=~50%), so
# every train instance is movable and the seed baseline should sit near 0.5.
# The same file marks the other 470 instances split="test" (disjoint), so the
# end-of-run test_frontier eval is on a clean held-out set.
#
# Mirrors the prior claudecode optimal1_no_diagnose swebench run one-for-one
# except --swebench-data-path; the proposer prompt / workspace CLAUDE.md are
# whatever is currently in src/optimizer1/prompts/ (materialized per iter).
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
  echo "error: TOGETHER_API_KEY not set (expected in .env; needed for trace_similar embeddings)" >&2; exit 1
fi

# trace_similar MCP tool's lazy diff embedding -> Together AI OpenAI-compat endpoint.
export OPENAI_API_KEY="${TOGETHER_API_KEY}"
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export DIFF_EMBEDDING_MODEL="${DIFF_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-30}"
DATA_PATH="${DATA_PATH:-data/swebench_train_volatile30.json}"
DOCKER_USER_SPEC="${DOCKER_USER_SPEC:-$(id -u):$(id -g)}"

MINISWE_RUNNER="${MINISWE_RUNNER:-/data/home/yuhan/MemoMemo/scripts/run_miniswe_swebench_single.py}"
MINISWE_RUN_CMD="python ${MINISWE_RUNNER} run --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir} --model openai/deepseek-v4-flash --base-url https://api.deepseek.com/v1 --max-tokens 4096 --api-key-env DEEPSEEK_API_KEY"
MINISWE_EVAL_CMD="python ${MINISWE_RUNNER} eval --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir}"

run_id="swebench_miniswe_deepseek_v4_flash_claudecode_deepseek_v4_pro_maxeffort_optimal1_no_diagnose_volatile30_${TS}"
mkdir -p logs runs
log_path="logs/${run_id}.log"
status_file="logs/launch_${run_id}.status"
: > "$status_file"
if [ -d "runs/${run_id}" ]; then
  echo "runs/${run_id} already exists; refusing to clobber" >&2; exit 1
fi

printf '[%s] START %s\n[%s] LOG   %s\n' "$(date -Is)" "$run_id" "$(date -Is)" "$log_path" >> "$status_file"

setsid nohup python -m optimizer1.cli optimize \
  --swebench \
  --run-id "$run_id" \
  --iterations "$ITERATIONS" \
  --split train \
  --limit 30 \
  --swebench-data-path "$DATA_PATH" \
  --eval-timeout-s 900 \
  --eval-workers 10 \
  --proposer-agent claude \
  --claude-model 'deepseek-v4-pro[1m]' \
  --claude-effort max \
  --force-budget high \
  --proposer-sandbox docker \
  --proposer-docker-image docker-claude:latest \
  --proposer-docker-user "$DOCKER_USER_SPEC" \
  --proposer-docker-home /tmp \
  --selection-policy pareto \
  --test-frontier-candidate-limit 1 \
  --mini-swe-agent-command "$MINISWE_RUN_CMD" \
  --mini-swe-agent-eval-command "$MINISWE_EVAL_CMD" \
  > "$log_path" 2>&1 < /dev/null &

pid=$!
printf '[%s] PID   %s %s\n' "$(date -Is)" "$run_id" "$pid" >> "$status_file"
printf '%s %s %s\n' "$pid" "$run_id" "$log_path"
printf 'status: %s\n' "$status_file"
