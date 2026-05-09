#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
project_root="$(pwd)"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY must be set in the environment or .env}"

timestamp="${OPTIMIZER_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
iterations="${OPTIMIZER_ITERATIONS:-20}"
limit="${OPTIMIZER_LIMIT:-0}"
eval_workers="${OPTIMIZER_EVAL_WORKERS:-1}"
eval_timeout_s="${OPTIMIZER_EVAL_TIMEOUT_S:-300}"
propose_timeout_s="${OPTIMIZER_PROPOSE_TIMEOUT_S:-2400}"
opencode_model="${OPENCODE_MODEL:-deepseek/deepseek-v4-pro}"
eval_model="${DEEPSEEK_EVAL_MODEL:-deepseek-v4-pro}"
eval_base_url="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}"

common=(
  env PYTHONPATH=src
  python -m optimizer1.cli optimize
  --iterations "${iterations}"
  --split train
  --limit "${limit}"
  --model "${eval_model}"
  --base-url "${eval_base_url}"
  --api-key "${DEEPSEEK_API_KEY}"
  --eval-timeout-s "${eval_timeout_s}"
  --proposer-agent opencode
  --opencode-model "${opencode_model}"
  --proposer-sandbox none
  --selection-policy default
  --eval-workers "${eval_workers}"
)

"${common[@]}" \
  --task locomo \
  --run-id "locomo_opencode_deepseek_v4_pro_default_${timestamp}"

longmemeval_extra=()
if [ -n "${TOGETHER_API_KEY:-}" ]; then
  longmemeval_extra+=(--longmemeval-judge-api-key "${TOGETHER_API_KEY}")
else
  longmemeval_extra+=(--longmemeval-no-llm-judge)
fi

"${common[@]}" \
  --task longmemeval \
  --longmemeval-variant s \
  "${longmemeval_extra[@]}" \
  --run-id "longmemeval_s_opencode_deepseek_v4_pro_default_${timestamp}"

miniswe_run_command="python ${project_root}/scripts/run_miniswe_swebench_single.py run --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir} --model openai/deepseek-v4-pro --base-url ${eval_base_url} --max-tokens 4096 --api-key-env DEEPSEEK_API_KEY"
miniswe_eval_command="python ${project_root}/scripts/run_miniswe_swebench_single.py eval --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir}"

"${common[@]}" \
  --task swebench \
  --swebench-data-path data/swebench_verified_full.json \
  --mini-swe-agent-source-path references/vendor/mini-swe-agent \
  --mini-swe-agent-command "${miniswe_run_command}" \
  --mini-swe-agent-eval-command "${miniswe_eval_command}" \
  --run-id "swebench_opencode_deepseek_v4_pro_default_${timestamp}"
