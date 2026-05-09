#!/usr/bin/env bash
# Launch claudekimi proposer with --selection-policy curaii on locomo,
# longmemeval, and swebench in parallel. Mirrors the canonical bandit_v4
# settings used in prior comparable runs so train metrics line up.
set -u -o pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

timestamp="${1:-$(date +%Y%m%d_%H%M%S)}"
driver_status="logs/claudekimi_curaii_three_tasks_${timestamp}.status"
mkdir -p logs runs
: > "$driver_status"

run_job() {
  local run_id="$1"
  shift
  local log_path="logs/${run_id}.log"
  {
    printf '[%s] START %s\n' "$(date -Is)" "$run_id"
    printf '[%s] LOG %s\n' "$(date -Is)" "$log_path"
  } >> "$driver_status"
  "$@" > "$log_path" 2>&1
  local status=$?
  printf '[%s] END %s status=%s\n' "$(date -Is)" "$run_id" "$status" >> "$driver_status"
  return "$status"
}

start_job() {
  local run_id="$1"
  if [ -d "runs/${run_id}" ]; then
    printf '[%s] SKIP %s existing_run_dir=runs/%s\n' "$(date -Is)" "$run_id" "$run_id" >> "$driver_status"
    return 0
  fi
  run_job "$@" &
  pids+=("$!")
}

pids=()

# Memory tasks (locomo + longmemeval) share the same proposer/eval config.
memory_common=(
  python -m optimizer1.cli optimize
  --iterations 30
  --split train
  --scaffolds memgpt_source
  --scaffold-extra-json @configs/source_memory.example.json
  --eval-workers 128
  --model /data/home/yuhan/model_zoo/Qwen3-8B
  --base-url http://127.0.0.1:8002/v1
  --proposer-agent kimi
  --proposer-sandbox docker
  --proposer-docker-image docker-claude-kimi:latest
  --proposer-docker-env KIMI_API_KEY
  --proposer-docker-user 1023:1023
  --proposer-docker-home /tmp
  --selection-policy curaii
)

locomo_run_id="locomo_memgpt_claudekimi_curaii_docker_iter30_train80_${timestamp}"
start_job "$locomo_run_id" "${memory_common[@]}" \
  --run-id "$locomo_run_id" \
  --locomo

longmemeval_run_id="longmemeval_memgpt_claudekimi_curaii_docker_iter30_train100_${timestamp}"
start_job "$longmemeval_run_id" "${memory_common[@]}" \
  --run-id "$longmemeval_run_id" \
  --longmemeval

# swebench (mini-swe-agent) needs its own runner / eval commands and the
# deepseek-v4-flash inference endpoint.
miniswe_run_command="python /data/home/yuhan/MemoMemo/scripts/run_miniswe_swebench_single.py run --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir} --model openai/deepseek-v4-flash --base-url https://api.deepseek.com/v1 --max-tokens 4096 --api-key-env DEEPSEEK_API_KEY"
miniswe_eval_command="python /data/home/yuhan/MemoMemo/scripts/run_miniswe_swebench_single.py eval --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir}"

swebench_run_id="swebench_miniswe_deepseek_v4_flash_claudekimi_curaii_docker_iter20_trainfirst30_w10_t900_${timestamp}"
start_job "$swebench_run_id" \
  python -m optimizer1.cli optimize \
  --swebench \
  --iterations 20 \
  --split train \
  --limit 30 \
  --swebench-data-path data/swebench_verified_full.json \
  --eval-timeout-s 900 \
  --eval-workers 10 \
  --proposer-agent kimi \
  --proposer-sandbox docker \
  --proposer-docker-image docker-claude-kimi:latest \
  --proposer-docker-env KIMI_API_KEY \
  --proposer-docker-user 1023:1023 \
  --proposer-docker-home /tmp \
  --selection-policy curaii \
  --mini-swe-agent-command "$miniswe_run_command" \
  --mini-swe-agent-eval-command "$miniswe_eval_command" \
  --run-id "$swebench_run_id"

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%s] COMPLETE failures=%s\n' "$(date -Is)" "$failures" >> "$driver_status"
exit "$failures"
