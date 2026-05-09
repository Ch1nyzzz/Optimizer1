#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

timestamp="${1:-$(date +%Y%m%d_%H%M%S)}"
max_parallel="${MAX_PARALLEL:-2}"
driver_status="logs/codex_file_access_baselines_${timestamp}.status"
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

wait_for_slot() {
  local pattern="python -m memomemo.cli optimize .*codex54_.*3_fileaccess.*_${timestamp}"
  while [ "$(pgrep -af "$pattern" | wc -l)" -ge "$max_parallel" ]; do
    sleep 30
  done
}

start_job() {
  local run_id="$1"
  if [ -d "runs/${run_id}" ]; then
    printf '[%s] SKIP %s existing_run_dir=runs/%s\n' "$(date -Is)" "$run_id" "$run_id" >> "$driver_status"
    return 0
  fi
  wait_for_slot
  run_job "$@" &
  pids+=("$!")
}

pids=()

memory_common=(
  python -m memomemo.cli optimize
  --iterations 30
  --split train
  --scaffolds memgpt_source
  --scaffold-extra-json @configs/source_memory.example.json
  --eval-workers 128
  --model /data/home/yuhan/model_zoo/Qwen3-8B
  --base-url http://127.0.0.1:8002/v1
  --proposer-agent codex
  --codex-model gpt-5.4
  --proposer-sandbox docker
  --proposer-docker-image docker-codex:latest
)

for policy in random recent; do
  for repeat in 1 2 3; do
    run_id="locomo_memgpt_codex54_${policy}3_fileaccess_docker_iter30_train80_r${repeat}_${timestamp}"
    start_job "$run_id" "${memory_common[@]}" \
      --run-id "$run_id" \
      --locomo \
      --selection-policy "$policy"
  done
done

for policy in random recent; do
  for repeat in 1 2 3; do
    run_id="longmemeval_memgpt_codex54_${policy}3_fileaccess_docker_iter30_train100_r${repeat}_${timestamp}"
    start_job "$run_id" "${memory_common[@]}" \
      --run-id "$run_id" \
      --longmemeval \
      --selection-policy "$policy"
  done
done

miniswe_run_command="python /data/home/yuhan/MemoMemo/scripts/run_miniswe_swebench_single.py run --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir} --model openai/deepseek-v4-flash --base-url https://api.deepseek.com/v1 --max-tokens 4096 --api-key-env DEEPSEEK_API_KEY"
miniswe_eval_command="python /data/home/yuhan/MemoMemo/scripts/run_miniswe_swebench_single.py eval --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir}"

miniswe_common=(
  python -m memomemo.cli optimize
  --swebench
  --iterations 20
  --split train
  --limit 30
  --swebench-data-path data/swebench_verified_full.json
  --eval-timeout-s 900
  --eval-workers 10
  --proposer-agent codex
  --codex-model gpt-5.4
  --proposer-sandbox docker
  --proposer-docker-image docker-codex:latest
  --mini-swe-agent-command "$miniswe_run_command"
  --mini-swe-agent-eval-command "$miniswe_eval_command"
)

for policy in random recent; do
  run_id="swebench_miniswe_deepseek_v4_flash_codex54_${policy}3_fileaccess_docker_iter20_trainfirst30_w10_t900_${timestamp}"
  start_job "$run_id" "${miniswe_common[@]}" \
    --run-id "$run_id" \
    --selection-policy "$policy"
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%s] COMPLETE failures=%s\n' "$(date -Is)" "$failures" >> "$driver_status"
exit "$failures"
