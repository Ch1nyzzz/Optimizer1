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
max_parallel="${MAX_PARALLEL:-3}"
driver_status="logs/claudekimi_swebench_random_recent_${timestamp}.status"
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
  local pattern="python -m optimizer1.cli optimize .*claudekimi_.*(3_fileaccess|bandit_v4_autobudget).*_${timestamp}"
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

miniswe_run_command="python /data/home/yuhan/MemoMemo/scripts/run_miniswe_swebench_single.py run --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir} --model openai/deepseek-v4-flash --base-url https://api.deepseek.com/v1 --max-tokens 4096 --api-key-env DEEPSEEK_API_KEY"
miniswe_eval_command="python /data/home/yuhan/MemoMemo/scripts/run_miniswe_swebench_single.py eval --source-path {source_path} --instance-path {instance_path} --patch-path {patch_path} --task-dir {task_dir}"

miniswe_common=(
  python -m optimizer1.cli optimize
  --swebench
  --iterations 20
  --split train
  --limit 30
  --swebench-data-path data/swebench_verified_full.json
  --eval-timeout-s 900
  --eval-workers 10
  --proposer-agent kimi
  --proposer-sandbox docker
  --proposer-docker-image docker-claude-kimi:latest
  --proposer-docker-env KIMI_API_KEY
  --proposer-docker-user 1023:1023
  --proposer-docker-home /tmp
  --mini-swe-agent-command "$miniswe_run_command"
  --mini-swe-agent-eval-command "$miniswe_eval_command"
)

for policy in random recent; do
  run_id="swebench_miniswe_deepseek_v4_flash_claudekimi_${policy}3_fileaccess_docker_iter20_trainfirst30_w10_t900_${timestamp}"
  start_job "$run_id" "${miniswe_common[@]}" \
    --run-id "$run_id" \
    --selection-policy "$policy"
done

# bandit_v4 baseline on the same swebench config (current branch is bandit-v4,
# so --selection-policy bandit dispatches to the v4 ref-iter selection rule).
bandit_v4_run_id="swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v4_autobudget_docker_iter20_trainfirst30_w10_t900_bw16_${timestamp}"
start_job "$bandit_v4_run_id" "${miniswe_common[@]}" \
  --run-id "$bandit_v4_run_id" \
  --selection-policy bandit \
  --bandit-reward-window 16

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%s] COMPLETE failures=%s\n' "$(date -Is)" "$failures" >> "$driver_status"
exit "$failures"
