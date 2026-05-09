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
driver_status="logs/codex_best_default_new_${timestamp}.status"
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
  local pattern="python -m memomemo.cli optimize .*codex54_(best3_fileaccess|default_new).*_${timestamp}"
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

# best@3: top-3 previous iterations by train passrate, fixed medium budget.
for repeat in 1 2 3; do
  run_id="locomo_memgpt_codex54_best3_fileaccess_docker_iter30_train80_r${repeat}_${timestamp}"
  start_job "$run_id" "${memory_common[@]}" \
    --run-id "$run_id" \
    --locomo \
    --selection-policy best
done

for repeat in 1 2 3; do
  run_id="longmemeval_memgpt_codex54_best3_fileaccess_docker_iter30_train100_r${repeat}_${timestamp}"
  start_job "$run_id" "${memory_common[@]}" \
    --run-id "$run_id" \
    --longmemeval \
    --selection-policy best
done

# default_new: fresh codex54 default runs, on the same prompt build as
# the random/recent/best baselines, kept distinct from the canonical
# `_default_codexlogin_autobudget_` runs already in docs/experiment_detail.md.
for repeat in 1 2 3; do
  run_id="locomo_memgpt_codex54_default_new_docker_iter30_train80_r${repeat}_${timestamp}"
  start_job "$run_id" "${memory_common[@]}" \
    --run-id "$run_id" \
    --locomo \
    --selection-policy default
done

for repeat in 1 2 3; do
  run_id="longmemeval_memgpt_codex54_default_new_docker_iter30_train100_r${repeat}_${timestamp}"
  start_job "$run_id" "${memory_common[@]}" \
    --run-id "$run_id" \
    --longmemeval \
    --selection-policy default
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%s] COMPLETE failures=%s\n' "$(date -Is)" "$failures" >> "$driver_status"
exit "$failures"
