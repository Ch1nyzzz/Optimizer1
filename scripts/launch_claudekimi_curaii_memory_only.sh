#!/usr/bin/env bash
# Launch claudekimi proposer with --selection-policy curaii on the two
# memory benchmarks (locomo + longmemeval) in parallel.  Mirrors
# launch_claudekimi_curaii_three_tasks.sh minus the swebench job, which
# still hits the SwebenchOptimizer._build_source_snapshot_workspace
# base_iter signature bug.
set -u -o pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

timestamp="${1:-$(date +%Y%m%d_%H%M%S)}"
driver_status="logs/claudekimi_curaii_memory_only_${timestamp}.status"
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

locomo_run_id="locomo_memgpt_claudekimi_curaii_v2_docker_iter30_train80_${timestamp}"
start_job "$locomo_run_id" "${memory_common[@]}" \
  --run-id "$locomo_run_id" \
  --locomo

longmemeval_run_id="longmemeval_memgpt_claudekimi_curaii_v2_docker_iter30_train100_${timestamp}"
start_job "$longmemeval_run_id" "${memory_common[@]}" \
  --run-id "$longmemeval_run_id" \
  --longmemeval

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%s] COMPLETE failures=%s\n' "$(date -Is)" "$failures" >> "$driver_status"
exit "$failures"
