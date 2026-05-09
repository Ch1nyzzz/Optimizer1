#!/usr/bin/env bash
# 3x3 ablation over progressive_low_best_count x progressive_medium_best_count
# on LoCoMo with claudekimi proposer.
#
#   low_best_count   in {1, 2, 3}
#   medium_best_count in {3, 5, 7}
#
# Worst-iter channel disabled (--no-progressive-include-worst) so the
# variation isolates the effect of the best-count knob.
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
driver_status="logs/claudekimi_progressive_bestcount_${timestamp}.status"
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
  local pattern="python -m memomemo.cli optimize .*claudekimi_progressive_l[0-9]+m[0-9]+_noworst.*_${timestamp}"
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
  --locomo
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
  --selection-policy progressive
  --no-progressive-include-worst
)

for low_n in 1 2 3; do
  for med_n in 3 5 7; do
    run_id="locomo_memgpt_claudekimi_progressive_l${low_n}m${med_n}_noworst_docker_iter30_train80_${timestamp}"
    start_job "$run_id" "${memory_common[@]}" \
      --run-id "$run_id" \
      --progressive-low-best-count "$low_n" \
      --progressive-medium-best-count "$med_n"
  done
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%s] COMPLETE failures=%s\n' "$(date -Is)" "$failures" >> "$driver_status"
exit "$failures"
