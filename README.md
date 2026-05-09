# OptiHarness

OptiHarness is a clean memory-specialized evolution harness. It is separate from
`skillevolve` and is scoped to LOCOMO conversational-memory QA.

The first objective axis is `passrate` (maximize). The second objective axis is
`token_consuming` (minimize). `average_score` is reported as a diagnostic but is
not optimized because it is strongly coupled to `passrate`. Every run writes a
Pareto frontier JSON with those fields.

## What Is Included

- LOCOMO importer and deterministic split writer.
- OpenAI-compatible local model client. Defaults match the local model setup:
  `/data/home/yuhan/model_zoo/Qwen3-8B` at `http://127.0.0.1:8000/v1`.
- Built-in memory scaffold:
  - `memgpt_source`: source-informed MemGPT/Letta seed that reproduces the core
    memory hierarchy: core memory blocks, recall search, archival search, and
    context-compaction summaries.
- Pareto frontier writer over `passrate` and `token_consuming`.
- Reference source for:
  - https://github.com/cpacker/MemGPT

## Install

```bash
cd /data/home/yuhan/OptiHarness
python -m pip install -e '.[dev]'
```

For optimization, the MemGPT scaffold is used as the seed mechanism:

```bash
optiharness optimize \
  --run-id locomo_memory_source_opt \
  --iterations 20 \
  --split train \
  --scaffolds memgpt_source \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1
```

## Prepare LOCOMO

The setup can reuse the local SkillEvolve cache if present:

```bash
optiharness locomo prepare
```

To allow downloading when no local cache exists:

```bash
optiharness locomo prepare --allow-download
```

## Run LongMemEval

LongMemEval uses the same memory-scaffold base as LOCOMO and defaults to
`memgpt_source`. Scoring follows the official LongMemEval LLM-as-judge
yes/no check, with Together AI `openai/gpt-oss-120b` as the default judge:

```bash
export TOGETHER_API_KEY=...
```

Prepare the cleaned LongMemEval-S file from Hugging Face:

```bash
optiharness longmemeval prepare --variant s --allow-download
```

Run a dry-run smoke benchmark:

```bash
optiharness longmemeval benchmark \
  --variant s \
  --limit 3 \
  --out runs/longmemeval_memory_smoke
```

Optimize it through the shared MemGPT proposer path:

```bash
optiharness optimize \
  --task longmemeval \
  --longmemeval-variant s \
  --iterations 20
```

## Run Initial Memory Frontier

Dry-run retrieval scoring, useful for a quick plumbing check:

```bash
optiharness evolve \
  --split train \
  --limit 20 \
  --dry-run \
  --scaffold-extra-json @configs/source_memory.example.json \
  --out runs/locomo_memory_scaffold_smoke
```

Full local-model scoring:

```bash
optiharness evolve \
  --split train \
  --scaffold-extra-json @configs/source_memory.example.json \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --out runs/locomo_memory_scaffold_run
```

Key outputs:

- `runs/<run>/candidate_results/*.json`
- `runs/<run>/best_candidates.json`
- `runs/<run>/run_summary.json`

## Run Reusable Baselines

Run the built-in `memgpt_source` baseline once across train/test, with three
repeated trials per split:

```bash
optiharness baseline \
  --splits train,test \
  --repeats 3 \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --out runs/baselines
```

The command reuses existing repeat directories by default. Add `--force` when
you intentionally want to rerun the cached baseline trials.

Key outputs:

- `runs/baselines/baseline_summary.json`
- `runs/baselines/train/repeat_01/run_summary.json`
- `runs/baselines/test/repeat_01/run_summary.json`
- `runs/baselines/<split>/repeat_<NN>/candidate_results/*.json`

## Run Claude Code Proposer Optimization

This is the real optimization loop. It follows the `skillevolve` /
`meta-harness` pattern:

1. evaluate built-in scaffold candidates as iteration 0,
2. call `claude -p` to propose new candidate memory-scaffold code,
3. require the Claude Code session to write `pending_eval.json`,
4. import and evaluate those candidates,
5. update `best_candidates.json` by highest `passrate`.

The proposer prompt is aligned with the stricter `meta-harness` discipline:
each iteration must produce exactly one candidate, start from a clean scoped
source snapshot, use historical iterations as diagnostic references only, avoid
parameter-only tuning, and write `pending_eval.json`.

Optimization seeds exactly one default top-k candidate:
`memgpt_source=top12`. The source evolution seed family is `memgpt`.

Small dry-run:

```bash
optiharness optimize \
  --run-id smoke_opt \
  --iterations 1 \
  --limit 3 \
  --dry-run \
  --scaffold-extra-json @configs/source_memory.example.json
```

Full local-model run:

```bash
optiharness optimize \
  --run-id locomo_memory_opt \
  --iterations 20 \
  --split train \
  --baseline-dir runs/baselines \
  --scaffold-extra-json @configs/source_memory.example.json \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --claude-model claude-sonnet-4-6
```

When `--baseline-dir` is set, iteration 0 loads the precomputed baseline
candidates for the selected split instead of rerunning the selected built-in
scaffolds. It loads the default top-k candidate for each selected scaffold.

Claude writes generated candidates and source snapshots under the run output:

- `runs/<run-id>/generated/`

The harness writes proposer/eval artifacts under:

- `runs/<run-id>/proposer_calls/iter_<NNN>/`
- `runs/<run-id>/proposer_calls/iter_<NNN>/workspace/`
- `runs/<run-id>/proposer_calls/iter_<NNN>/source_snapshot/`
- `runs/<run-id>/proposer_calls/iter_<NNN>/eval/`
- `runs/<run-id>/pending_eval.json`
- `runs/<run-id>/reports/`
- `runs/<run-id>/candidate_results/`
- `runs/<run-id>/traces/` (manifest, `spans/iter_NNN/<candidate>.jsonl`, `index.db`, pre-rendered `diagnostic/iter_NNN.md` per iteration)
- `runs/<run-id>/evolution_summary.jsonl`
- `runs/<run-id>/best_candidates.json`
- `runs/<run-id>/candidate_score_table.json`
- `runs/<run-id>/retrieval_diagnostics_summary.json`
- `runs/<run-id>/iteration_index.json`
- `runs/<run-id>/diff_summary.jsonl`
- `runs/<run-id>/progressive_state.json` when adaptive progressive loading is used

Each Claude proposer session also writes `meta.json`, `tool_access.json`, and
`metrics.json`. These include input/output/cache tokens, estimated USD cost,
duration, tool counts, per-file Read counts and line counts, and Write/Edit line
counts. The optimizer appends the same proposer metrics to
`evolution_summary.jsonl` as `proposer_result` events and aggregates them in
`optimizer_summary.json`.

Proposer runs use the Docker filesystem sandbox by default. Provide
`--proposer-docker-image` for the image that contains the selected code-agent
CLI and any auth/config mounts needed by that agent. Use
`--proposer-sandbox none` only when intentionally running the proposer directly
on the host.

Default optimization uses the same scoped workspace builder with a fixed high
context budget. `--selection-policy progressive` starts with low context for
the first five proposer iterations, then escalates low -> medium -> high on
passrate stagnation and resets to low after a passrate improvement. Every
budget sees full cumulative summaries in `workspace/summaries/`; raw
per-iteration artifacts are copied into `workspace/reference_iterations/`
according to the current budget. Low and medium trace slices prioritize
failures that no previous iteration has answered correctly.

## Fetch Reference Repos

The adapters are intentionally clean and local. To inspect the upstream
reference repositories side-by-side:

```bash
scripts/fetch_reference_repos.sh
```
