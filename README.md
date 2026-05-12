# Optimizer1

Optimizer1 is a benchmark-specific agent-evolution harness. It runs a
proposer code agent (Claude Code) inside a Docker sandbox, lets it edit a
memory or solver scaffold, evaluates the resulting candidate, and iterates.

The package is scoped to four benchmarks:

- **LoCoMo** conversational-memory QA
- **LongMemEval** long-context memory QA (variant `s` by default)
- **SWE-bench mini** code-repair (`mini-swe-agent` solver)
- **Terminal-Bench 2.0** long-horizon terminal tasks (Terminus-KIRA agent
  scaffold, Harbor-backed) — see [Terminal-Bench 2.0](#terminal-bench-20-terminus) below

Two objective axes are tracked: `passrate` (maximize) and `token_consuming`
(minimize). `average_score` is logged as a diagnostic. Every run emits a
Pareto frontier over those two axes.

## Optimal1 vs the meta-harness baseline

This repo's main pushed algorithm is **Optimal1**. The reference comparison
arm is the **meta-harness baseline** — the older `skillevolve` /
`meta-harness` style loop where the proposer gets nothing except its own
prompt and the scoped workspace.

Both arms share **everything** that is not part of the algorithm under test:
docker sandbox, Claude Code OAuth, the per-workspace
`.claude/agents/proposer.md` subagent system prompt, the auto-loaded
`CLAUDE.md` constraints, the same scaffold seed, the same eval stack,
iterations, split, and seed candidates. The only differences are three
optimization-level deltas:

| component                    | meta-harness baseline                       | Optimal1                                                 |
|------------------------------|---------------------------------------------|----------------------------------------------------------|
| `--selection-policy`         | `default` (fixed-high context, fixed base)  | `pareto` (patch base resampled from the Pareto frontier) |
| `--diagnose` subagent        | off                                         | on (writes `diagnoser_report.md` per iteration)          |
| trace-harness exposure       | `--proposer-no-trace-harness-section` (no MCP tools, raw trace only) | trace MCP server registered, including `trace_similar` lazy-embedding |
| historian subagent           | off                                         | invoked when `stagnation_count >= 1`                     |

The diagnoser, historian, and `trace_similar` are wired into Claude Code as
proper subagents and MCP tools (see `src/optimizer1/prompts/diagnoser.md`,
`historian.md`, and `src/optimizer1/traces/mcp_server.py`). They live with
the workspace, not the prompt string, so the proposer pulls them in via
`Task`/`mcp__*` tool calls only when it needs them.

The canonical side-by-side launch is in
[`scripts/launch_claude_native_baseline_vs_optimal1.sh`](scripts/launch_claude_native_baseline_vs_optimal1.sh).
It boots both arms on both LoCoMo and LongMemEval with native Claude OAuth.

## Install

```bash
cd /data/home/yuhan/Optimizer1
python -m pip install -e '.[dev]'
```

This exposes the `optimizer1` CLI entry point.

## Quickstart: Optimal1 vs baseline

### One-shot launch script

```bash
ARMS=baseline,optimal1 \
TASKS=locomo,longmemeval \
ITERATIONS=20 \
scripts/launch_claude_native_baseline_vs_optimal1.sh
```

Both arms detach with `setsid`. Status, log paths, and PIDs are appended to
`logs/launch_claude_native_<TS>.status`. Run dirs land at
`runs/<task>_claude_native_{baseline,optimal1}_<TS>/`.

### Manual invocation

Baseline arm (meta-harness style):

```bash
python -m optimizer1.cli optimize \
  --locomo \
  --run-id locomo_meta_baseline \
  --iterations 20 \
  --split train \
  --proposer-agent claude --claude-native-auth \
  --proposer-sandbox docker \
  --proposer-docker-image docker-claude:latest \
  --proposer-docker-home /home/yuhan \
  --proposer-docker-mount $HOME/.claude:/home/yuhan/.claude:ro \
  --proposer-docker-mount $HOME/.claude.json:/home/yuhan/.claude.json:ro \
  --selection-policy default \
  --proposer-no-trace-harness-section
```

Optimal1 arm (the same command, with the three deltas flipped):

```bash
python -m optimizer1.cli optimize \
  --locomo \
  --run-id locomo_optimal1 \
  --iterations 20 \
  --split train \
  --proposer-agent claude --claude-native-auth \
  --proposer-sandbox docker \
  --proposer-docker-image docker-claude:latest \
  --proposer-docker-home /home/yuhan \
  --proposer-docker-mount $HOME/.claude:/home/yuhan/.claude:ro \
  --proposer-docker-mount $HOME/.claude.json:/home/yuhan/.claude.json:ro \
  --selection-policy pareto \
  --diagnose
```

The MCP trace server (with `trace_similar` lazy embedding) is registered
automatically by the optimizer when the trace-harness section is not
suppressed. The historian subagent is deployed for any
stagnation-eligible policy (`pareto`, `curai`, `curaii`) and only invoked
once `stagnation_count >= 1`.

`trace_similar` calls Together AI through an OpenAI-compatible endpoint
to embed historical diffs at query time. Export
`OPENAI_API_KEY`/`OPENAI_BASE_URL` (or `TOGETHER_API_KEY`) and optionally
`DIFF_EMBEDDING_MODEL` before launching Optimal1. The launch script does
this automatically.

## Prepare data

```bash
optimizer1 locomo prepare --allow-download
optimizer1 longmemeval prepare --variant s --allow-download
```

The setup reuses any existing local SkillEvolve cache. Drop
`--allow-download` to fail fast when the cache is missing.

## Outputs

Per-iteration the harness writes:

- `runs/<run-id>/proposer_calls/iter_<NNN>/workspace/` — the proposer's
  Docker-mounted workspace, including the deployed
  `.claude/agents/{proposer,diagnoser,historian}.md`, the auto-loaded
  `CLAUDE.md`, the source snapshot, summaries, and reference iterations.
- `runs/<run-id>/proposer_calls/iter_<NNN>/eval/` — eval inputs/outputs
  for the candidate the proposer wrote into `pending_eval.json`.
- `runs/<run-id>/proposer_calls/iter_<NNN>/{meta,tool_access,metrics}.json`
  — proposer telemetry: input/output/cache tokens, USD cost, duration,
  per-file Read counts, Write/Edit line counts.

Run-level artifacts:

- `runs/<run-id>/best_candidates.json` — current frontier members.
- `runs/<run-id>/candidate_results/*.json` — every evaluated candidate.
- `runs/<run-id>/candidate_score_table.json` — passrate × cost table.
- `runs/<run-id>/evolution_summary.jsonl` — append-only event log; one
  row per proposer/eval/policy event.
- `runs/<run-id>/optimizer_summary.json` — aggregated proposer metrics.
- `runs/<run-id>/traces/` — `manifest.json`, `spans/iter_NNN/<candidate>.jsonl`,
  `index.db`, pre-rendered `diagnostic/iter_NNN.md`. The MCP trace tools
  read from this tree.
- `runs/<run-id>/historian_report.md` — present once historian has been
  invoked at least once during the run.
- `runs/<run-id>/progressive_state.json` — stagnation streak, last frontier
  best, used by Optimal1's pareto policy and historian gating.
- `runs/<run-id>/iteration_index.json`, `diff_summary.jsonl`,
  `retrieval_diagnostics_summary.json` — diagnostic indices.

## Reusable baselines

To precompute the seed scaffold result so iteration 0 of subsequent
optimize runs can reuse it instead of rerunning:

```bash
optimizer1 baseline \
  --splits train,test \
  --repeats 3 \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --out runs/baselines
```

Then point optimize runs at `--baseline-dir runs/baselines`. Add
`--force` to invalidate cached repeat dirs.

## Scaffolds

The single default seed is `memgpt_source` (source-informed MemGPT/Letta
seed: core memory blocks, recall search, archival search, context
compaction). Source family for evolution is `memgpt`. Reference upstream:
<https://github.com/cpacker/MemGPT>.

For SWE-bench mini, the source backend is `mini_swe_agent_source`,
selected via `--swebench`. Pass `--mini-swe-agent-source-path` if your
checkout lives outside the default location.

For Terminal-Bench 2.0, the source backend is `terminus_kira_source`,
selected via `--terminus`. See [Terminal-Bench 2.0](#terminal-bench-20-terminus).

## Terminal-Bench 2.0 (Terminus)

The `--terminus` target evolves an **agent scaffold** for
[Terminal-Bench 2.0](https://tbench.ai): a single Python file under
`agents/<name>.py` in the vendored `terminal_bench_2` reference project
(`references/vendor/meta-harness/reference_examples/terminal_bench_2/`) that
subclasses `harbor.agents.terminus_2.terminus_2.Terminus2`, seeded from
[Terminus-KIRA](https://github.com/krafton-ai/KIRA) (`agents/baseline_kira.py`).
This is the same setup as the [Meta-Harness](https://github.com/stanford-iris-lab/meta-harness)
TB2 reference experiment; the only deliberate deviations are: (1) the search
algorithm is **Optimal1** (pareto base resampling + `--diagnose` + trace MCP +
historian) instead of the plain propose→eval loop, (2) the Terminus base model
is **DeepSeek v4 Pro** at high (`max`) reasoning effort instead of Claude Opus
4.6, and (3) rollouts run in Harbor's local `docker` environment instead of
`runloop`.

Rollouts are driven by [Harbor](https://github.com/harbor-framework) — install
the optional extra so `harbor` is on PATH:

```bash
python -m pip install -e '.[dev,terminus]'
```

Splits: `--split train` uses the 30-task `hard` subset (the cheap development
split from meta-harness's `run_eval.sh`); `--split test` uses the other 59 TB2
tasks and needs the full task list (`--terminus-test-tasks task1,task2,...` or a
`--terminus-tasks-path` JSON of `{"all": [...]}`).

Side-by-side launch (claudekimi proposer in docker, DeepSeek v4 Pro solver):

```bash
ARMS=baseline,optimal1 ITERATIONS=10 \
scripts/launch_terminus_baseline_vs_optimal1.sh
```

Manual invocation:

```bash
python -m optimizer1.cli optimize \
  --terminus \
  --run-id terminus_kira_optimal1 \
  --iterations 10 \
  --split train \
  --terminus-solver-model openai/deepseek-v4-pro \
  --terminus-solver-base-url https://api.deepseek.com/v1 \
  --terminus-solver-api-key-env DEEPSEEK_API_KEY \
  --terminus-reasoning-effort high \
  --terminus-rollout-trials 2 \
  --terminus-rollout-concurrency 12 \
  --proposer-agent kimi --proposer-sandbox docker \
  --proposer-docker-image docker-claude-kimi:latest \
  --proposer-docker-home /tmp --proposer-docker-env KIMI_API_KEY \
  --selection-policy pareto --diagnose
```

Per-candidate harbor jobs land under `runs/<run-id>/terminus_jobs/<candidate_id>/`;
stdout/stderr/meta for each `harbor run` land under `runs/<run-id>/terminus_logs/`.

## Documentation

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — full Optimal1 pipeline,
  per-policy decision logic, prompt-block matrix, and headline results
  per benchmark.
- [`docs/EXPERIMENT_INSIGHTS.md`](docs/EXPERIMENT_INSIGHTS.md) — cross-run
  observations (where breakthroughs land by budget, frontier-vs-process
  read distribution, etc.).
- [`docs/experiment_detail.md`](docs/experiment_detail.md) — per-cell
  result tables and the run paths they came from.
- [`AGENTS.md`](AGENTS.md) — repository contribution guidelines and the
  paper-editing workflow used in `paper/`.

## Fetch reference repos

Optional vendor checkouts (MemGPT, mini-swe-agent, etc.) for side-by-side
inspection:

```bash
scripts/fetch_reference_repos.sh
```
