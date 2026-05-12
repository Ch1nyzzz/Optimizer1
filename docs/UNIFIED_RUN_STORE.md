# Unified Run Store — design note

Status: **proposal** (no code written yet). Scope assumes the parallel
selection-policy cleanup that reduces `selection_policy` to `{default,
pareto}` (Optimal1 = `pareto` + `--diagnose` + historian + trace MCP).
Nothing here depends on that cleanup landing first, but the examples use
the post-cleanup vocabulary.

This note answers three questions that came up while reviewing how a run
records its history:

1. What does an "edit-the-scaffold" agent actually need to read?
2. Are the existing trace tools the right orthogonal set?
3. How do we collapse the two parallel record systems (`summaries/` flat
   files + the `traces/` harness) into one without breaking the
   proposer-facing contract?

---

## 1. First principles: what the editing agents need

The proposer, the diagnoser subagent, and the historian subagent all do
the same thing: **read history + read the current state → decide the next
edit**. Everything they need falls into three buckets:

| Question | What it is asking | Backing data |
|---|---|---|
| **(a) What does the artifact look like now?** | current best scaffold source, this iteration's diff base, which files exist | ② modification side (source snapshot + Pareto frontier state) |
| **(b) What did past edits do?** | which diffs raised / lowered the score, on which tasks, why | ① evaluation side × ② modification side, **joined** |
| **(c) What is broken right now?** | this iteration's `fail` / `regressed` / `persistent_fail` tasks | ① evaluation side |

(b) is the load-bearing one and it is a **join of the two streams**. That
is why the trace records and the summary records cannot be treated as two
independent things — they have to line up on `iteration` / `candidate_id`
/ `task_id`.

### The two provenance streams

Neither `summaries/` nor `traces/` is a *source*. Both are *derived* from
two upstream raw artifacts:

| Stream | Raw artifact (who produces it) | Contents |
|---|---|---|
| **① evaluation** | `runs/<run>/candidate_results/<id>.json` (running the candidate on the benchmark) | per-task `passed` / `score`, prediction, retrieval hits/misses, token consumption — "how the candidate scored" |
| **② modification + selection** | `runs/<run>/proposer_calls/iter_NNN/` (`diff.patch`, `agent/tool_access.json`, `agent/metrics.json`) **plus** the optimizer's in-memory decisions (`selection_policy`, `patch_base`, `budget`, reference iterations) | what changed this iteration and how it was produced: the diff, which files changed, which files the proposer read/wrote, token cost, which policy, which base |

---

## 2. Current state: two parallel record systems

### A. `summaries/` — the flat-file set (what the proposer reads as context)

Written by `optimizer.py` + `post_eval.py`; `_copy_workspace_summaries`
(`optimizer.py:1158`) mirrors these into the proposer workspace under
`summaries/`:

| File | Contents | Writer | Stream |
|---|---|---|---|
| `evolution_summary.jsonl` | append-only event log: *candidate rows* `{ts, iteration, candidate(full to_dict), proposal, self_best}` and *`proposer_result` rows* `{ts, iteration, event, selection_policy, proposer_agent, returncode, timed_out, proposer_metrics(tokens/cost/duration/tool calls), usage, files_read, files_written, grep_requests, tool_counts, budget, reference_iterations, call_dir, workspace_dir, target_system, …}` | `_append_summary` / `_append_proposer_result_event` → `_append_event` | ①(candidate rows) + ②(proposer_result rows) |
| `best_candidates.json` | running Pareto-best set (scaffold_name, candidate_id, config, score, passrate, token_consuming, …) | frontier update | ① + frontier rule |
| `candidate_score_table.json` | flat per-candidate score table | `_write_candidate_score_table_from_candidates` | ① |
| `retrieval_diagnostics_summary.json` | per-iteration retrieval diagnostics (hit/miss previews) — LoCoMo / LongMemEval only | `post_eval._append_retrieval_diagnostics_summary` | ① |
| `diff_summary.jsonl` | per-iteration diff stats `{iteration, iteration_dir, diff_path, diff_digest_path, files_changed, insertions, deletions}` | `_append_diff_summary` | ② |
| `iteration_index.json` | iteration → call_dir / result_path / retrieval_diagnostics_path index | `_write_iteration_index` | ① + ② |

End-of-run rollups also exist: `optimizer_summary.json` (cost totals from
`_aggregate_proposer_metrics`, which re-parses `evolution_summary.jsonl`),
`run_summary.json`, `pareto_frontier.json`, `test_frontier_summary.json`.

### B. `traces/` — the trace harness (structured, queryable)

Written by `TraceHarness.record_iteration` (`traces/harness.py`), one call
per evaluated batch:

- `traces/manifest.json` — backend / schema version, benchmark,
  `baseline_path`, `is_baseline_run`
- `traces/spans/iter_NNN/<candidate_id>.jsonl` — one `Trace` per task:
  `{trace_id, iteration, candidate_id, task_id, benchmark, summary
  (adapter-defined: question/gold/prediction/score/passed/…), diff
  (status vs baseline), spans (nested retrieval/generation/tool steps)}`
- `traces/index.db` — SQLite, tables: `traces`, `diffs` (status:
  `baseline` / `breakthrough` / `regressed` / `persistent_fail` /
  `stable_pass` / `no_baseline`, + `delta`), `file_modifications`
  (iter → path), `iteration_diffs` (raw diff text), `diff_embeddings`
  (lazy, written by the MCP layer), `iteration_meta` (`patch_base`,
  `budget`, `selection_policy`, `advanced_frontier`, `on_pareto_frontier`,
  `passrate`, `mean_score`, `proposer_call_dir`), `manifest`
- `traces/diagnostic/iter_NNN.md` — pre-rendered markdown the proposer /
  diagnoser reads
- the `trace_*` MCP tools (`traces/mcp_server.py`) query `index.db`

### The asymmetry

The **evaluation side is a declarative store + query tools**: data lives
in `index.db` + `spans/*.jsonl`, has a `SCHEMA_VERSION`, is idempotent per
iteration, and the vs-baseline semantics (`breakthrough` / `regressed` /
`persistent_fail` / `stable_pass`) are *computed into the data*, not
re-explained in the prompt. The proposer **pulls** the slice it wants via
MCP.

The **modification side is an imperative push + heuristics baked into the
optimizer**, and the proposer cannot query it:

- `_bandit_core_files` — hard-codes which files are always "hot"
- `_bandit_reference_iterations` / `_best_reference_iterations` /
  `_recent_reference_iterations` / `_random_reference_iterations` /
  `_reference_iterations_for_budget` / `_curaii_select_for_budget` — six
  functions for "which historical iterations to show the proposer", one
  per policy
- `_progressive_budget_for_iteration` — file-budget tiering
- `_pareto_select_base` — sample a base from the frontier
- `_recompute_bandit_scores` — per-file Beta-smoothed UCB scoring (the
  big block in `PIPELINE.md §4.3`)
- `_evict_failed_iteration` (`optimizer.py:2146`) — a failed iteration has
  to **hand-edit** `evolution_summary.jsonl` / `diff_summary.jsonl` to
  remove the dangling rows

These all answer the same question — "which modification-side history do
we feed the proposer?" — but the answer is decided by the optimizer and
pushed into the prompt, instead of being pulled by the proposer through a
tool. And the data lives in scattered JSON files (`bandit_state.json`,
`progressive_state.json`, `pareto_frontier.json`, `iteration_index.json`,
`diff_summary.jsonl`), which is precisely why `_evict_failed_iteration`
needs a manual patch.

> After the `selection_policy → {default, pareto}` cleanup, most of the
> bandit/progressive machinery is gone. What remains on the modification
> side is small — `_pareto_select_base`, "full-history reference
> iterations", stagnation accounting for the historian — but it is still
> imperative push, not a queryable store. The same unification applies.

---

## 3. Are the seven `trace_*` tools the right orthogonal set?

Close, with two gaps.

| Tool | What it does | Verdict |
|---|---|---|
| `trace_task_history(task)` | one task's status timeline across iterations | keep — irreducible |
| `trace_compare_iterations(L, R)` | per-task pass/fail comparison + classification between any two iters | keep — the workhorse |
| `trace_file_history(path)` | iterations that touched `path` + each one's outcome | keep — the entry point for question (b) |
| `trace_candidate_outcome(iter, cand)` | one `(iter, cand)` deep view: passrate, modified files, representative task examples | keep |
| `trace_iteration_metadata(iters)` | `iteration_meta` rows | keep |
| `trace_similar(diff_or_text)` | semantically similar historical diffs | keep — the only fuzzy one |
| `trace_list_tasks(iter)` | enumerate task ids | trivial — fold into the others |

The seven are mutually orthogonal (nothing reconstructs another, except
`list_tasks`). **The two gaps:**

1. **No tool returns a diff's text.** `iteration_diffs` holds the patch,
   `trace_similar` uses it, but nothing exposes it — the proposer has to
   go read `proposer_calls/iter_NNN/diff.patch` off the filesystem.
2. **No tool returns "what the proposer did that iteration."** Token
   consumption, files read, tools used — that lives in the
   `proposer_result` row of `evolution_summary.jsonl`, not in `index.db`
   and not behind a tool. So `_recompute_bandit_scores` has to re-parse
   the jsonl.

Also `traces/diagnostic/iter_NNN.md` is a pre-baked view of what
`trace_compare_iterations` + `trace_candidate_outcome` already return — a
redundant render of data the tools expose.

---

## 4. Target design — make the modification side symmetric with the evaluation side

One sentence: **promote `index.db` to the single `RunStore`; the
"heuristics" become pure functions over it; `summaries/` becomes a thin
export view; add two modification-side query tools so the proposer can
pull instead of being pushed.**

### 4.1 `RunStore` — the one write path

`RunStore` owns `runs/<run>/traces/index.db` and is the **only** writer.
The optimizer stops writing summaries inline (`_append_summary`,
`_append_diff_summary`, `_write_iteration_index`,
`_write_candidate_score_table_from_candidates`) and stops separately
driving `TraceHarness.record_iteration`. Instead:

```python
store = RunStore(run_dir, benchmark=..., baseline_path=...)
store.record_proposer_call(iteration, result, selection_meta)  # ② -> proposer_calls table (NEW) + iteration_meta
store.record_candidates(iteration, candidates)                 # ② -> candidates table (NEW)
store.record_eval(iteration, candidates)                       # ① -> traces / diffs / spans  (today's record_iteration)
store.record_diff(iteration, diff_text)                        # ② -> iteration_diffs / file_modifications (exists)
```

Everything is idempotent per iteration (the trace tables already are), so
`_evict_failed_iteration` is deleted — a retried iteration overwrites
itself cleanly.

### 4.2 New tables (close the schema gap)

```sql
CREATE TABLE IF NOT EXISTS proposer_calls (
    iteration         INTEGER PRIMARY KEY,
    proposer_agent    TEXT,
    returncode        INTEGER,
    timed_out         INTEGER,
    metrics_json      TEXT,     -- proposer_metrics: tokens / cost / duration / tool_calls / …
    usage_json        TEXT,
    files_read_json   TEXT,     -- {path: meta}
    files_written_json TEXT,
    grep_requests_json TEXT,
    tool_counts_json  TEXT,
    call_dir          TEXT,
    workspace_dir     TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    iteration    INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    scaffold_name TEXT,
    passrate     REAL,
    average_score REAL,
    token_consuming INTEGER,
    config_json  TEXT,     -- the candidate's config dict
    result_path  TEXT,
    proposal_json TEXT,
    PRIMARY KEY (iteration, candidate_id)
);
```

`iteration_meta`, `iteration_diffs`, `file_modifications` already exist
and already cover `iteration_index.json` / `diff_summary.jsonl`.

### 4.3 `summaries/` becomes `RunStore.export_summaries(dir)`

The six flat files become pure projections — byte-compatible, so the
proposer prompt paths, the MCP server, and the tests do not change:

| File | Projection |
|---|---|
| `evolution_summary.jsonl` | `candidates` rows + `proposer_calls` rows, ordered by iteration |
| `candidate_score_table.json` | `SELECT … FROM candidates` |
| `retrieval_diagnostics_summary.json` | from the `kind="retrieval"` spans (or keep `post_eval`'s writer pointed at the store) |
| `diff_summary.jsonl` | `iteration_diffs` + `file_modifications` aggregated per iteration |
| `iteration_index.json` | `proposer_calls.call_dir` + `candidates.result_path` joined per iteration |
| `best_candidates.json` | `candidates JOIN diffs` under the frontier rule; `iteration_meta.on_pareto_frontier` already tracks it |
| `optimizer_summary.json` (cost totals) | `SUM(...)` over `proposer_calls.metrics_json` instead of re-parsing jsonl |

`_copy_workspace_summaries` becomes "render `summaries/` from the store".
Keep the files on disk — humans grep them and the prompt cites their
paths — but the rule is **the DB is the truth, the files are a faithful
projection**.

### 4.4 Heuristics take `RunStore`, not scattered JSON

`_pareto_select_base`, the reference-iteration selector, and (if any
survives the cleanup) per-file scoring become `(store, config) -> …` pure
functions. Behaviour does not change — only the data source. After the
`{default, pareto}` cleanup the surviving pieces are:

- `select_reference_iterations(store, policy)` — `default` and `pareto`
  both return "all recorded iterations" (optionally capped by
  `--max-reference-iterations`)
- `pareto_base(store)` — sample uniformly from the current frontier
- stagnation counter — read/write one row in the store (replacing
  `progressive_state.json`)

### 4.5 Two new MCP tools — symmetric tool surface

The read API becomes one MCP server backed by `RunStore`, with an
evaluation half and a modification half that mirror each other:

| Evaluation side (exists) | Modification side |
|---|---|
| `compare_iterations(L, R)` | `iteration_diff(iter)` — raw patch + `files_changed` / `insertions` / `deletions` — **NEW** |
| `candidate_outcome(iter, cand)` | `proposer_call(iter)` — files read/written, token cost, tools used — **NEW** |
| `file_history(path)` | (same tool — `file_history` already joins `file_modifications`) |
| `iteration_metadata(iters)` | (same — meta already carries `patch_base` / `budget` / `policy`) |
| `similar_changes(diff_or_text)` | (same) |
| `task_history(task)` | — |

With `iteration_diff` and `proposer_call` available, the proposer prompt
stops **pushing** reference iterations and summaries inline; the proposer
pulls them the same way it already pulls traces. `reference_iterations`
degrades from "copy a stack of `iter_NNN` bundles into the workspace" to
"name a few iteration numbers in the prompt and let the agent query
them". `traces/diagnostic/*.md` can be retired (or kept as a cache).

### 4.6 The minimal orthogonal read set, after unification

One read-only MCP server, backed by `RunStore`:

- `iteration_meta(iters?)` — per iteration: policy / base / budget /
  passrate / frontier flag
- `iteration_diff(iter)` — what changed that iteration (patch + stats)
- `proposer_call(iter)` — how the proposer changed it (files read/written
  / token cost / tools)
- `candidate_outcome(iter, cand)` — how that iteration scored (per-task
  deep view)
- `compare_iterations(L, R)` — per-task comparison of any two iterations
  (covers vs-baseline / vs-frontier-best / vs-patch-base)
- `task_history(task)` — one task's timeline
- `file_history(path)` — a file's edit history + each edit's outcome
- `similar_changes(diff_or_text)` — semantically similar historical
  changes

Three evaluation-side granularities (task / iter-pair / candidate), three
modification-side granularities (meta / diff / proposer-call), two
cross-iteration aggregates (`task_history`, `file_history`), one fuzzy
search. No overlap; nothing reconstructs another.

Write side (used by the optimizer, not a tool):
`RunStore.record_{proposer_call, candidates, eval, diff}` +
`export_summaries(dir)` + the pure-function heuristics
(`select_reference_iterations`, `pareto_base`, …).

---

## 5. Migration — four steps, each its own commit, no behaviour change

1. **One write path.** Introduce `RunStore`; route `record_eval` (==
   today's `TraceHarness.record_iteration`) and the summary writers
   through it; add the `proposer_calls` and `candidates` tables; delete
   `_evict_failed_iteration` (idempotency makes it unnecessary).
2. **`summaries/` becomes an export view.** Replace the inline writers
   with `RunStore.export_summaries`; assert byte-compatibility against
   golden files in the tests.
3. **Heuristics take `RunStore`.** `_pareto_select_base` /
   `select_reference_iterations` (and whatever per-file scoring survives)
   read the store instead of `*_state.json` / `*.jsonl`.
4. **Two new tools + retire the renderer.** Add `iteration_diff` and
   `proposer_call` MCP tools; stop pushing reference iterations and
   summaries inline in the proposer prompt (cite tools instead); retire
   or demote `traces/diagnostic/*.md`.

### Impact

- **Proposer prompt** (`proposer_prompt.py`): unchanged through Step 2
  (paths identical). Step 4 removes the reference-iterations / summaries
  push and adds the two tool names to the trace-harness section.
- **MCP** (`traces/mcp_server.py`, `traces/query.py`): Step 4 adds two
  tools; the existing seven keep working.
- **Diagnoser / historian prompts**: Step 4 — they currently point at
  `traces/diagnostic/iter_NNN.md`; switch them to the tools.
- **Tests**: Step 1–2 need golden-file round-trip tests
  (`store.record_* → export_summaries → byte-equal`); existing
  `tests/traces/*` and `test_optimizer.py` summary assertions stay valid
  because the files do not change shape.
- **`SCHEMA_VERSION`**: bump (two new tables); `manifest.json` and the
  `manifest` table both record it, so old runs are detectable.

---

## 6. What this is not

- Not a change to *what* gets recorded — same fields, fewer files, one
  source of truth.
- Not deleting the flat files — they remain as a projection for humans
  and for prompt-path stability.
- Not coupled to the policy cleanup — it lands on top of whatever set of
  selection policies exists, and the surviving modification-side
  heuristics get pulled into `RunStore` either way.
