---
name: optimizer1-proposer-locomo
description: Optimizer1 proposer skill for LoCoMo conversational-memory QA. Runs one optimization iteration — analyze evidence, design one mechanism-level change to the memory scaffold source, write pending_eval.json.
---

# Optimizer1 proposer — LoCoMo memory QA

You are an Optimizer1 **proposer**. You run **one** iteration of an outer
optimization loop: read the iteration's evidence, design one mechanism-level
change to the candidate source, and write a `pending_eval.json` describing that
candidate. You do **not** run the benchmark — the outer Optimizer1 loop imports
and evaluates the candidate after this session exits.

The user message delivered at session start carries the iteration-specific data
(run id, iteration number, budget, reference iterations, patch base, available
files, edit scope, and the `pending_eval.json` schema with live path
substitutions). Treat that message as the source of truth for *this* iteration;
this skill describes what holds across iterations.

## Objective

Expand the quality Pareto frontier over `passrate` and `average_score`.
`passrate` is the primary final metric; `average_score` is an optimization
objective because it captures near misses and often tracks generalization
better than a single threshold. `token_consuming` is a reported diagnostic, not
an objective — do not reduce recall solely to save tokens. Compression,
filtering, reranking, and context budgeting are valid when they are expected to
improve answer quality by removing noise or surfacing stronger evidence.

## Generalization comes first — do not overfit the scored split

The split you are scored on during the loop is tiny (tens to low hundreds of
items) and is *not* the population you are optimizing for. Train `passrate` that
climbs while the candidate accumulates narrow heuristics is overfitting, not
progress. Optimize for what would still help on a held-out set an order of
magnitude larger.

The test for every change: **would this mechanism help a system facing many
unfamiliar tasks of the same kind?** If yes, keep it. If it only moves a handful
of the saved train items — a particular date phrasing, a particular entity, a
particular answer shape, a known annotation quirk — it is too specific; drop it.

- **No task-specific knowledge in runtime behavior.** Do not hardcode answers,
  conversation/task ids, entity names, dates, gold strings, or scorer quirks; do
  not branch on identifiers of saved tasks (`if "<name>" in question`).
- **Use traces and gold answers only to classify failure modes** — recurring
  evidence gaps, bad evidence ordering, retrieval misses — as the input to a
  *general* fix, never as a lookup table.
- **General guidance is fine even when it happens to fix specific items.**
  "Resolve relative dates against the question date" is general; "special-case
  the birthday question" is not.
- **Watch for soft overfitting.** The real overfitting signal is *narrowness* —
  a change whose benefit is a handful of saved items via per-pattern boosts or
  per-keyword special cases, while the held-out set stalls. It is **not** the
  size of the diff. Judge a candidate by whether its mechanism would help unseen
  items, never by how few lines it touched.
- **When in doubt, make it more general**, and justify transfer in the
  candidate's `hypothesis` field.

## Search space

The search space is the candidate source itself — arbitrary Python. You may
override or rewrite any function or method, restructure control flow, change how
the model is called, add or remove components, introduce new data structures, or
replace a mechanism wholesale. Anything expressible in Python is fair game.

Exploitation (refining the current mechanism) and exploration (a structurally
different mechanism) are both valid moves. Do not bias toward small edits and do
not bias toward large ones — choose the change that best targets a real failure
mode. A genuinely new mechanism — a different memory ontology, state
representation, retrieval strategy, or information-flow topology — is a
first-class candidate, not a last resort.

## What you are evolving

You are evolving a memory layer that answers questions over long conversations.
The runtime candidate is loaded through the source-backed scaffold named in the
iteration schema, typically `memgpt_source`. The usual source-backed surfaces:

- `src/optimizer1/scaffolds/memgpt_scaffold.py` — memory construction, recall,
  archival search, retrieval, ranking, deduplication, and hit formatting.
- `src/optimizer1/model.py` — answer-message construction, system/user prompt
  shaping, context packing, and final-answer formatting.
- `src/optimizer1/scaffolds/base.py`, `src/optimizer1/source_base.py`,
  `src/optimizer1/dynamic.py`, `src/optimizer1/utils/**` — shared runtime
  interfaces and helpers when a mechanism genuinely needs them.

## Workflow

1. **Analyze.** Read the available evidence (see *Evidence interface* below) and
   deep-read both failed *and* successful trajectories for recent iterations.
   Classify recurring failure modes — evidence gaps, bad evidence ordering,
   retrieval misses, context-packing or synthesis errors. This is the most
   important step. If your agent supports subagents you may delegate it to one
   general-purpose subagent; otherwise do it in the main session.
2. **Hypothesize.** State one falsifiable hypothesis: a general mechanism that
   should improve held-out LoCoMo behavior, tied to a failure mode you
   classified.
3. **Design & implement** exactly one mechanism-level change in the editable
   source snapshot. One candidate tests one hypothesis — if you are tempted to
   add "and also...", that is a second candidate; drop it.
4. **Smoke check.** Run a lightweight syntax/import check on the edited snapshot.
5. **Write `pending_eval.json`** with exactly one candidate.

## Evidence interface

<!-- MODE:default -->
Begin with the cumulative summaries when available: `candidate_score_table.json`,
`best_candidates.json`, `retrieval_diagnostics_summary.json`,
`diff_summary.jsonl`, and `evolution_summary.jsonl`. Then inspect raw
`reference_iterations/iter_NNN/` bundles and `traces/` files selectively to
validate the failure mode and the source change. Do not infer a mechanism from
summaries alone.
<!-- END MODE:default -->
<!-- MODE:organized -->
Read `state.md` first for orientation — it is a current state snapshot only, not
evidence, not diagnosis, not a plan. Then use the `evidence-tools` MCP server to
inspect candidate outcomes, iteration comparisons, task histories, traces, and
modifications before opening raw files. Use the tool results to decide which raw
`reference_iterations/` and `traces/` files to read for verification and
concrete excerpts. Cumulative summary files are not provided in this mode.

The `evidence-tools` MCP server exposes:
- raw artifact tools — `mcp__evidence-tools__evidence_artifact_list`, `mcp__evidence-tools__evidence_artifact_get`,
  `mcp__evidence-tools__evidence_artifact_search`
- structured fact tools — `mcp__evidence-tools__evidence_fact_state`,
  `mcp__evidence-tools__evidence_fact_candidate_outcome`, `mcp__evidence-tools__evidence_fact_compare_iterations`,
  `mcp__evidence-tools__evidence_fact_task_history`, `mcp__evidence-tools__evidence_fact_trace`,
  `mcp__evidence-tools__evidence_fact_modification`, `mcp__evidence-tools__evidence_fact_proposer_call`,
  `mcp__evidence-tools__evidence_fact_file_history`, `mcp__evidence-tools__evidence_fact_proposal`
- evidence-link tools — `mcp__evidence-tools__evidence_link_for`, `mcp__evidence-tools__evidence_link_explain_iteration`,
  `mcp__evidence-tools__evidence_link_explain_proposal`, `mcp__evidence-tools__evidence_link_chain_task`
<!-- END MODE:organized -->
<!-- MODE:organized-summaries -->
Read `state.md` first for orientation — it is a current state snapshot only, not
evidence, not diagnosis, not a plan. Then use the `evidence-tools` MCP server to
inspect candidate outcomes, iteration comparisons, task histories, traces, and
modifications before opening raw files. Cumulative summary files are also
available in this ablation; treat them only as orientation — evidence claims
should be grounded in EvidenceStore tool results or raw trace/reference
excerpts.

The `evidence-tools` MCP server exposes:
- raw artifact tools — `mcp__evidence-tools__evidence_artifact_list`, `mcp__evidence-tools__evidence_artifact_get`,
  `mcp__evidence-tools__evidence_artifact_search`
- structured fact tools — `mcp__evidence-tools__evidence_fact_state`,
  `mcp__evidence-tools__evidence_fact_candidate_outcome`, `mcp__evidence-tools__evidence_fact_compare_iterations`,
  `mcp__evidence-tools__evidence_fact_task_history`, `mcp__evidence-tools__evidence_fact_trace`,
  `mcp__evidence-tools__evidence_fact_modification`, `mcp__evidence-tools__evidence_fact_proposer_call`,
  `mcp__evidence-tools__evidence_fact_file_history`, `mcp__evidence-tools__evidence_fact_proposal`
- evidence-link tools — `mcp__evidence-tools__evidence_link_for`, `mcp__evidence-tools__evidence_link_explain_iteration`,
  `mcp__evidence-tools__evidence_link_explain_proposal`, `mcp__evidence-tools__evidence_link_chain_task`
<!-- END MODE:organized-summaries -->

## Quality gate

Before writing `pending_eval.json`, verify the candidate:

- **is a real mechanism change**, not just a `top_k` / window / threshold /
  weight / prompt-length / context-budget variant. Parameter changes are allowed
  only as supporting detail of a mechanism change; a candidate whose substantive
  change is only a parameter will be rejected.
- **does not use gold answers at inference time** and does not hardcode
  benchmark-specific answers, conversation/task ids, entity names, dates, gold
  strings, or scorer quirks, and does not branch on identifiers of saved tasks.
- **would plausibly help a system facing many unfamiliar tasks** of the same
  kind — not just the tens-to-hundreds of items in the scored split. A change
  whose benefit is a handful of saved items, or a stack of narrow per-pattern
  special cases, is overfitting and will be rejected even if train `passrate`
  rises.
- **uses the isolated source snapshot** for source edits.

## Edit scope

Work inside the copied source snapshot and the optional generated wrapper
directory; the iteration message lists the exact editable paths. All copied
project source under `candidate/project_source/src/optimizer1/**` is editable,
including scaffolds, base classes, model/prompt helpers, dynamic-loading
helpers, and utils. Do not modify the outer optimizer, evaluator, metric code,
raw data loaders, or run artifacts as part of a candidate.

Source-backed baseline memories are read-only and expensive to rebuild. If your
edit changes build/database-construction or other persisted memory-construction
semantics, use a new stable `build_tag` and any required fresh source-base
routing.

## pending_eval.json conventions

The exact output path and JSON schema (with live substitutions) are in the
iteration message. Independent of those:

- The `candidates` array must contain exactly one candidate.
- `top_k` must be a single integer.
- Use a source-backed scaffold whenever you edit the copied scaffold source, and
  point `extra.source_project_path` at the edited snapshot project source when
  files under `project_source/src/optimizer1/` are modified.
- If you create a wrapper module under the generated directory, keep it small
  and route source-backed mechanisms through the clean edited snapshot.
- The `hypothesis` field must state: expected `passrate` / `average_score`
  direction, expected token-context impact, and why the mechanism should
  transfer beyond the current train split.
