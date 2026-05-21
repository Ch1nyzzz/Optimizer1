---
name: optimizer1-proposer-terminus
description: Optimizer1 proposer skill for Terminal-Bench 2.0 AgentHarness evolution. Runs one optimization iteration — analyze trajectories, design one mechanism-level change to the agent scaffold, write pending_eval.json.
---

# Optimizer1 proposer — Terminal-Bench 2.0 AgentHarness

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

Primary objective: maximize `passrate` on the Terminal-Bench 2.0 hard split.
`average_score` is reported alongside it and tracks partial progress;
`token_consuming`, turn count, tool-call count, and wall-clock are reported
diagnostics, not objectives. Predict cost impact, but do not trade away task
resolution to shrink it.

## Generalization comes first — do not overfit the scored split

The split you are scored on during the loop is small and is *not* the population
you are optimizing for. Train `passrate` that climbs while the agent accumulates
narrow heuristics is overfitting, not progress. Optimize for what would still
help on a held-out set an order of magnitude larger.

The test for every change: **would this mechanism help an agent facing many
unfamiliar terminal tasks?** If yes, keep it. If it only moves a handful of
saved tasks — a particular command, a particular tool, a known verifier quirk —
it is too specific; drop it.

- **No task-specific knowledge in runtime behavior.** Do not hardcode task ids,
  task names, file names, verifier shortcuts, or solution scripts; do not branch
  on identifiers of saved tasks (no `if task contains 'async'`).
- **Use traces only to classify failure modes** — recurring control-loop
  mistakes, bad command construction, broken observation handling, premature
  completion — as the input to a *general* fix, never as a lookup table.
- **General guidance is fine even when it happens to fix specific items.** "Back
  up a file before opening it with a tool that modifies on read" is general;
  "special-case the DNA-assembly verifier" is not.
- **Watch for soft overfitting.** The real overfitting signal is *narrowness* —
  a change whose benefit is a handful of saved tasks via per-pattern special
  cases, while the held-out set stalls. It is **not** the size of the diff.
  Judge a candidate by whether its mechanism would help unseen tasks, never by
  how few lines it touched.
- **When in doubt, make it more general**, and justify transfer in the
  candidate's `hypothesis` field.

## Search space

The search space is the candidate agent source — arbitrary Python. You may
override any method, call any library, make raw API calls, add new tools, change
how the LLM is called, rewrite command execution, intercept and transform
observations, restructure the episode loop, or replace a mechanism wholesale.
Anything expressible in Python is fair game. The only hard constraint is that
the candidate agent must remain compatible with the Harbor/Terminus runtime and
subclass the runtime `AgentHarness`/`Terminus2` base in the same way the seed
agent does, so it loads through the import path recorded in `pending_eval.json`.

Exploitation (refining the current mechanism) and exploration (a structurally
different mechanism) are both valid moves. Do not bias toward small edits and do
not bias toward large ones — choose the change that best targets a real failure
mode. A genuinely new mechanism — a different loop topology, tool set, context
strategy, or information-flow structure — is a first-class candidate, not a last
resort.

## What you are evolving

You are evolving an `AgentHarness` agent scaffold for Terminal-Bench 2.0 task
resolution. The candidate is loaded through the import path recorded in
`pending_eval.json`. Usual editable surfaces:

- a new or edited `agents/<candidate>.py` file containing class `AgentHarness`;
- prompt templates under the copied `prompt-templates/` directory when the
  candidate explicitly points to them;
- methods on the copied agent class that control LLM calls, tool parsing,
  command execution, loop control, completion confirmation, prompt-template
  selection, context summarization, and multimodal/image handling.

## Workflow

1. **Analyze.** Read the available evidence (see *Evidence interface* below) and
   deep-read both failed *and* successful trajectories for recent iterations.
   Classify recurring agent failure modes — context overflow, bad tool use,
   command-execution errors, premature or missed completion. This is the most
   important step. If your agent supports subagents you may delegate it to one
   general-purpose subagent; otherwise do it in the main session.
2. **Hypothesize.** State one falsifiable hypothesis: a general AgentHarness
   mechanism that should improve held-out Terminal-Bench behavior, tied to a
   failure mode you classified.
3. **Design & implement** exactly one mechanism-level change in the editable
   source snapshot. One candidate tests one hypothesis — if you are tempted to
   add "and also...", that is a second candidate; drop it.
4. **Smoke check.** Run a lightweight syntax/import check on the edited snapshot.
5. **Write `pending_eval.json`** with exactly one candidate.

## Evidence interface

<!-- MODE:default -->
Begin with whichever cumulative summary files are present under `summaries/` —
`evolution_summary.jsonl` (the full event history) and `best_candidates.json`
(the current quality frontier). If no `summaries/` directory is provided this
run, work directly from the raw `reference_iterations/iter_NNN/` bundles
instead. Either way, inspect raw `reference_iterations/iter_NNN/` bundles and
`traces/` files selectively to validate the failure mode and the source change.
Do not infer a mechanism from summaries alone.
<!-- END MODE:default -->
<!-- MODE:organized -->
Read `state.md` first for orientation — it is a current state snapshot only, not
evidence, not diagnosis, not a plan. Then use the `runstore-tools` MCP server to
inspect candidate outcomes, iteration comparisons, task histories, traces, and
modifications before opening raw files. Use the tool results to decide which raw
`reference_iterations/` and `traces/` files to read for verification and
concrete excerpts. Cumulative summary files are not provided in this mode.

The `runstore-tools` MCP server exposes:
- raw artifact tools — `mcp__runstore-tools__runstore_artifact_list`, `mcp__runstore-tools__runstore_artifact_get`,
  `mcp__runstore-tools__runstore_artifact_search`
- structured fact tools — `mcp__runstore-tools__runstore_fact_state`,
  `mcp__runstore-tools__runstore_fact_candidate_outcome`, `mcp__runstore-tools__runstore_fact_compare_iterations`,
  `mcp__runstore-tools__runstore_fact_task_history`, `mcp__runstore-tools__runstore_fact_trace`,
  `mcp__runstore-tools__runstore_fact_modification`, `mcp__runstore-tools__runstore_fact_proposer_call`,
  `mcp__runstore-tools__runstore_fact_file_history`, `mcp__runstore-tools__runstore_fact_proposal`
- evidence-link tools — `mcp__runstore-tools__runstore_link_for`, `mcp__runstore-tools__runstore_link_explain_iteration`,
  `mcp__runstore-tools__runstore_link_explain_proposal`, `mcp__runstore-tools__runstore_link_chain_task`
<!-- END MODE:organized -->
<!-- MODE:organized-summaries -->
Read `state.md` first for orientation — it is a current state snapshot only, not
evidence, not diagnosis, not a plan. Then use the `runstore-tools` MCP server to
inspect candidate outcomes, iteration comparisons, task histories, traces, and
modifications before opening raw files. Cumulative summary files are also
available in this ablation; treat them only as orientation — evidence claims
should be grounded in RunStore tool results or raw trace/reference
excerpts.

The `runstore-tools` MCP server exposes:
- raw artifact tools — `mcp__runstore-tools__runstore_artifact_list`, `mcp__runstore-tools__runstore_artifact_get`,
  `mcp__runstore-tools__runstore_artifact_search`
- structured fact tools — `mcp__runstore-tools__runstore_fact_state`,
  `mcp__runstore-tools__runstore_fact_candidate_outcome`, `mcp__runstore-tools__runstore_fact_compare_iterations`,
  `mcp__runstore-tools__runstore_fact_task_history`, `mcp__runstore-tools__runstore_fact_trace`,
  `mcp__runstore-tools__runstore_fact_modification`, `mcp__runstore-tools__runstore_fact_proposer_call`,
  `mcp__runstore-tools__runstore_fact_file_history`, `mcp__runstore-tools__runstore_fact_proposal`
- evidence-link tools — `mcp__runstore-tools__runstore_link_for`, `mcp__runstore-tools__runstore_link_explain_iteration`,
  `mcp__runstore-tools__runstore_link_explain_proposal`, `mcp__runstore-tools__runstore_link_chain_task`
<!-- END MODE:organized-summaries -->

## Quality gate

Before writing `pending_eval.json`, verify the candidate:

- **is a real mechanism change**, not just a retry-count / timeout / context-
  budget / prompt-length variant. Parameter changes are allowed only as
  supporting detail of a mechanism change; a candidate whose substantive change
  is only a parameter will be rejected.
- **does not inspect held-out solutions at inference time** and does not
  hardcode task ids, task names, file names, verifier shortcuts, or solution
  scripts; candidate runtime code must not call the evaluator.
- **would plausibly help an agent facing many unfamiliar terminal tasks** — not
  just the saved split. A change whose benefit is a handful of saved tasks, or a
  stack of narrow per-pattern special cases, is overfitting and will be rejected
  even if train `passrate` rises.
- **stays compatible** with the Harbor/Terminus runtime and the recorded import
  path, and uses the isolated source snapshot for source edits.

## Edit scope

Work inside the copied Terminal-Bench reference source snapshot and the optional
generated wrapper directory; the iteration message lists the exact editable
paths. Do not modify `meta_harness.py`, `claude_wrapper.py`, the evaluator, task
verifiers, official task files, the outer optimizer, or run artifacts as part of
a candidate.

## pending_eval.json conventions

The exact output path and JSON schema (with live substitutions) are in the
iteration message. Independent of those:

- The `candidates` array must contain exactly one candidate.
- The candidate must load through the recorded import path and subclass the
  Harbor/Terminus `AgentHarness` base.
- If you create a wrapper module under the generated directory, keep it small
  and route source-backed mechanisms through the clean edited snapshot.
- The `hypothesis` field must state: expected `passrate` direction, expected
  token / turn / tool-call impact, and why the mechanism should transfer beyond
  the current train split.
