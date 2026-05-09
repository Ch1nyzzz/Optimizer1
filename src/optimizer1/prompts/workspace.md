# Optimizer1 workspace — constraints and conventions

This file describes constraints and conventions that hold across all
optimization iterations in this workspace. The agent's role and
objective are loaded separately as the session system prompt; this
file is project context, not identity.

## Quality Gate

Before writing `pending_eval.json`, verify that the candidate is a real
mechanism change, is not just a `top_k`/`window`/threshold/weight variant, does
not use gold answers at inference time, does not hardcode benchmark-specific
answers, and uses the isolated source snapshot for source edits.
Parameter changes are allowed only as supporting details of a mechanism change.
A candidate whose substantive change is only `top_k`, window size, thresholds,
weights, prompt length, or context budget will be rejected.
Run a lightweight syntax/import smoke check against the edited snapshot before
writing `pending_eval.json`; do not run the full harness evaluation.

## Edit scope

Source-backed baseline memories and source bases are read-only and expensive
to rebuild. If your source edit changes build/database-construction logic or
other persisted memory construction semantics, use a fresh `source_base_dir`
and a new stable `build_tag`. For upstream source edits, route explicit paths
such as `memgpt_source_path` through the copied source snapshot.

The copied `optimizer1` package inside this workspace is intentionally
benchmark-scoped and incomplete. Do not add runtime imports from repo-root
harness modules such as `optimizer1.evaluation`, `optimizer1.pareto`,
`optimizer1.metrics`, optimizer modules, or any module not listed in the
iteration's available-files section. Keep `optimizer1/__init__.py` minimal;
do not make it import top-level repository APIs. Candidate runtime code
must not access benchmark raw data, `candidate_results/**`, or
`optimizer1.metrics.score_prediction`.

## pending_eval.json conventions

The exact required output path and the JSON schema (with the live
substitutions for `scaffold_name`, `source_family`,
`reference_iterations`, `source_snapshot_path`, and
`extra.source_project_path`) are given in the iteration-specific
section of the user message. Independent of those substitutions:

- The `candidates` array must contain exactly one candidate.
- `top_k` must be a single integer.
- Use a source-backed scaffold whenever you edit the copied scaffold
  source.
- The `source_project_path` under `extra` must point at the edited
  snapshot project source when files under
  `project_source/src/optimizer1/` are modified, or at the edited
  mini-SWE-agent snapshot when working on the coding-agent tree.
- If you create a wrapper module under the iteration's generated
  directory, keep it small and route source-backed mechanisms through
  the clean edited snapshot.
- The `reference_iterations` field records the raw bundles available
  for diagnosis; it is not a parent list.
