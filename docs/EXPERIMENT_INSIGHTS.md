# Experiment Insights

This note captures cross-run observations that are not visible from the
headline result tables alone. The scope is the retained experiments in
`docs/PIPELINE.md`, with the following exclusions for budget-stage analysis:

- default and default+direction are excluded because every iteration is fixed
  `high`;
- claude opus is excluded because there is no retained v3 bandit comparison;
- text classification is excluded from budget-stage comparisons because no
  bandit result is retained.

Unless otherwise stated, a breakthrough means a training iteration whose
candidate score improved over all earlier candidates in that same run. Score is
ordered by `passrate`, then `average_score`; equal-score ties are not counted
as new breakthroughs.

## Final Frontier vs Process Breakthroughs

Do not infer where optimization happens from `best_candidates.json` alone.
That file records the candidates left on the final frontier, so it can collapse
many earlier improvements into one or two retained candidates.

For example, `locomo_memgpt_claudekimi_bandit_v4_autobudget_..._r1_20260505_040626`
(the LoCoMo·kimi/bandit "best by test" run cited in `experiment_detail.md`)
has only one final retained best candidate (iter011, test 0.3616), but it
had four process-breakthrough events during training:

| iteration | budget | passrate | average_score |
|---:|---|---:|---:|
| 2 | medium | 0.3750 | 0.5397 |
| 3 | medium | 0.4250 | 0.5531 |
| 11 | high | 0.4375 | 0.5855 |
| 30 | high | 0.4750 | 0.5971 |

Note that the train-best (0.4750 at iter 30) was *not* the held-out
test-best — the iter011 candidate generalized better to the held-out
set. This is the typical pattern across the experiment_detail set: train
passrate climbs across the run while the test-leading iter is often
mid-run, which is why per-iter test eval is needed to surface the right
candidate.

## Breakthroughs By Budget

Across the 17 bandit and progressive auto-budget runs cited at the artifact
level in `docs/experiment_detail.md` (default-family excluded because every
iter is fixed `high` by construction), the process-breakthrough distribution
aggregated by (benchmark, proposer, policy) cell:

| benchmark | proposer | policy | n runs | low | medium | high | total |
|---|---|---|---:|---:|---:|---:|---:|
| LoCoMo | claudekimi | bandit | 2 | 0 | 2 | 6 | 8 |
| LoCoMo | claudekimi | progressive | 2 | 3 | 1 | 3 | 7 |
| LoCoMo | codex54 | bandit | 2 | 0 | 3 | 4 | 7 |
| LoCoMo | codex54 | progressive | 2 | 4 | 0 | 1 | 5 |
| LongMemEval | claudekimi | bandit | 3 | 2 | 6 | 7 | 15 |
| LongMemEval | claudekimi | progressive | 1 | 7 | 0 | 1 | 8 |
| LongMemEval | codex54 | bandit | 3 | 1 | 7 | 5 | 13 |
| LongMemEval | codex54 | progressive | 2 | 5 | 0 | 4 | 9 |
| **subtotal (auto-budget)** |  |  | **17** | **22** | **19** | **31** | **72** |

SWE-bench mini retained artifacts are all single-tier (force=high or
default+direction at high). Including them for completeness, all
breakthroughs land at high by construction:

| benchmark | proposer | policy | n runs | low | medium | high | total |
|---|---|---|---:|---:|---:|---:|---:|
| SWE-bench mini | claudekimi | default+direction | 1 | 0 | 0 | 1 | 1 |
| SWE-bench mini | claudekimi | bandit_v3 budgethigh | 3 | 0 | 0 | 2 | 2 |

The PIPELINE.md docs-only SWE-bench rows (`default` train 0.5000,
`progressive` train 0.5333, `bandit (fixedsource)` train 0.5333,
verified-set 0.4580 / 0.6200 / 0.6400) are not aggregated here because
their per-iter breakthrough sequences are not retained in
`runs/`; see `docs/PIPELINE.md` Section 5.3 for the headline numbers.

On the full training horizon, in the adaptive (auto-budget) cells low+medium
together account for 41/72 = 57% of breakthroughs, and high alone for
31/72 = 43%. This is the optimistic view of the adaptive policies: narrow
contexts find a meaningful fraction of the early gains, and medium budget
is especially productive for bandit on LongMemEval (6 medium breakthroughs
across 3 kimi-bandit runs, 7 across 3 codex-bandit runs).

## After Iteration 5

The full-horizon count is biased by the first few iterations. Early candidates
start from a low baseline, so they have more headroom and often improve without
needing much context. A stricter view is to count only breakthroughs after
iteration 5.

Raw breakthrough counts and exposure after iteration 5, restricted to the
17 bandit/progressive runs in `docs/experiment_detail.md`:

| benchmark | proposer | policy | n | iters L/M/H | breakthroughs L/M/H |
|---|---|---|---:|---|---|
| LoCoMo | claudekimi | bandit | 2 | 0 / 16 / 34 | 0 / 0 / 6 |
| LoCoMo | claudekimi | progressive | 2 | 4 / 6 / 40 | 0 / 1 / 3 |
| LoCoMo | codex54 | bandit | 2 | 0 / 19 / 31 | 0 / 1 / 4 |
| LoCoMo | codex54 | progressive | 2 | 2 / 3 / 45 | 0 / 0 / 1 |
| LongMemEval | claudekimi | bandit | 3 | 0 / 38 / 37 | 0 / 1 / 7 |
| LongMemEval | claudekimi | progressive | 1 | 5 / 2 / 18 | 3 / 0 / 1 |
| LongMemEval | codex54 | bandit | 3 | 0 / 25 / 50 | 0 / 1 / 5 |
| LongMemEval | codex54 | progressive | 2 | 6 / 7 / 37 | 0 / 0 / 4 |
| **total** |  |  | **17** | **17 / 116 / 292** | **3 / 4 / 31** |

Aggregated by budget:

| budget | post-5 iterations | post-5 breakthroughs | breakthrough rate |
|---|---:|---:|---:|
| low | 17 | 3 | 17.6% |
| medium | 116 | 4 | 3.4% |
| high | 292 | 31 | 10.6% |
| **total** | **425** | **38** | **8.9%** |

The picture is clearer than before: high captures **31/38 = 81.6% of
post-iter-5 breakthroughs in absolute count**, and per-iter rate (10.6%)
is materially higher than medium (3.4%). Low has the highest rate (17.6%)
but only 17 post-5 opportunities (longmemeval claudekimi progressive
contributes 5/17 of those), so the denominator is too small to draw a
strong claim. The interpretation:

- low and medium remain useful for early exploration and cheap candidate
  discovery within the first 5 iters (covered in `Breakthroughs By
  Budget` above);
- after iter 5, medium becomes the weakest tier (only 4/116 = 3.4% rate)
  — most policies do not productively spend medium late in a run;
- high is where the late-stage rescue work happens — both in absolute
  count (31) and per-iter rate (10.6%);
- LongMemEval claudekimi progressive r1 is unusual: 3 of the 5 post-5
  low-budget breakthroughs in the entire sample come from this single
  run, where the proposer kept finding gains at low budget through iter
  17 (see Section 1 / experiment_detail.md for the per-iter sequence).

## Token Trend Context

The budget-stage breakthrough counts should be read alongside proposer-token
curves:

- bandit usually ramps up: `low` is cheap, `medium` is moderate, and `high`
  approaches default-like context cost;
- progressive does not always save tokens. It saves clearly for claudekimi
  memory runs, but can be flat or more expensive for codex54 and SWE-bench
  mini because the proposer may compensate for smaller granted context by
  reading fewer but larger or more information-dense files;
- default should not be mixed into budget-stage breakthrough counts, because
  default has no budget transition. Its breakthroughs are all `default-high`
  by construction.

## Default Proposer Lacks True Incremental Optimization

Output-token volume is *not* the right diagnostic for proposer saturation on
the `default` family. Across five default-series LoCoMo+kimi runs (one
`default`, two `default+direction`, two `default+autobudget`),
per-iteration `output_tokens` on the proposer agent stay flat or rise into
late iterations — the proposer is not producing shorter responses over time.
The actual saturation signal is in `diff.patch` line counts.

Per-iter `+add` / `-rem` line counts (first 1/3 vs last 1/3 of 30
iterations, averaged):

| run | first out_tok | last out_tok | first +add / -rem | last +add / -rem |
|---|---:|---:|---|---|
| `default` (5/01) | 21,643 | 27,723 | 156 / 13 | 236 / 20 |
| `default+direction` r1 (5/02 a) | 19,870 | 35,200 | 130 / 15 | **742 / 5** |
| `default+direction` r2 (5/02 b) | 31,725 | 31,609 | 287 / 17 | 437 / 31 |
| `default+autobudget` r1 (5/04) | 28,473 | 25,017 | 308 / 15 | 312 / **2** |

Output tokens move within ±1.8x first-third vs last-third — there is no
late-iter shrinkage. But in two of the four runs the `-rem` (lines removed
or replaced from the patch base) collapses toward 1–5 lines/iter in the
late phase while `+add` stays high or grows; the patch shape becomes
*append-only* against the original baseline.

The deeper issue is structural, not just empirical in those two runs: the
proposer's patch base is **always the original baseline** by construction
— past best candidates enter only as the `diff_digest` summary in the
history block, not as the workspace the proposer edits on top of. Three
consequences follow, and they are the actual saturation mechanism on
`default`:

- **Each iteration freelances on top of the baseline.** Because the
  starting point never rotates to the current best candidate, different
  iterations explore optimizations in scattered directions rather than
  building on each other.
- **Validated `+add` blocks do not survive into the next iteration's
  patch.** Whatever extra logic the previous iter added is, by
  construction, absent from the next iter's diff base — it can only
  reappear if the model independently re-derives the same idea from the
  digest summary.
- **Rising `+add` line counts look like ongoing work but are not
  layered.** Late-iter patches rewrite "another flavour of first-layer
  optimization" against the same baseline rather than stacking a second
  layer onto an already-working first layer.

The implication for future ablations: replacing the proposer's patch base
with the current frontier-best candidate (rather than the fixed baseline)
is the natural test for whether `default` saturates because the search has
genuinely run out of headroom or because the evaluator never lets the
search compound. Under the current design the two are observationally
indistinguishable from `best_candidates.json`.

## Progressive Advantage: Fewer Files, Better Iteration Routing

The headline claim is that adaptive policies (bandit, progressive) sit on
a Pareto frontier with default: **comparable or lower per-propose cost
AND comparable or higher test score**. We use two run-selection rules
deliberately:

- **Test-score columns** report mean ± std across all retained runs in
  the cell (the way accuracy is reported in the rest of `docs/experiment_detail.md`),
  because a single best-test run is statistically thin for an accuracy
  claim.
- **Per-propose cost columns** are taken from the best-by-test single
  run in each cell (matching `Per-Propose Token Cost Across Policies`
  above), because cost is measured on the run we are actually proposing
  as the cell's representative result.

Restricted to the claudekimi cells in `docs/experiment_detail.md`:

| benchmark | policy | n | test mean ± std | best test | tools/iter (best-run) | unique_files/iter (best-run) | total/iter (best-run) |
|---|---|---:|---|---:|---:|---:|---:|
| LoCoMo | default-family | 3 | 0.3315 ± 0.0153 | 0.3423 | 46.6 | 4.0 | 3.42M |
| LoCoMo | bandit | 2 done + 1 docs | 0.3554 ± 0.0084 | 0.3616 | 41.8 | 3.0 | 2.63M |
| LoCoMo | progressive | 2 done + 1 docs | 0.3545 ± 0.0214 | **0.3734** | **35.2** | 15.1* | **1.86M** |
| LongMemEval | default-family | 2 done + 1 docs | 0.4983 ± 0.0301 | 0.5300 | 49.8 | 3.7 | 3.71M |
| LongMemEval | bandit | 3 | **0.5100 ± 0.0378** | **0.5450** | **38.2** | **3.3** | **2.55M** |
| LongMemEval | progressive | 2 done + 1 docs | 0.5050 ± 0.0132 | 0.5200 | 47.3 | 3.8 | 3.37M |

*The LoCoMo progressive `unique_files/iter` of 15.1 comes from
`PIPELINE.md`'s `files/iter` column on the docs-only pipeline progressive
row, which uses a different definition (workspace files touched per
iter) than the artifact-derived `unique_files_read` in the other rows.
Treat it as a different metric.

**Mean-test reading** (across all retained runs in each cell):

- LoCoMo: bandit mean 0.3554 ≈ progressive mean 0.3545, both **clear of
  default's 0.3315 by ~2.3 ppt** (default-family std 0.0153, so the gap
  is ~1.5σ). Adaptive policies' mean test is materially above default's
  mean.
- LongMemEval: bandit mean 0.5100 > progressive mean 0.5050 > default
  mean 0.4983. Smaller margins (~1-1.2 ppt), all within std overlap, so
  the per-cell mean ranking is suggestive rather than significant on
  this benchmark.

**Best-test reading** (single run per cell, used for cost):

- LoCoMo: **progressive Pareto-dominates** — best test 0.3734 (highest
  in the cell) at 1.86M total/iter (lowest cost) and 35.2 tools/iter
  (lowest tool count).
- LongMemEval: **bandit Pareto-dominates** — best test 0.5450 (highest
  in the cell) at 2.55M total/iter (lowest cost) and 38.2 tools/iter
  (lowest tool count).

So the cost-quality picture is Pareto-favorable across both run-selection
rules: mean test confirms the adaptive policies sit at or above default
for accuracy, and best-test cost confirms they do so without paying more
per-propose. The Pareto reading is robust to which rule is used for
accuracy reporting; only the cost column requires picking a specific
run.

The mechanism-level claim — that best/worst iteration labels concentrate
the proposer's reads on a small subset of available iter dirs — is
verified against current bandit artifacts in the
`Reference-Iteration Read Distribution` section below (14-22x reads/slot
concentration on best-tagged dirs).

This is presented as a Pareto trade-off rather than a strict ordering:
not every retained adaptive run beats default on cost or test (e.g.
LoCoMo bandit_v3 r1 test 0.3458 / per-iter 4.24M is below the default
best-test on accuracy and above on cost). The claim is that the
*best-by-test* adaptive run from each cell sits at or below the
default-family baseline on cost, while the *cell mean* test is at or
above default's mean. Cheaper operating points at lower test scores
(e.g. bandit force=low at LongMemEval, train 0.6300 / test 0.4550) are
explored in the `Force-Budget Ablation` subsection later.

## Per-Propose Token Cost Across Policies

This section compares default, bandit, and progressive at per-propose
granularity for all four (benchmark, proposer) cells, using the
"best retained run by test score" rows in `docs/experiment_detail.md`.
It extends `Progressive Advantage` above by including bandit and codex54,
and adds a mechanistic decomposition of why a longer initial prompt produces
lower total token consumption per propose.

### Cross-cell summary table

| cell | policy | input | output | cache_read | total | tools | read_files | read_lines | best test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo claudekimi | default | 151.3k | 26.7k | 3.25M | **3.42M** | 46.6 | 27.7 | 4,825.5 | 0.3423 |
|  | bandit | 156.9k | 21.1k | 2.45M | **2.63M** (-23%) | 41.8 | 26.3 | 3,970.3 | 0.3616 |
|  | progressive | 138.9k | 25.5k | 1.70M | **1.86M** (-46%) | 35.2 | NA | NA | **0.3734** |
| LoCoMo codex54 | default | 1.79M | 25.7k | 1.61M | **3.42M** | 37.1 | 23.0 | 2,893.3 | 0.3402 |
|  | bandit | 1.13M | 20.7k | 0.995M | **2.14M** (-37%) | 34.6 | NA | NA | 0.3865 |
|  | progressive | 1.27M | 25.9k | 1.16M | **2.46M** (-28%) | 30.5 | 21.9 | 2,579.5 | **0.3879** |
| LongMemEval claudekimi | default | 176.2k | 30.4k | 3.50M | **3.71M** | 49.8 | 27.6 | 4,286.3 | 0.5300 |
|  | bandit | 209.5k | 18.7k | 2.32M | **2.55M** (-31%) | 38.2 | 23.8 | 3,526.7 | **0.5450** |
|  | progressive | 177.7k | 28.4k | 3.16M | **3.37M** (-9%) | 47.3 | 28.8 | 3,772.8 | 0.5200 |
| LongMemEval codex54 | default | 1.51M | 25.3k | 1.39M | **2.93M** | 33.9 | 22.5 | 2,487.0 | 0.5075 |
|  | bandit | 1.13M | 24.6k | 1.03M | **2.18M** (-26%) | 34.8 | 24.7 | 2,879.6 | 0.4725 |
|  | progressive | 1.38M | 23.9k | 1.26M | **2.66M** (-9%) | 32.2 | 22.5 | 2,800.0 | **0.5275** |

The two NA rows (LoCoMo claudekimi progressive, LoCoMo codex54 bandit) are
docs-only entries: their best retained run was rolled up into `PIPELINE.md`
but the artifact directory is no longer on disk, so per-iteration
`metrics.json` cannot be re-aggregated. PIPELINE.md only stores seven
aggregate fields for those rows (`input/iter`, `output/iter`,
`cache reads/iter`, `total/iter`, `tools/iter`, `files/iter`, `dur/iter`),
which is why `read_files/propose` and `read_lines/propose` are unavailable.
Note the docs-only `unique_files/propose` value reported in
`docs/experiment_detail.md` (15.1 for LoCoMo kimi progressive, 18.5 for
LoCoMo codex bandit) uses a different definition than the artifact-derived
rows: PIPELINE's `files/iter` counts workspace files touched per iter,
while artifact rows count distinct files read via the Read tool. The
two columns are not directly comparable, so we omit `unique_files/propose`
from the summary table.

### Headline observations

1. **Default is always the most expensive policy on `total/propose`** in this
   four-cell view. Savings vs default range from -9% (LongMemEval, both
   proposers, progressive) to -46% (LoCoMo claudekimi, progressive).
2. **The cheapest policy outscores default on test in three of four cells**.
   LoCoMo claudekimi has progressive winning both cost and test
   (0.3423 -> 0.3734); LoCoMo codex54 has bandit cheapest at test 0.3865
   vs default 0.3402 (with progressive marginally higher at 0.3879);
   LongMemEval claudekimi has bandit winning both (0.5300 -> 0.5450).
   Only LongMemEval codex54 inverts: bandit is cheapest but its test
   0.4725 is below default 0.5075, while progressive 0.5275 is the
   higher-scoring option.
3. **Bandit is the most reliable cost-saver**: -23% to -37% vs default
   across all four cells. Progressive saves 28-46% on LoCoMo cells and only
   9% on LongMemEval cells; its state machine is benchmark-sensitive.

### Why a longer initial prompt produces lower total tokens

Initial prompt sizes (claudekimi LoCoMo, `prompt.md` median bytes across the
retained training runs):

| policy | prompt.md median | rel to default |
|---|---:|---:|
| default | 6.68 KB | 1.00x |
| progressive | 8.05 KB | 1.20x |
| bandit | 9.97 KB | 1.49x |

Both bandit and progressive carry a longer initial prompt than default
(bandit ~50% larger, progressive ~20% larger) and yet both produce lower
per-propose totals. The reason is that per-propose total decomposes as

```
total ~= initial_prompt * turns * cache_amplification
       + sum_t (per-turn intermediate output + tool output)
       + final output
```

The longer prompt is a fixed cost paid once per propose. The reference
content it carries (best/worst iteration digests for bandit, state-machine
direction for progressive) substitutes for a much larger amount of
exploration cost in the variable sum_t term. Cross-cell decomposition of
the bandit-vs-default delta on the kimi side, where the per-turn prefix
re-read makes the effect easiest to read:

| dimension | LoCoMo bandit delta | LongMemEval bandit delta |
|---|---:|---:|
| input/propose | +4% | +19% |
| output/propose | -21% | -38% |
| tools/propose | -10% | -23% |
| read_lines/propose | -18% | -18% |
| cache_read/propose | -25% | -34% |
| **total/propose** | **-23%** | **-31%** |

Per-turn fresh input grows on LongMemEval (+19% from the +1k initial-prompt
token increase being re-counted across ~38 turns) and stays nearly flat on
LoCoMo. The dominant savings come from output (-21 to -38%), tool turns
(-10 to -23%), and cache_read (-25 to -34%). cache_read declines because
both the per-turn prefix and the turn count fall: the proposer no longer
enumerates exploratory thinking, so per-turn intermediate output is shorter,
and fewer turns are needed because the policy already named the directions
to try.

For codex54 the same direction holds but the accounting differs: codex's
`input_tokens` includes the cached portion (OpenAI Responses convention),
so input and cache_read move together. LoCoMo codex54 bandit shows
input -37% / cache -38%; LongMemEval codex54 bandit shows input -25% /
cache -26%. Codex saves through a smaller final-turn cumulative prefix
(shorter conversation tail) rather than fewer accumulated turns.

A back-of-envelope reconciliation for LoCoMo claudekimi bandit: the longer
initial prompt adds ~1k tokens, re-read across 41.8 turns contributes
~+42k accumulated cache_read. Tool turns drop by 4.8 (46.6 to 41.8), and
each turn's prefix averages ~70k, removing ~336k cache_read. Per-turn
intermediate output also shrinks because the proposer no longer enumerates
options, removing roughly another ~400k cache_read. Net: +42k - 336k - 400k
~ -694k, about -21% of the 3.25M default cache_read. This is in line with
the measured -25% cache_read drop and the headline -23% total drop.

### SWE-bench exception: longer tool trajectories reverse the cost pattern

The memory-cell explanation above does not automatically transfer to
SWE-bench mini. For SWE-bench, the relevant claim is narrower: the proposer
has longer interactive debugging trajectories, so the repeated cache-read of
Read observations is amplified. "Longer" here means more tool/read turns and
larger proposer transcripts, not more wall-clock time or substantially more
source lines read.

Raw claudekimi proposer artifacts re-aggregated from the retained local runs
and the `/helios-storage/...` run directories show the domain shift:

| domain slice | runs | total/iter | cache/iter | tools/iter | Read calls/iter | read lines/iter | stream MB/iter | prompt KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | 4 | 3.53M | 3.35M | 47.8 | 27.8 | 4,096 | 1.00 | 8.0 |
| LongMemEval | 3 | 3.18M | 2.97M | 45.5 | 26.8 | 3,854 | 1.01 | 8.4 |
| SWE-bench mini | 4 | 3.56M | 3.39M | 59.7 | 34.9 | 3,981 | 1.17 | 9.1 |

SWE-bench mini has about 25% more tool turns than LoCoMo (59.7 vs 47.8)
and 31% more than LongMemEval (59.7 vs 45.5). Read calls rise by a similar
amount, while proposer stream files are 15-17% larger. The average number
of read lines is not materially larger than the memory cells, so the cost
increase is not because SWE simply dumps more source text in one shot. The
cost comes from more rounds of source inspection, trace inspection, and
candidate implementation comparison; those observations then remain in the
conversation prefix and are repeatedly cache-read on later turns.

The same pattern appears inside SWE-bench when comparing policies against
full context:

| SWE policy | total/iter | cache/iter | tools/iter | Read calls/iter | read lines/iter | stream MB/iter | prompt KB |
|---|---:|---:|---:|---:|---:|---:|---:|
| full context default | 3.22M | 3.06M | 56.0 | 31.1 | 3,701 | 1.18 | 6.5 |
| progressive | 3.78M | 3.61M | 61.0 | 33.4 | 4,198 | 1.24 | 6.8 |
| bandit fixed-source | 3.51M | 3.35M | 56.6 | 35.1 | 3,831 | 1.05 | 11.4 |
| bandit v4 | 3.73M | 3.56M | 65.1 | 40.1 | 4,195 | 1.19 | 11.6 |

Unlike the memory cells, adaptive policies do not shorten SWE trajectories.
Progressive adds 5.0 tool turns over full context; bandit v4 adds 9.1 tool
turns and 9.0 Read calls. This is enough to offset the intended savings from
curated context, because each additional turn re-reads the accumulated
debugging transcript.

Read-line buckets explain what the extra turns are doing. Full context reads
38.2% of its lines from current Mini-SWE-agent source and 5.4% from reference
Mini-SWE source. Progressive shifts to 36.3% current source and 11.9%
reference source. Fixed-source bandit shifts further to 51.8% current source
and 15.7% reference source, while bandit v4 spends 58.3% on current source.
Thus the extra cache is not generic summary reading; it is concentrated in
the Mini-SWE control loop and prior candidate implementations.

Representative proposer outputs connect those reads to behavior. Full-context
iter002 reads summaries, the iter001 trace, `swebench.py`, `DefaultAgent`,
and `swebench.yaml`, then proposes `stack_trace_context`: parse file paths
and line numbers from the issue and inject those snippets. Full-context
iter010 reads iter009 diagnostics, retrieval outputs, `swebench.py`,
`DefaultAgent`, and the benchmark runner, then proposes `test_oracle_context`:
find tests, run `pytest`, and inject traceback/source snippets. Progressive
iter016 reads the candidate table, prior diagnostics, candidate source, trace
slices, and `DefaultAgent`, then proposes
`final_fallback_traceback_retrieval_v1`, explicitly combining the earlier
traceback-aware retrieval direction with final-fallback canonicalization.
Bandit iter005 and iter013 read hot summaries, current source, previous
candidate source, evaluation summaries, trace slices, `DefaultAgent`,
`InteractiveAgent`, and `swebench.yaml`, then propose feedback/recovery and
impact-aware agent changes.

The resulting interpretation is different from the memory domains. On memory
tasks, curation mostly substitutes for exploratory turns. On SWE-bench mini,
curation gives the proposer actionable debugging targets inside a multi-file
agent control loop, and the proposer responds by doing more mechanism
diagnosis. That is why default can read fewer tokens on SWE while bandit v4
and progressive read more: their additional context increases the amount of
specific source/trace comparison work rather than eliminating the local scan.

### Per-turn cost decomposition: where the SWE inversion lives

The mem vs SWE divergence becomes precise when total cache is factored as
`tools_per_iter × cache_per_turn` and the bandit / default+direction cells
are compared inside each domain (claudekimi, run-mean of per-iter means
across the runs cited in `docs/experiment_detail.md`):

| cell | tools/iter Δ | cache/turn Δ | read_lines/turn Δ | total cache Δ |
|---|---:|---:|---:|---:|
| LongMemEval (mem) | 50.2 → 38.6 (**−23%**) | 72.4k → 52.0k (**−28%**) | flat | (1−0.23)×(1−0.28) = 0.55, so **−45%** |
| SWE-bench mini | 63.5 → 73.3 (**+16%**) | 73.0k → 66.5k (−9%) | 52.2 → 65.6 (**+26%**) | (1+0.16)×(1−0.09) = 1.06, so **+6%** |

`cache/turn` is `cache_read_input_tokens / tools_count`, i.e. the average
cache consumed per LLM call inside a propose. `total cache` per iter
satisfies `total ≈ tools × cache_per_turn`, so the relative change in the
total decomposes multiplicatively into the two factors above. The
"complete" change is just the product of the two; we report it that way
to make explicit which factor each domain rides on.

Both domains see a small per-turn cache reduction from the longer bandit
prompt — the pointer replaces some of the proposer's own option
enumeration, so its intermediate output per turn shortens (cache/turn −28%
on LME, −9% on SWE). What flips the sign is the tool-turn factor: turns
*drop* 23% on LME (pointer collapses exploration into prescription on a
~10-file scaffold) but *rise* 16% on SWE, and each Read pulls back 26%
more lines (longer debugging-target inspections like `DefaultAgent`,
`swebench.py`, `trace_slices/`). The pointer mechanism is the same; the
trajectory it induces is collapse-style on a small candidate codebase
(memgpt) and amplification-style on a multi-file agent control loop.

This decomposition is consistent with `Within ref_iter: a policy-agnostic
attention ceiling` below: the proposer's per-touched-dir depth in
`reference_iterations/iter_M/` stays at ~1.7–2.6 files in both domains,
so the cross-domain difference does not come from how deeply ref iters
are inspected. It comes from how much *current source* and *trace slice*
each Read pulls back per turn, which on SWE is dominated by a handful of
large agent-loop files (`DefaultAgent`, `InteractiveAgent`,
`mini_swe_agent.py`) that simply have more lines than any memgpt file.

### Progressive vs bandit: state machine fragility on larger codebases

Progressive's mechanism is more aggressive than bandit: bandit surfaces a
menu of best-k and worst-k digests and the proposer picks a direction;
progressive's state machine prescribes the direction directly. The
state-machine constraint translates into fewer tool turns *only when the
candidate codebase is small enough that direct prescription replaces a
localization scan*:

| cell | progressive tools/propose | bandit tools/propose | default tools/propose |
|---|---:|---:|---:|
| LoCoMo claudekimi | 35.2 | 41.8 | 46.6 |
| LoCoMo codex54 | 30.5 | 34.6 | 37.1 |
| LongMemEval claudekimi | 47.3 | 38.2 | 49.8 |
| LongMemEval codex54 | 32.2 | 34.8 | 33.9 |

On LoCoMo (smaller candidate code) progressive's state machine succeeds:
tool turns drop 18-24% below default and below bandit. On LongMemEval
(larger candidate code) progressive's tool count returns to near-default
(47.3 vs 49.8 on kimi); the state machine cannot constrain enough to skip
the localization scan, and bandit's anchor-based hints become more
effective. This is consistent with the divergence in the per-cell summary
table above: progressive saves 28-46% on LoCoMo and only 9% on LongMemEval,
while bandit's saving stays in the -23% to -37% band across all four cells.

The practical heuristic: progressive scales with the codebase size for
which its state machine can substitute for exploration. For larger
codebases, anchor-based hints (bandit) are more robust because they tell
the proposer "look here" instead of "do this".

### Headline cost-vs-score per cell

| cell | cheapest policy | savings vs default | best-test policy |
|---|---|---:|---|
| LoCoMo claudekimi | progressive (1.86M) | -46% | progressive (0.3734) |
| LoCoMo codex54 | bandit (2.14M) | -37% | progressive (0.3879) |
| LongMemEval claudekimi | bandit (2.55M) | -31% | bandit (0.5450) |
| LongMemEval codex54 | bandit (2.18M) | -26% | progressive (0.5275) |

Default is consistently the most expensive policy in this slice. Bandit is
the universal cost-saver. Progressive saves 28-46% on LoCoMo and 9% on
LongMemEval, which is consistent with the state-machine-fragility finding
above; progressive should be paired with smaller candidate codebases, or
its state machine widened for larger ones.

## Budget-Conditioned Proposer Behavior

This section uses artifact-level `agent/tool_access.json` traces from the
8 claudekimi memory runs in `docs/experiment_detail.md` (5 bandit + 3
progressive, 240 iterations total). Codex54 is excluded because its tool
traces do not preserve the same Read-tool structure (all calls are bucketed
under `Shell`), and `docs/experiment_detail.md` docs-only rows cannot be
reconstructed per iteration. The earlier `force=low` / `force=high`
ablation runs are also excluded because they are not in
`docs/experiment_detail.md`.

Auto-budget behavior by actual budget tier:

| policy | budget | iters | tools/propose | reads/propose | lines/propose | unique files/propose | source reads/propose | summary reads/propose | ref reads/propose | ref read share | ref lines/propose | best reads/propose |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bandit | low |  5 | 37.40 | 19.60 | 2,176 | 16.00 | 12.00 | 5.60 |  0.00 |  0.0% |   0.0 | 0.00 |
| bandit | medium | 74 | 41.65 | 23.14 | 3,225 | 17.70 |  8.86 | 5.93 |  7.23 | 31.2% | 426.6 | 6.21 |
| bandit | high | 71 | 43.77 | 26.27 | 4,233 | 20.48 |  8.37 | 5.37 | 12.30 | 46.8% | 703.2 | 9.25 |
| progressive | low | 24 | 40.25 | 22.88 | 3,179 | 18.29 |  9.25 | 5.50 |  6.96 | 30.4% | 826.2 | NA |
| progressive | medium |  8 | 59.50 | 27.25 | 3,442 | 19.62 | 12.00 | 4.88 |  9.00 | 33.0% | 683.1 | NA |
| progressive | high | 58 | 50.26 | 27.52 | 3,925 | 19.97 |  9.28 | 6.14 | 11.95 | 43.4% | 960.8 | NA |

`best reads/propose` for bandit is averaged only over iterations whose
policy state had a non-empty `best_iterations` list (otherwise the metric is
not defined). Progressive does not surface `best_iterations` directly to the
proposer, so the column is NA.

The key pattern is that budget mostly changes *reference-history exposure*,
not source-code reading. Source reads stay in a 8-12 / propose band across
all (policy, budget) cells, with no clear monotone trend. What changes is
the reference-iteration channel: bandit moves from 0.00 ref reads/propose
at low (small n=5 sample, but matches the 0.00 force=low result from the
earlier ablation) to 7.23 at medium and 12.30 at high; progressive moves
from 6.96 at low to 9.00 at medium and 11.95 at high. The line volume
shows the same pattern. High budget is therefore not merely "more files";
it is specifically more iteration-history reading.

For bandit, this also directly increases reads of policy-labelled best
iterations: best-iteration reads/propose are 0.00 at low (no best refs
surfaced), 6.21 at medium, and 9.25 at high. This supports the
mechanism-level interpretation that the policy's best-iteration pointers
are an attention prior; the next section
(`Reference-Iteration Read Distribution`) shows the same effect at
slot-normalized resolution.

The bandit `low` cell here only has 5 iterations because the bandit policy
naturally schedules low almost exclusively in the warm-up phase, and most
runs in the experiment_detail set then escalate. The qualitative claim
that low budget effectively removes the reference-iteration channel was
also confirmed by the earlier force=low ablation runs (60 iters, 0.02
ref reads/propose), which are no longer in `docs/experiment_detail.md`
but are kept in `PIPELINE.md` as supporting evidence.

The practical interpretation is budget-dependent:

- low budget is a source/summaries mode; it is cheap and good for local
  candidate edits, but for bandit it largely removes historical iteration
  evidence;
- medium budget is the first tier where bandit best-iteration hints become
  behaviorally active;
- high budget mainly buys more reference-history reading, and especially
  more reads of policy-labelled best iterations.

### Force-Budget Ablation

The auto-budget table above lets the policy choose its tier each iter, so
the bandit `low` row in particular has only n=5 (the policy almost
immediately escalates). To get a clean tier comparison we use the six
claudekimi force-budget runs cataloged in
`docs/experiment_detail.md` (`Ablation: Force-Budget Runs`). These pin
every iter to a single budget so behavioral signatures are not confounded
with the natural escalation schedule. Per-iter means across 60 iters per
(policy, forced-tier) cell:

| policy | forced | iters | tools/propose | reads/propose | lines/propose | unique files/propose | source reads | summary reads | ref reads | ref read share | ref lines | best reads/propose |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bandit | high | 60 | 47.50 | 26.33 | 4,041 | 20.03 |  9.90 | 5.43 | **9.58** | **36.4%** | 724.9 | **7.84** |
| bandit | low  | 60 | 37.08 | 20.52 | 2,361 | 15.65 | 10.93 | 7.93 | **0.02** | **0.1%** | 0.0 | **0.00** |
| progressive | low | 60 | 43.67 | 24.35 | 3,638 | 17.80 |  8.77 | 6.95 | 7.42 | 30.5% | 652.4 | NA |

Two clean ablation findings:

- **bandit force=low almost completely removes the reference-iteration
  channel** (0.02 ref reads / propose, 0 best-iter reads), while
  force=high opens it (9.58 / 7.84). The 0.0 ref_lines confirms the
  reads aren't even partial — bandit at low is a pure source/summaries
  proposer with no history exposure. This is the controlled-ablation
  evidence that the policy's best-iteration pointers, not just budget
  tier, are what drives the iteration-history reading we measure in the
  auto-budget table.
- **progressive at force=low still does ref reading** (7.42 ref reads,
  30.5% share) — unlike bandit, progressive surfaces a small reference
  set even at the lowest tier (the state-machine direction is encoded
  through reference iters). So "low budget = no history" is a bandit
  property, not a policy-agnostic one.

This also reconciles with `docs/PIPELINE.md` results: LongMemEval bandit
force=low scores train 0.6300 (the highest train passrate on
LongMemEval claudekimi bandit), so removing the ref channel does *not*
destroy bandit on LongMemEval; the file-prior alone via UCB hot-files is
sufficient for that benchmark. On LoCoMo the force=low bandit lands at
train 0.3875, modestly below force=high's 0.4000 — the ref channel is
more useful for LoCoMo than for LongMemEval. The Pareto trade is
benchmark-conditional, not absolute.

## Workspace Pool Inflates with Budget, Reads Do Not

The "After Iteration 5" view counts breakthroughs but not how much the
proposer has to wade through to find them. This section asks a separate
mechanistic question: when budget escalates from `low` to `high`, does the
agent actually read more files? The answer is that the workspace file pool
grows nearly an order of magnitude, but per-iteration unique-file reads stay
roughly flat. The implication is that adaptive-policy gains come from
*directing* the agent's attention, not from giving it a bigger pile to look
at.

Aggregating `agent/metrics.json` across all bandit and progressive runs
explicitly cited at the artifact level in `docs/experiment_detail.md`
(10 bandit + 7 progressive runs, mixed claudekimi and codex54,
510 proposer iterations total). Because metrics.json fields are populated
for both proposer agents, this slice is not restricted to claudekimi the
way the prior "Budget-Conditioned Proposer Behavior" section is.

| policy | budget | iters | workspace files (mean) | unique files read | read_file_calls | read_lines | duration_s |
|---|---|---:|---:|---:|---:|---:|---:|
| bandit | low    |  10 |  1,222 | 15.9 | 20.7 | 2,314 | 781 |
| bandit | medium | 138 |  6,191 | 17.9 | 23.7 | 2,975 | 923 |
| bandit | high   | 152 | 21,648 | 19.2 | 25.3 | 3,472 | 917 |
| progressive | low    |  52 |  3,170 | 17.5 | 22.3 | 2,847 | 615 |
| progressive | medium |  18 |  6,102 | 20.2 | 26.1 | 3,230 | 743 |
| progressive | high   | 140 | 21,790 | 18.7 | 25.3 | 3,204 | 709 |

The workspace file count is a proxy for what the agent could reach.
Escalating from `low` to `high` inflates the pool ~17.7x for bandit and
~6.9x for progressive. The agent's `unique_files_read` does grow, but
roughly an order of magnitude less: bandit 15.9 → 19.2 (+21%),
progressive 17.5 → 18.7 (+7%, with a peak at 20.2 in medium). Read
lines grow modestly (bandit +50%, progressive +13%) and per-iter wall
time grows 17% for bandit and 15% for progressive.

The deeper structure behind the +21% / +7% net is *substitution within a
roughly fixed read budget*. Re-bucketing reads by target type across the
LongMemEval+LoCoMo claudekimi runs cited in `docs/experiment_detail.md`
(bandit_v3_banditfix r1+r2, bandit_v4 r1, progressive autobudget r1+r2;
240 propose iters total):

| policy | budget | iters | source unique | summary unique | ref_iter unique | total unique |
|---|---|---:|---:|---:|---:|---:|
| bandit | low |  6 | 8.17 | 5.67 | 0.00 | 15.83 |
| bandit | medium | 82 | 5.71 | 4.97 | 5.53 | 17.85 |
| bandit | high | 92 | 5.42 | 4.62 | **8.68** | 20.30 |
| progressive | low | 24 | 7.50 | 4.92 | 4.31 | 19.05 |
| progressive | high | 58 | 5.86 | 4.82 | 7.74 | 20.34 |

Source-side reads drop ~30% (8.17 → 5.42 for bandit) as ref_iter reads
climb from 0 to ~8.7, while summary reads stay flat (~4.5–5.7 across all
cells). The +21–38% net unique growth happens *on top of* a near
one-for-one substitution: the proposer has a self-imposed ~17–21
files/iter budget and shifts its share rather than stacking new reads
onto the source-scanning baseline.

This is the structural confirmation behind the `Reference-Iteration Read
Distribution` finding below: the agent is not surveying the bigger pool and
picking more sources; it follows the curated menu the policy surfaces (hot
paths, best-iter pointers), and exposes that menu by *redirecting* the
existing read budget. The pool growth therefore mostly buys I/O cost
without proportional behavioral change.

### Within ref_iter: a policy-agnostic attention ceiling

The ref_iter unique growth decomposes into "more dirs touched" rather
than "deeper into the same dir", and the same shape appears under all
three policies once we pick the axis along which the proposer's
exposed-dir count grows. For bandit and progressive that axis is
**budget tier**; for default+direction (which is fixed-high but exposes
*all prior iter dirs*) it is **iter index**. Bucketing every Read
landing on `reference_iterations/iter_M/` by M, then aggregating per
propose iter (claudekimi runs in `docs/experiment_detail.md`):

**Adaptive policies, by budget tier** (bandit_v3_banditfix r1+r2 +
bandit_v4 r1, progressive autobudget r1+r2):

| benchmark | policy | budget | iters | dirs/iter | files / touched_dir | total ref unique |
|---|---|---|---:|---:|---:|---:|
| LongMemEval | bandit | low | 3 | 0.00 | — | 0.00 |
| LongMemEval | bandit | medium | 50 | 2.30 | 2.54 | 5.12 |
| LongMemEval | bandit | high | 37 | **4.11** (+79%) | **2.15** (−15%) | 8.43 |
| LongMemEval | progressive | low | 10 | 1.50 | **3.61** | 5.40 |
| LongMemEval | progressive | medium | 2 | 1.00 | **4.50** | 4.50 |
| LongMemEval | progressive | high | 18 | **3.33** | **2.47** (−45% vs medium) | 7.17 |
| LoCoMo | bandit | low | 3 | 0.00 | — | 0.00 |
| LoCoMo | bandit | medium | 32 | 3.34 | 2.05 | 6.12 |
| LoCoMo | bandit | high | 55 | **5.20** (+56%) | **1.84** (−10%) | 9.04 |
| LoCoMo | progressive | low | 14 | 1.43 | 2.46 | 3.21 |
| LoCoMo | progressive | medium | 6 | 3.83 | 1.94 | 7.33 |
| LoCoMo | progressive | high | 40 | **4.45** | **2.02** | 8.15 |

**default+direction, by iter range** (LME 015454+152524, LoCoMo
015441+154556 — fixed-high schedule but `avail dirs` grows from 2 at
iter 5 to 27 at iter 30):

| benchmark | iter range | n | avail dirs | dirs touched | files / touched_dir | total ref unique | touch_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| LongMemEval | 01-05 | 10 | 2.0 | 0.90 | **5.38** | 4.50 | 45% |
| LongMemEval | 06-10 | 10 | 7.0 | 2.00 | 3.22 | 5.60 | 29% |
| LongMemEval | 11-15 | 10 | 12.0 | 2.70 | 2.40 | 6.20 | 23% |
| LongMemEval | 21-25 | 10 | 22.0 | 3.70 | 2.35 | 7.50 | 17% |
| LongMemEval | 26-30 | 10 | 27.0 | **3.50** | **2.48** | 7.40 | **13%** |
| LoCoMo | 01-05 | 10 | 2.0 | 1.70 | **3.04** | 4.50 | 85% |
| LoCoMo | 06-10 | 10 | 7.0 | 2.20 | 3.16 | 6.10 | 31% |
| LoCoMo | 11-15 | 10 | 12.0 | 3.80 | 2.38 | 8.50 | 32% |
| LoCoMo | 21-25 | 10 | 22.0 | 3.70 | 2.25 | 7.70 | 17% |
| LoCoMo | 26-30 | 10 | 27.0 | **4.70** | **2.26** | 10.50 | **17%** |

Three observations across the two tables:

1. **Every (policy, axis-direction) pair shows dirs↑ / files-per-dir↓
   when the exposed-dir count grows.** bandit medium→high (LME +79% /
   −15%, LoCoMo +56% / −10%); progressive medium→high (LME +233% /
   −45%, LoCoMo +16% / +4%); default+direction iter01-05 → iter26-30
   (LME +289% / −54%, LoCoMo +176% / −26%). The default magnitude is
   *larger*, not smaller, than the adaptive policies — it is the
   strongest evidence the effect is not policy-induced.

2. **Two universal levels emerge: a `dirs touched` ceiling around 3–5
   per iter, and a `files / touched_dir` floor around ~2.** No cell
   sustains more than ~5 dirs touched even when 27 dirs are exposed;
   no high-exposure cell drops below ~2 files per touched dir. The
   touch_rate column on default+direction makes this explicit:
   exposed dirs grow 14× from iter 01-05 to 26-30 while touched dirs
   grow only 4× (LME) / 2.8× (LoCoMo); the proposer self-throttles to
   the ceiling regardless of the supply.

3. **Below the ceiling the proposer reverts to narrow-deep**:
   default+direction iter 01-05 has 0.9–1.7 dirs × 3.0–5.4
   files/touched_dir, mirroring progressive low/medium (1.0–1.5 dirs ×
   3.6–4.5 files/touched_dir). When the supply of exposed ref dirs is
   below the ~3-dir ceiling, every available dir tends to get inspected
   to multi-file depth (typically `diff.patch`, 1–3 files under
   `source_snapshot/`, occasionally a `trace_slice`). This narrow-deep
   regime is policy-agnostic: it appears whenever exposed-dir count is
   small, whether because budget is low (progressive low/medium) or
   because the run is still in early iters (default+direction iter
   01-05).

The mechanism-level reading is that the proposer carries a
**policy-agnostic attention budget for ref iters**: roughly 3–5 dirs at
~2 files each (≈6–10 ref reads per iter at saturation), which it fills
narrow-deep when exposed dirs are scarce and broadens-but-thins when
exposed dirs exceed the ceiling. What policy choice changes is *which*
dirs enter that pool, not how big the pool is. For bandit, the
best-tagged dirs concentrate 14–22× more reads/slot than unmarked refs
(see `Reference-Iteration Read Distribution`). For default+direction,
the same pool is filled from a recency prior (recent-3 = 2.24
reads/slot vs early = 0.04). For progressive, the pool is anchored on
the state-machine-prescribed best/worst pair, which forces narrow-deep
behavior at low/medium budget and a broader fill at high. The
attention ceiling and depth floor are properties of the proposer; the
policy only routes attention into them.

### Quality vs budget: Pareto trade-off, not a free lunch

Per-budget mean passrate of *successfully-evaluated* candidates, restricted
to the 17 auto-budget runs in `docs/experiment_detail.md`:

| policy | low n | low pass | medium n | medium pass | high n | high pass |
|---|---:|---:|---:|---:|---:|---:|
| bandit | 9 | 0.255 | 121 | 0.357 | 134 | **0.378** |
| progressive | 50 | **0.369** | 17 | 0.346 | 127 | 0.355 |

The two policies show different but mutually consistent shapes. **Bandit's
mean passrate is monotonic in budget** (low < medium < high) — more
context, higher mean candidate quality, exactly the cost-quality
trade-off you would expect when budget is the only knob being changed.
**Progressive's low tier has the highest mean passrate** (0.369) but its
high tier (0.355) is only slightly behind, with medium between — the
state machine constrains the search direction enough that smaller
budgets are not losing much candidate quality.

The bandit `low` cell is small (n=9, almost entirely warm-up
iterations), so the 0.255 mean should be read as a weak signal rather
than a stable estimate; the medium and high cells are the ones to trust
for budget comparison. Either way, the headline framing is
**"bandit can pay more budget for marginally higher mean candidate
quality"** and **"progressive's quality is roughly tier-insensitive at
similar absolute level"**. Neither is a free lunch; both are
points on a Pareto curve where the user picks an operating point.

Where `high` does pay off is the long tail: the train-best candidate per
run lands at `high` in 9/10 bandit runs and 5/7 progressive runs (the
other 2 progressive bests are at `low`, both LongMemEval r1 / r2 from the
LongMemEval claudekimi cell). After base-rate adjustment — `high`
occupies 50.8% of evaluated iters for bandit and 65.5% for progressive —
the relative over-representation of `high` for the run-best candidate is
**1.77x for bandit** and only **1.09x for progressive**. Bandit clearly
prefers high for its train-best; progressive's preference is essentially
no preference once you control for tier exposure.

Iter-position controls clarify the timing. Improvement rate (fraction of
iters whose evaluated passrate strictly exceeded the prior running best),
bucketed by iter range and budget:

| policy | iter range | low | medium | high |
|---|---|---:|---:|---:|
| bandit | 01-10 | 30.0% (3/10) | 21.3% (16/75) | 20.0% (3/15) |
| bandit | 11-20 |            - | 0.0% (0/36) | 14.1% (9/64) |
| bandit | 21-30 |            - | 3.7% (1/27) |  9.6% (7/73) |
| progressive | 01-10 | 35.0% (14/40) | 12.5% (1/8)  |  4.5% (1/22) |
| progressive | 11-20 | 22.2% (2/9)   |  0.0% (0/7)  |  9.3% (5/54) |
| progressive | 21-30 |  0.0% (0/3)   |  0.0% (0/3)  |  4.7% (3/64) |

Two patterns stand out:

- **Early iters at `low` are the most productive single setting**:
  progressive's 35% improvement rate at iter01-10 / `low` and bandit's
  30% are the highest cells in their respective rows. This is consistent
  with the warm-up phase finding cheap improvements before the candidate
  approaches the easy ceiling.
- **Late iters (11-30) are dominated by `high`**: `medium` and `low`
  produce almost no improvements after iter 10 (0/36 + 1/27 for bandit
  medium; 0/9 + 0/3 for progressive medium; 0/3 for progressive late
  low), while `high` keeps a 4.7-14.1% per-iter improvement rate. This
  is the role where `high` is irreplaceable — late-stage stagnation
  rescue.

### Why this matters: focus is the real lever

Putting `After Iteration 5`, this section, and `Reference-Iteration Read
Distribution` together:

1. Budget controls how many reference-iter directories are physically
   exposed in the workspace (low ~1 to high ~20 dirs), which inflates the
   workspace pool ~17.7x for bandit and ~6.9x for progressive.
2. The agent does not respond to that inflation. Unique reads per iter stay
   at ~18-20 across all budgets and policies.
3. What the agent does read is overwhelmingly steered by surfaced labels.
   In the next section, bandit-marked best-iter dirs receive 14-22x more
   Read calls per available slot than unmarked refs; without policy hints,
   the proposer falls back to a strong recency prior.
4. Therefore the marginal quality benefit from `high` (the late-iter
   rescue) comes from *which* iters are pointed to as best/worst, not from
   piling more iter dirs into the workspace.

Two design implications follow:

- A `high`-budget cap on `reference_iterations` (e.g., 8-10 instead of full
  history) should preserve the late-stage rescue role of `high` while
  cutting workspace I/O roughly 2x. The change is low-risk because the
  agent was already not reading the extra dirs. v4's ref-iter selection
  rewrite is in this direction (best-3 + worst, capped at 5 for medium and
  3 for low, full history at high); a `high` cap would close the loop.
- `medium` is the weakest standalone tier in this sample: 1/63 mid- and
  late-iter improvements for bandit (only 1 improvement in 36+27 medium
  iters from iter 11 onward), 0/10 for progressive. Its main remaining
  role appears to be transitional. A binary "stagnated -> high, otherwise
  -> low" policy would lose little observable optimization value in the
  retained runs while removing a tier.

## Reference-Iteration Read Distribution

The adaptive policies do not always read fewer files than `default+direction`
(see "Token Trend Context" above), but they redistribute *which* iteration
directories the proposer attends to. This section quantifies that shift using
the per-iteration `tool_access.json` traces from the kimi proposer runs on
LongMemEval and LoCoMo. Codex5.4 runs are excluded because tool_uses are not
logged for that proposer.

Methodology. For every iteration we count, per Read tool call, which
`reference_iterations/iter_NNN/` directory it lands in (if any). Each call also
exposes a set of *available* `iter_NNN` directories under
`workspace/reference_iterations/`. We then bucket each available directory by
how the bandit policy labelled it **at that specific iteration**
(`bandit_policy.best_iterations`, `bandit_policy.worst_iteration`, other
listed `reference_iterations`, or unlisted) and report **reads per available
slot** across the run. The slot normalization removes the confound of
"how many iter dirs were exposed". Note: `best_iterations` is per-iteration
state — at iteration N the policy can only point to iters whose evaluation
finished before N, so labels evolve along the run. Slots and reads are
bucketed against the policy state used in *that* iteration's prompt.

### Bandit best-iter hints concentrate attention strongly

Three LongMemEval kimi bandit runs cited in `docs/experiment_detail.md`,
restricted to iterations whose policy included a non-empty `best_iterations`
list:

| run | iters w/ best | bucket | slots | reads | reads/slot | lines/slot |
|---|---:|---|---:|---:|---:|---:|
| bandit_v3_banditfix r1 | 27 | best-iter dirs | 63 | 231 | **3.67** | **258.9** |
|  |  | worst-iter dir | 24 | 3 | 0.12 | 0.0 |
|  |  | other ref dirs | 131 | 22 | 0.17 | 4.8 |
| bandit_v3_banditfix r2 | 29 | best-iter dirs | 74 | 200 | **2.70** | **180.1** |
|  |  | worst-iter dir | 18 | 4 | 0.22 | 0.0 |
|  |  | other ref dirs | 166 | 21 | 0.13 | 10.5 |
| bandit_v4 r1 | 29 | best-iter dirs | 71 | 245 | **3.45** | **133.9** |
|  |  | worst-iter dir | 18 | 8 | 0.44 | 0.0 |
|  |  | other ref dirs | 183 | 45 | 0.25 | 9.1 |

A `best`-tagged directory receives **14-22x more Read calls per available
slot** than an unmarked reference directory across all three runs, and
**15-54x more lines per slot**. The `worst` directory is read at the same
rate as or below a random reference directory (0.12-0.44 reads/slot), and
zero lines in all three runs — the proposer selectively trusts the `best`
hint and effectively ignores `worst`.

### Without best/worst hints, attention defaults to recency

For `default+direction` the policy exposes *all* prior iter dirs without any
best/worst label. Bucketing the same reads-per-slot metric by recency of the
exposed iter, recomputed against the three default+direction runs in
`docs/experiment_detail.md`:

| run | recent (last 3) | middle | early (first 3) |
|---|---:|---:|---:|
| LME default+direction (015454) | 2.24 | 0.31 | 0.04 |
| LME default+direction (152524) | 2.23 | 0.68 | 0.01 |
| LoCoMo default+direction (015441) | 2.93 | 0.39 | 0.09 |

Early iterations are read 30-220x less often than the most recent three.
With no policy hint, the proposer falls back to a strong recency prior.

### Bandit pulls attention back into early and middle iters

The same recency bucketing applied to LongMemEval bandit_v3_banditfix r1
versus the matched LME default+direction (015454):

| bucket | bandit (banditfix r1) reads/slot | default+direction (015454) reads/slot |
|---|---:|---:|
| recent (last 3) | 2.27 | 2.24 |
| middle | **0.71** | 0.31 |
| early (first 3) | **0.18** | 0.04 |

Recent-iter coverage is essentially unchanged, but middle iters receive ~2.3x
more reads per slot under bandit, and early iters ~4.5x more. The bandit's
`best_iterations` list pulls the proposer back to older iterations that the
recency prior would otherwise skip.

### Progressive shares the labelled-prior mechanism

The bandit-vs-default analysis above relies on the bandit-specific
`bandit_policy.best_iterations` JSON state field. Progressive does not write
that field — it embeds the best/worst tag directly in the prompt text at
`low` and `medium` budget tiers (`proposer_prompt.py:77`–`91`) — so the
slot-normalised reads/slot statistic is bandit-only as written. The
substantive question of whether progressive's labelling also redirects
proposer attention is answerable through a simpler aggregation that does
not depend on the JSON state field: for every Read tool call landing on
`reference_iterations/iter_M/`, was iter_M the top-1 (or top-3) by
candidate passrate at the moment iter_N's prompt was constructed?

The post-hoc rank pool is restricted to `iter < N` so the bucketing only
uses information that was already in `candidate_score_table.json` at
prompt-build time. The "recent-3" bucket (the three most recent iters
strictly less than N) is reported alongside as the recency-baseline
contrast.

| benchmark | policy | runs | total ref Reads | % in top-1 | % in top-3 | % in recent-3 | % in other |
|---|---|---|---:|---:|---:|---:|---:|
| LongMemEval | default+direction | r1+r2 | 430 | 52.1% | 82.3% | 60.2% | 3.7% |
| LongMemEval | progressive       | r1+r2 | 421 | **77.7%** | 87.6% | 55.6% | 5.2% |
| LoCoMo      | default+direction | r1+r2 | 485 | 31.1% | 66.6% | 59.4% | 8.7% |
| LoCoMo      | progressive       | r1+r2 | 422 | **53.8%** | 70.4% | 36.3% | 15.6% |

The top-3 and recent-3 buckets overlap by construction; "other" is reads on
iters that are in neither bucket.

The headline finding is that progressive's reads land on the current top-1
iter at substantially higher share than default's:

- LongMemEval: 77.7% vs 52.1% (**+25.6 pp** absolute, +49% relative).
- LoCoMo: 53.8% vs 31.1% (**+22.7 pp** absolute, +73% relative).

This is the same labelled-prior effect that the 14-22x reads/slot statistic
quantifies for bandit, only measured through a bucket scheme that does not
require the bandit JSON state. Progressive is particularly informative as a
control because its `best_iterations` choice is just `_best_iterations` /
`_worst_iteration` over current passrate (no UCB), so the redirection
cannot be attributed to a sophisticated estimator — the labelling itself,
even when chosen by simple rank, is what moves the proposer's reads.

LoCoMo additionally shows a **−23.1 pp drop in recent-3 share** under
progressive (59.4% → 36.3%). On LoCoMo, progressive does not just add
attention onto top-1 — it actively pulls reads off the recency tail. This
matches the bandit observation that labelled hints can override the
recency prior. LongMemEval shows a smaller −4.6 pp recency drop, consistent
with LongMemEval progressive runs spending more iterations at the high
budget tier (where progressive surfaces no labels and therefore behaves
like default+direction on the ref-iter channel).

Methodology details:
- Source: `proposer_calls/iter_NNN/agent/tool_access.json`, one entry per
  Read tool call. Targets are extracted from each call's `file_path` via the
  regex `reference_iterations/iter_(\d+)/`.
- Rank pool: `candidate_score_table.json`, restricted to rows with iteration
  in `[1, N-1]`. Per-iter row passrate is taken as `max` over candidates of
  that iter (matches what the proposer would see in the score table).
- Sample: two `default+direction` runs and two `progressive` autobudget
  runs per benchmark, kimi proposer only (codex54 does not log per-call
  `tool_uses` for Read). Run IDs are listed in
  `scripts/measure_top1_attention.py`.
- Caveat: passrate values are read from `candidate_score_table.json` after
  the run finished, which is the final value for each iter row. Because
  evaluation runs to completion before the next iter's prompt is built,
  the prompt-time pool is approximated well; either way the relative
  comparison between progressive and default uses the same rule on both
  sides, so any approximation bias cancels.

### Workspace cardinality changes inspection type, not amount

The two preceding subsections (bandit best-tag and progressive top-$k$) measure
the effect of *labelling* on read targets. A separate question is whether
*workspace cardinality* — how many ref-iter dirs the policy physically copies
into the proposer's sandbox — also changes proposer behavior in a way that
explains the cardinality-bounded baselines (Random@3, Recent@3) outperforming
default-family on cost. The best comparison case is `default` (workspace
exposes all $\sim$$13$–$15$ prior iter dirs) vs `recent3` (workspace exposes
only the most recent $3$), both with the codex54 proposer.

Codex54 wraps every proposer operation as a single Shell tool call, so the
per-Read bucketing used above for kimi (slot-normalised reads/slot, top-$k$
share) is unavailable. As a coarse proxy, we instead extract every
file-path token from each Shell command's text matching the regex
`(summaries|reference_iterations|source_snapshot)/[\w./_-]+` and assign each
path to one of seven exclusive buckets (script:
`scripts/measure_path_buckets.py`). The buckets are: aggregated summaries,
other summaries, ref-iter `diff.patch`/`diff_digest.md`, ref-iter
`trace_slices/`, ref-iter `source_snapshot/`, ref-iter other (e.g.
`pending_eval.json`), and the current clean source under
`source_snapshot/candidate/`.

Sample: 3 `default` runs and 3 `recent3` runs per benchmark on LoCoMo, plus
2+3 on LongMemEval (LongMemEval default has only two retained 30-iter runs;
the third crashed early). Aggregating across all 11 runs:

| cell | summ-agg | summ-oth | **ref-iter-diff** | **ref-iter-trace** | ref-iter-source | ref-iter-other | clean-source |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo / default (n=3)      | 18.0% | 0.0% | **5.1%**  | **7.0%**  | 3.9% | 9.5%  | 56.4% |
| LoCoMo / recent3 (n=3)      | 18.5% | 0.0% | **8.4%**  | **4.3%**  | 4.0% | 11.2% | 53.7% |
| LongMemEval / default (n=2) | 16.3% | 0.0% | **4.4%**  | **7.1%**  | 4.4% | 12.0% | 55.8% |
| LongMemEval / recent3 (n=3) | 19.4% | 0.0% | **8.8%**  | **4.1%**  | 3.6% | 9.9%  | 54.2% |

Three observations:

1. **`recent3` reads `diff.patch` / `diff_digest.md` at $\sim$$2\times$ the rate
   of `default`** ($+1.6\times$ on LoCoMo, $+2.0\times$ on LongMemEval). Both
   benchmarks point the same way, and every individual `recent3` run sits above
   every individual `default` run on this column ($7.2$–$9.2\%$ vs $4.0$–$6.5\%$).
2. **`default` reads `trace_slices/` at $\sim$$1.7\times$ the rate of `recent3`**
   ($+1.6\times$ on LoCoMo, $+1.7\times$ on LongMemEval). Same direction across
   benchmarks and across individual runs ($6.2$–$7.9\%$ vs $3.5$–$5.7\%$).
3. **Aggregated `summaries/` reading is policy-invariant** (LoCoMo $18.0\%$ vs
   $18.5\%$; LongMemEval $16.3\%$ vs $19.4\%$, with `recent3` *higher* on
   LongMemEval). The intuition that "larger pool $\Rightarrow$ proposer falls
   back to broad summary scanning" is **not** supported by these traces.

Total ref-iter inspection share (`diff` + `trace` + `source`) is also
policy-invariant: LoCoMo default $16.0\%$ vs recent3 $16.7\%$; LongMemEval
default $15.9\%$ vs recent3 $16.5\%$. The proposer spends roughly the same
share of its file-path references on past-iter contents in both cases. What
changes is the *type* of artifact within that fixed budget:

- **`recent3` favors code-level diffs** ("what did the last 3 iterations
  actually edit, line-by-line").
- **`default` favors example traces** ("what did the candidates from past
  iterations produce on sample inputs").

A plausible mechanistic reading: with only $3$ ref dirs available, reading all
$3$ `diff.patch` files is cheap and information-dense — a complete picture of
recent mechanism evolution in a few hundred lines. With $13$–$15$ dirs
available, reading all diffs is expensive, so the proposer retreats to
trace-level sampling instead. The "deep inspection" budget is roughly
constant; what shifts is which artifact type is the rational cheapest signal.

Caveats. Sample is codex54-only: kimi tool calls are typed (Read / Grep / etc.)
rather than uniformly Shell, so the path-extraction proxy is unnecessary
there but the per-Read bucketing in the prior subsection cannot be
computed for codex54 in return. LongMemEval `default` is $n=2$ because the
third 30-iter run crashed early. Path tokens are extracted by regex from
command text, so any path embedded inside a heredoc or python literal is
counted; we did not attempt to deduplicate identical paths within a single
command.

### Takeaways

- The adaptive policies' main effect on read behavior is *redistribution*, not
  reduction: total Read calls per iteration are similar to `default+direction`
  (~21–23 reads/iter) but the targets shift.
- The `best_iterations` slot is the primary lever; bandit-marked best dirs
  receive **14-22x** more reads per slot than unmarked refs across the three
  retained LongMemEval bandit runs.
- The same labelled-prior effect appears in **progressive** under a
  bucket-by-rank measurement that does not need the bandit JSON state:
  progressive's reads land on the current top-1 iter at +25.6 pp share on
  LongMemEval and +22.7 pp share on LoCoMo relative to `default+direction`.
  Because progressive's labels are just rank-by-passrate (no UCB), this
  shows the labelling alone — not the estimator — is what redirects
  proposer attention.
- Workspace cardinality changes the *type* of reference-iteration artifact the
  proposer inspects, not the *amount*. Total ref-iter inspection share is
  policy-invariant (~16% across LoCoMo and LongMemEval, both `default` and
  `recent3`), but `recent3` directs that budget at `diff.patch` (~2× over
  `default`) while `default` directs it at `trace_slices` (~1.7× over
  `recent3`). The "broad summary scanning under large pools" hypothesis is
  not supported: aggregated-summary reading is essentially policy-invariant
  (16–19%) on both benchmarks.
- The `worst_iteration` slot has near-zero observable effect on the proposer's
  read distribution; if a future revision keeps it, the worst summary should be
  surfaced more directly in the prompt rather than only as a referenced dir.
- Without any best/worst hints, the proposer defaults to strong recency bias
  and almost never re-reads early iterations; this is the failure mode that
  the best-iter pointer corrects.
