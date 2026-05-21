# Memory benchmarks — Codex (Azure) proposer × remote eval model

This is the runbook for the **LoCoMo** and **LongMemEval** default-vs-organized
comparison experiments. The proposer is the Codex CLI authenticated against
**Azure OpenAI**; the agent scaffold and (for LongMemEval) the LLM judge run
against **your own remote OpenAI-compatible endpoints**.

Launch scripts:
- LoCoMo — [`scripts/launch_locomo_codex_azure.sh`](../scripts/launch_locomo_codex_azure.sh)
- LongMemEval — [`scripts/launch_longmemeval_codex_azure.sh`](../scripts/launch_longmemeval_codex_azure.sh)

The two experiments share the same setup; the only LongMemEval-specific extra
is the LLM judge endpoint.

## What the experiment does

- **30 evolution iterations** (override with `ITERATIONS`), evaluated on the
  whole train split — 80 LoCoMo questions / 100 LongMemEval questions.
- **Two arms**, identical except for the RunStore tool surface. Both arms
  expose the same upstream-2 summary files (`evolution_summary.jsonl` +
  `best_candidates.json`), so the only variable across arms is the tools:
  - `default` — `--selection-policy default` (skill mode `default`, no
    RunStore tools).
  - `organized` — `--organized --selection-policy default` (skill mode
    `organized-summaries`: generates `state.md`, registers RunStore tools).
- Both arms **share one primed seed frontier** (the `memgpt_source` seed,
  reused via `--baseline-dir`), so the seed eval is paid once, not twice.
- After the iterations each arm **automatically evaluates its best train
  Pareto-frontier candidate on the held-out test split**.

## Prerequisites

| Need | How |
| --- | --- |
| This repo | `git clone` + use the `main` branch |
| Python env | Install the project (`pip install -e .` or the project's usual setup) |
| Codex CLI | `npm install -g @openai/codex` — verify with `codex --version` |
| Azure OpenAI | A deployment of a GPT-5 codex/reasoning model; copy its endpoint + key |
| Eval endpoint | A remote OpenAI-compatible endpoint serving the model the scaffold queries |
| Judge endpoint | (LongMemEval only) a remote OpenAI-compatible endpoint for the LLM judge |

There is **no local model server and no local dataset to stage** — the eval and
judge models are remote endpoints you provide, and the datasets are downloaded
automatically on first run (see *Datasets* below).

## Setup

### 1. Codex CLI → Azure OpenAI

Same as the Terminal-Bench experiment. Copy the template config into place and
edit it:

```bash
mkdir -p ~/.codex
cp docs/codex_config.azure.toml ~/.codex/config.toml
$EDITOR ~/.codex/config.toml
```

In `~/.codex/config.toml` set:

- `base_url` — `https://YOUR_RESOURCE_NAME.openai.azure.com/openai/v1`
  (the trailing `/openai/v1` is required).
- `model` — your Azure **deployment** name.

Azure auth is an API key (set in `.env`, next step) — there is no interactive
`codex login` for Azure, and Entra ID / Azure AD SSO is not supported.

> To keep this separate from a personal ChatGPT-login Codex config, copy the
> template to a dedicated directory instead and run the launcher with
> `CODEX_HOME=/path/to/that/dir`.

### 2. Secrets → `.env`

```bash
cp .env.example .env
$EDITOR .env
```

Fill in:

- `AZURE_OPENAI_API_KEY` — the key for the Azure resource in your `config.toml`.
- `EVAL_BASE_URL` / `EVAL_MODEL` / `EVAL_API_KEY` — your remote endpoint for the
  scaffold's eval model (LoCoMo **and** LongMemEval).
- `JUDGE_BASE_URL` / `JUDGE_MODEL` / `JUDGE_API_KEY` — **LongMemEval only**, your
  remote endpoint for the LLM judge. Skip these if you run with `NO_LLM_JUDGE=1`.

`.env` is git-ignored — **never commit it**. This repo is public.

## Run

```bash
# LoCoMo — launch under nohup/tmux (dataset prepare + baseline prime are
# long foreground steps)
nohup bash scripts/launch_locomo_codex_azure.sh > /dev/null 2>&1 &

# LongMemEval
nohup bash scripts/launch_longmemeval_codex_azure.sh > /dev/null 2>&1 &
```

Each script first downloads the dataset if missing, primes the shared baseline
(foreground), then launches both arms detached. It prints a `status:` file path
— `tail -f` it, and the per-run logs under `logs/`, to follow progress.

## Datasets

`data/` is git-ignored, so a fresh clone has no benchmark data. Each launcher
auto-runs the `prepare` step on first use:

- LoCoMo — downloads `locomo10.json` from the public snap-research mirror.
- LongMemEval — downloads the cleaned `s` variant (~277 MB) from Hugging Face.

Both then write **deterministic** warmup/train/test splits (seed 13). Because
the splits are deterministic, every machine that runs `prepare` with the default
parameters gets the identical split — no need to ship the split files. If you
hand-tuned a split file, copy it to `data/locomo/splits.json` or
`data/longmemeval/splits_s.json` *before* the first run so the launcher's
presence check skips the regenerating `prepare`.

To prepare manually instead (e.g. to pre-download on a fast link):

```bash
python -m optimizer1.cli locomo prepare --allow-download
python -m optimizer1.cli longmemeval prepare --variant s --allow-download
```

## LongMemEval LLM judge

LongMemEval scores each answer with an LLM-as-judge. Point it at your own
OpenAI-compatible endpoint via `JUDGE_BASE_URL` / `JUDGE_MODEL` / `JUDGE_API_KEY`
in `.env`. The judge is used both during the iterations and for the final
test-frontier evaluation.

To skip the LLM judge entirely and use LongMemEval's local token/F1 scorer:

```bash
NO_LLM_JUDGE=1 bash scripts/launch_longmemeval_codex_azure.sh
```

With `NO_LLM_JUDGE=1` the `JUDGE_*` variables are not needed.

## Knobs (env vars)

| Var | Default | Meaning |
| --- | --- | --- |
| `ARMS` | `default,organized` | Which arms to launch |
| `ITERATIONS` | `30` | Evolution iterations per arm |
| `SPLIT` | `train` | Split to optimize on |
| `LIMIT` | `0` | Questions per iteration (`0` = whole split) |
| `EVAL_WORKERS` | `32` | Concurrent eval calls — **the #1 knob to tune to your endpoint** |
| `EVAL_TIMEOUT_S` | `300` | Per-eval-call timeout |
| `SCAFFOLDS` | `memgpt_source` | Seed scaffold |
| `VARIANT` | `s` | LongMemEval variant (`s` / `m` / `oracle`) — LongMemEval only |
| `NO_LLM_JUDGE` | `0` | `1` = local token/F1 scoring — LongMemEval only |
| `CODEX_MODEL` | `gpt-5.1-codex` | Your Azure deployment name |
| `CODEX_REASONING_EFFORT` | `high` | Codex proposer reasoning effort |
| `CODEX_HOME` | _(unset → `~/.codex`)_ | Dir holding the Azure `config.toml` |
| `EVAL_BASE_URL` / `EVAL_MODEL` / `EVAL_API_KEY_ENV` | from `.env` | Scaffold eval endpoint |
| `JUDGE_BASE_URL` / `JUDGE_MODEL` / `JUDGE_API_KEY_ENV` | from `.env` | LLM judge endpoint |
| `BASELINE_DIR` | `runs/...baseline...` | Reuse a primed baseline across launches |
| `DRY_RUN` | `0` | `1` = wire-check only, no eval calls |

`EVAL_API_KEY_ENV` / `JUDGE_API_KEY_ENV` name the `.env` variable that holds the
key (default `EVAL_API_KEY` / `JUDGE_API_KEY`) — set them only if your key lives
under a different variable name.

If your endpoint rate-limits, lower `EVAL_WORKERS`; if it has high throughput,
raise it (the local-vLLM runs in this repo use up to 128). Repeated 429s during
the run usually mean `EVAL_WORKERS` is too high for the endpoint, not a
reasoning failure.

## Results

Per arm, under `runs/<benchmark>_codex_azure_<arm>_train_<ts>/`:

- `best_candidates.json` — the train Pareto frontier.
- `evolution_summary.jsonl` — the cumulative per-iteration event log.
- `optimizer_summary.json` — the final run summary (also the marker the
  launcher uses to confirm the baseline prime finished).
- `test_frontier/` — the automatic held-out test-split evaluation.

## Known limitation — `trace_similar` under Codex + Azure

The `organized` arm registers RunStore tools. One of them, `trace_similar`,
embeds historical diffs against an OpenAI-compatible endpoint. The other
RunStore tools query the local SQLite store and need no embedding endpoint.

If you have an OpenAI-compatible embedding endpoint, set `OPENAI_API_KEY` /
`OPENAI_BASE_URL` / `DIFF_EMBEDDING_MODEL` in `.env` (see `.env.example`).
Otherwise `trace_similar` degrades gracefully and the rest of the organized
tool surface still works.
