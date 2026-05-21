# Terminal-Bench comparison — Codex (Azure) proposer × Daytona

This is the runbook for the Terminal-Bench 2.0 default-vs-organized comparison
experiment. The proposer is the Codex CLI authenticated against **Azure
OpenAI**; every Harbor rollout runs in a remote **Daytona** cloud sandbox.

Launch script: [`scripts/launch_terminus_codex_azure_daytona.sh`](../scripts/launch_terminus_codex_azure_daytona.sh)

## What the experiment does

- **20 evolution iterations**, **20 tasks/iter** (the first 20 of the 30-task
  TB2 "hard" split), **2 attempts/task** — all 40 trials per iteration run in
  parallel on Daytona.
- **Two arms**, identical except for the RunStore tool surface. Both arms
  expose the same upstream-2 summary files (`evolution_summary.jsonl` +
  `best_candidates.json`), so the only variable across arms is the tools:
  - `default` — `--selection-policy default` (skill mode `default`, no
    RunStore tools).
  - `organized` — `--organized --selection-policy default` (skill mode
    `organized-summaries`: generates `state.md`, registers RunStore tools).
- Both arms **share one primed iter-0 KIRA baseline** (reused via
  `--baseline-dir`), so the seed frontier is evaluated once, not twice.
- After the 20 iterations each arm **automatically evaluates its best train
  Pareto-frontier candidate on the full TB2 test split** — the 59 held-out
  tasks (89 TB2 tasks minus the 30 hard tasks).

## Prerequisites

| Need | How |
| --- | --- |
| This repo | `git clone` + `git checkout codex/organized-trace-modification-report` |
| Python env | Install the project (`pip install -e .` or the project's usual setup) |
| Codex CLI | `npm install -g @openai/codex` — verify with `codex --version` |
| Azure OpenAI | A deployment of a GPT-5 codex/reasoning model; copy its endpoint + key |
| Daytona | An account at <https://app.daytona.io> with a concurrent-sandbox quota **≥ 40** |
| DeepSeek-V4 | An endpoint you can reach for the Terminus solver |

## Setup

### 1. Codex CLI → Azure OpenAI

Copy the template config into place and edit it:

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

- `DAYTONA_API_KEY` — managed-cloud key from app.daytona.io.
- `AZURE_OPENAI_API_KEY` — the key for the Azure resource in your `config.toml`.
- `DEEPSEEK_API_KEY` — key for your DeepSeek-V4 solver endpoint.

`.env` is git-ignored — **never commit it**. This repo is public.

## Run

```bash
# launch under nohup/tmux — the baseline prime is a long foreground step
nohup bash scripts/launch_terminus_codex_azure_daytona.sh > /dev/null 2>&1 &
```

The script first primes the shared baseline (foreground), then launches both
arms detached. It prints a `status:` file path — `tail -f` it, and the
per-run logs under `logs/`, to follow progress.

### Pointing the solver at your DeepSeek-V4 endpoint

The solver defaults to the official DeepSeek API
(`openai/deepseek-v4-pro` @ `https://api.deepseek.com/v1`, key
`DEEPSEEK_API_KEY`). Override all three to use your own endpoint:

```bash
SOLVER_MODEL=openai/your-deepseek-v4 \
SOLVER_BASE_URL=https://your-endpoint/v1 \
SOLVER_API_KEY_ENV=YOUR_KEY_VAR \
  bash scripts/launch_terminus_codex_azure_daytona.sh
```

`SOLVER_API_KEY_ENV` names the variable in `.env` that holds the key.

### Knobs (env vars)

| Var | Default | Meaning |
| --- | --- | --- |
| `ARMS` | `default,organized` | Which arms to launch |
| `ITERATIONS` | `20` | Evolution iterations per arm |
| `LIMIT` | `20` | Tasks per iteration (first N of the 30 hard tasks) |
| `ROLLOUT_TRIALS` | `2` | Attempts per task |
| `ROLLOUT_CONCURRENCY` | `40` | Max concurrent Daytona sandboxes — **keep ≤ your quota** |
| `CODEX_MODEL` | `gpt-5.1-codex` | Your Azure deployment name |
| `CODEX_REASONING_EFFORT` | `high` | Codex proposer reasoning effort |
| `CODEX_HOME` | _(unset → `~/.codex`)_ | Dir holding the Azure `config.toml` |
| `SOLVER_MODEL` / `SOLVER_BASE_URL` / `SOLVER_API_KEY_ENV` | DeepSeek API | Terminus solver endpoint |
| `BASELINE_DIR` | `runs/...baseline...` | Reuse a primed baseline across launches |
| `DRY_RUN` | `0` | `1` = wire-check only, no Harbor/Daytona calls |

## Daytona quota

With `--terminus-env daytona`, concurrency is bounded by your account's
**concurrent-sandbox quota**, not local CPU/RAM. The script defaults
`ROLLOUT_CONCURRENCY=40` (20 tasks × 2 trials, fully parallel). If your quota
is lower, set `ROLLOUT_CONCURRENCY` to it — trials past the quota simply queue.
If trials fail to start, check the quota before assuming a reasoning failure.
See [`TERMINUS_DAYTONA.md`](TERMINUS_DAYTONA.md) for Daytona-specific detail.

## Results

Per arm, under `runs/terminus_kira_codex_azure_daytona_<arm>_train_<ts>/`:

- `best_candidates.json` — the train Pareto frontier.
- `evolution_summary.jsonl` — the cumulative per-iteration event log.
- `test_frontier/` — the automatic full-test-split evaluation
  (`test_results.json`, `test_pareto_frontier.json`, `test_frontier_summary.json`).

## Known limitation — `trace_similar` under Codex + Azure

The `organized` arm registers RunStore tools. One of them, `trace_similar`,
embeds historical diffs against an OpenAI-compatible endpoint. The other
RunStore tools query the local SQLite store and need no embedding endpoint.

If you have an OpenAI-compatible embedding endpoint, set `OPENAI_API_KEY` /
`OPENAI_BASE_URL` / `DIFF_EMBEDDING_MODEL` in `.env` (see `.env.example`).
Otherwise `trace_similar` degrades gracefully and the rest of the organized
tool surface still works.
