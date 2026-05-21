# Running the Terminal-Bench 2.0 target on Daytona

The `--terminus` benchmark scores each candidate by running Harbor on
`terminal-bench@2.0`. Harbor executes every trial inside a sandbox; the
sandbox backend is selected with `--terminus-env`. The default is the local
`docker` environment. This document covers the `daytona` backend, which runs
each trial in a remote [Daytona](https://www.daytona.io/) cloud sandbox.

Use Daytona when the local docker box is disk- or CPU-constrained, or when you
want to offload rollout load off the optimizer host. The proposer still runs
locally (or in its own proposer docker) — only the *rollout* sandbox moves to
the cloud.

## Prerequisites

1. A Daytona account and an API key (managed cloud: <https://app.daytona.io>).
2. Add the key to the repo `.env` (it is git-ignored; the launch scripts
   `source .env`):

   ```
   DAYTONA_API_KEY=dtn_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

   `DAYTONA_API_URL` defaults to `https://app.daytona.io/api` and only needs
   setting for a self-hosted Daytona server. `DAYTONA_TARGET` (region) is
   optional.

Harbor's own preflight raises only once the first trial starts. The optimizer
adds an earlier check (`preflight_terminus_env`): a `daytona` run with no
`DAYTONA_API_KEY` fails immediately, before the seed frontier or the proposer
container spin up. `--dry-run` skips the check, since it never touches Harbor.

## Running

### Launch script (recommended)

`scripts/launch_terminus_daytona.sh` is the Daytona twin of
`launch_terminus_baseline_vs_optimal1.sh`: same algorithm/solver/proposer, but
`--terminus-env daytona`. It refuses to start if `DAYTONA_API_KEY` is unset.

```bash
# baseline + optimal1 arms on the 30-task hard split
bash scripts/launch_terminus_daytona.sh

# smoke first: single task, single arm, dry-run-free end-to-end check
ARMS=baseline SPLIT=warmup ITERATIONS=1 ROLLOUT_CONCURRENCY=1 \
  bash scripts/launch_terminus_daytona.sh
```

Key env knobs (see the script header for the full list): `ARMS`, `SPLIT`,
`ITERATIONS`, `ROLLOUT_TRIALS`, `ROLLOUT_CONCURRENCY`, `ENV_KWARGS`.

### Direct CLI

```bash
python -m optimizer1.cli optimize --terminus \
  --terminus-env daytona \
  --terminus-env-kwargs auto_stop_interval_mins=20,auto_delete_interval_mins=120 \
  --split warmup --iterations 1 \
  ...
```

## `--terminus-env-kwargs`

Harbor environment kwargs are passed as `key=value`, forwarded to Harbor as
`--ek`. The flag is repeatable and also accepts comma-separated values. Harbor
parses values as JSON/Python literals, so `key=20` becomes an int and
`key=false` a bool. Daytona-relevant keys:

| Key | Meaning |
| --- | --- |
| `auto_stop_interval_mins` | Minutes of inactivity before a sandbox auto-stops. `0` = never. |
| `auto_delete_interval_mins` | Minutes after stop before a sandbox is deleted. `0` = delete immediately on stop. |
| `dind_image` | Base image for the Docker-in-Docker sandbox (default `docker:28.3.3-dind`). |
| `dind_snapshot` | Pre-created Daytona snapshot for faster DinD startup. |
| `network_block_all` | Block all sandbox network access (overrides the task's `allow_internet`). |

`launch_terminus_daytona.sh` defaults `ENV_KWARGS` to
`auto_stop_interval_mins=20,auto_delete_interval_mins=120` so that sandboxes
leaked by a crashed or hung Harbor run get reclaimed automatically. A normal
run deletes its sandboxes on stop regardless. Set `ENV_KWARGS=` (empty) to use
pure Harbor defaults.

## Concurrency and quota

With `docker`, concurrency is bounded by local CPU/RAM. With `daytona`, it is
bounded by your account's **concurrent-sandbox quota**. Keep
`--terminus-rollout-concurrency` at or below that quota — the launch script
defaults it to `8` (vs `12` for the docker launcher). If trials fail to start,
check the quota before assuming a reasoning failure.

## Single-container vs Docker-in-Docker

Harbor's Daytona environment auto-detects the topology per task:

- Tasks with only a `Dockerfile` use a **direct** single sandbox.
- Tasks shipping a `docker-compose.yaml` use a **Docker-in-Docker** sandbox
  (`docker:28.3.3-dind`) and run `docker compose` inside it.

Both paths are handled automatically; no configuration is required. DinD
startup is slower — `dind_snapshot` can speed it up if you create a snapshot.

## Recommended bring-up order

1. `SPLIT=warmup` (the single `extract-elf` task), one arm,
   `ROLLOUT_CONCURRENCY=1` — verify the end-to-end Daytona path.
2. The 30-task `hard` split (`SPLIT=train`) at a small concurrency.
3. Raise concurrency toward your quota once the path is stable.
