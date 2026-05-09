#!/usr/bin/env python3
"""Measure where the proposer's reference-iteration Read calls land.

For every Read tool call into ``reference_iterations/iter_M/`` in a run's
``proposer_calls/iter_N/agent/tool_access.json`` traces, classify the
target ``iter_M`` against the run's ``candidate_score_table.json`` at
prompt-build time:

- top-1: ``iter_M`` is the highest-passrate iter among iters in ``[0, N-1]``
- top-3: ``iter_M`` is among the top-3 by passrate in ``[0, N-1]``
- recent-3: ``iter_M`` is among the three most recent iters strictly < N
- other: reads on iters that are in neither top-3 nor recent-3

The top-3 and recent-3 buckets overlap by construction; "other" is reads on
iters in neither bucket. Top-k uses the per-iter row max passrate (we take
the strongest candidate per iteration row when the table lists multiple
candidates per iter), which approximates what the proposer would see in the
candidate score table at the moment of prompt construction. Run IDs cover
the kimi proposer only because codex54 traces do not preserve per-call
``tool_uses``.

Output: a per-run row plus a benchmark-by-policy aggregate.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path("/data/home/yuhan/MemoMemo/runs")
REF_ITER_RE = re.compile(r"reference_iterations/iter_(\d+)/")

# kimi proposer runs cited in docs/experiment_detail.md
RUNS = {
    "LongMemEval / default+direction r1": "longmemeval_memgpt_claudekimi_default_direction_docker_iter30_train100_20260502_015454",
    "LongMemEval / default+direction r2": "longmemeval_memgpt_claudekimi_default_direction_docker_iter30_train100_20260502_152524",
    "LongMemEval / progressive r1":      "longmemeval_memgpt_claudekimi_progressive_autobudget_docker_iter30_train100_r1_20260504_162844",
    "LongMemEval / progressive r2":      "longmemeval_memgpt_claudekimi_progressive_autobudget_docker_iter30_train100_r2_rerun429_20260504_212541",
    "LoCoMo / default+direction r1":     "locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_015441",
    "LoCoMo / default+direction r2":     "locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_154556",
    "LoCoMo / progressive r1":           "locomo_memgpt_claudekimi_progressive_autobudget_docker_iter30_train80_r1_20260504_162844",
    "LoCoMo / progressive r2":           "locomo_memgpt_claudekimi_progressive_autobudget_docker_iter30_train80_r2_20260504_162844",
}


def load_passrate_per_iter(run_dir: Path) -> dict[int, float]:
    """Return ``iter -> max passrate`` over candidates of that iter."""
    rows = json.load((run_dir / "candidate_score_table.json").open())
    out: dict[int, float] = {}
    for r in rows:
        it = r.get("iteration")
        pr = r.get("passrate") or 0.0
        if it is None:
            continue
        out[it] = max(out.get(it, -1.0), pr)
    return out


def cumulative_top_k(pr_by_iter: dict[int, float], at_iter: int, k: int) -> set[int]:
    pool = [(it, pr) for it, pr in pr_by_iter.items() if it < at_iter]
    pool.sort(key=lambda x: (-x[1], x[0]))
    return {it for it, _ in pool[:k]}


def analyze_run(run_dir: Path) -> dict:
    pr_by_iter = load_passrate_per_iter(run_dir)
    pc_dir = run_dir / "proposer_calls"

    n_total = 0
    bucket: dict[str, int] = defaultdict(int)
    n_iters_with_reads = 0

    for it_dir in sorted(pc_dir.iterdir()):
        m_iter = re.match(r"iter_(\d+)", it_dir.name)
        if not it_dir.is_dir() or not m_iter:
            continue
        iter_n = int(m_iter.group(1))
        if iter_n == 0:
            continue
        ta = it_dir / "agent" / "tool_access.json"
        if not ta.exists():
            continue
        d = json.load(ta.open())

        reads_per_target: dict[int, int] = defaultdict(int)
        for u in d.get("tool_uses", []):
            if u.get("name") != "Read":
                continue
            path = (u.get("input") or {}).get("file_path", "") or ""
            mref = REF_ITER_RE.search(path)
            if mref:
                reads_per_target[int(mref.group(1))] += 1

        if not reads_per_target:
            continue
        n_iters_with_reads += 1

        top1 = cumulative_top_k(pr_by_iter, iter_n, k=1)
        top3 = cumulative_top_k(pr_by_iter, iter_n, k=3)
        recent3 = set(sorted([i for i in pr_by_iter if i < iter_n], reverse=True)[:3])

        for target, cnt in reads_per_target.items():
            n_total += cnt
            if target in top1:
                bucket["top1"] += cnt
            if target in top3:
                bucket["top3"] += cnt
            if target in recent3:
                bucket["recent3"] += cnt
            if target not in top3 and target not in recent3:
                bucket["other"] += cnt

    return {
        "run": run_dir.name,
        "iters_with_ref_reads": n_iters_with_reads,
        "total_ref_reads": n_total,
        "top1": bucket["top1"],
        "top3": bucket["top3"],
        "recent3": bucket["recent3"],
        "other": bucket["other"],
    }


def main() -> None:
    print(f"{'run label':<42} | {'iters':>5} | {'reads':>6} | {'top1%':>6} | {'top3%':>6} | {'recent3%':>9} | {'other%':>6}")
    print("-" * 100)
    summaries: dict[str, dict] = {}
    for label, name in RUNS.items():
        run_dir = RUNS_DIR / name
        if not run_dir.exists():
            print(f"{label}: MISSING {name}")
            continue
        s = analyze_run(run_dir)
        summaries[label] = s
        n = max(s["total_ref_reads"], 1)
        print(
            f"{label:<42} | {s['iters_with_ref_reads']:>5} | {s['total_ref_reads']:>6} | "
            f"{s['top1']/n*100:>5.1f}% | {s['top3']/n*100:>5.1f}% | "
            f"{s['recent3']/n*100:>8.1f}% | {s['other']/n*100:>5.1f}%"
        )

    # Aggregate by (benchmark, policy)
    print("\n--- aggregated ---")
    agg: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for label, s in summaries.items():
        bench, policy = [x.strip() for x in label.split("/")[:2]]
        policy = policy.rsplit(" r", 1)[0].strip()
        for k in ("total_ref_reads", "top1", "top3", "recent3", "other"):
            agg[(bench, policy)][k] += s[k]

    print(f"{'bench':<14} | {'policy':<22} | {'reads':>6} | {'top1%':>6} | {'top3%':>6} | {'recent3%':>9} | {'other%':>6}")
    print("-" * 90)
    for (bench, policy), counts in sorted(agg.items()):
        n = max(counts["total_ref_reads"], 1)
        print(
            f"{bench:<14} | {policy:<22} | {counts['total_ref_reads']:>6} | "
            f"{counts['top1']/n*100:>5.1f}% | {counts['top3']/n*100:>5.1f}% | "
            f"{counts['recent3']/n*100:>8.1f}% | {counts['other']/n*100:>5.1f}%"
        )


if __name__ == "__main__":
    main()
