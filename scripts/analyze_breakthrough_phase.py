#!/usr/bin/env python3
"""De-confounded breakthrough-phase analysis (paper Finding C1).

The headline C1 claim -- "wide context earns its cost mainly as late-run
rescue" -- was originally measured on *auto-budget* runs, where the budget tier
is chosen by the policy's state machine, so the tier is correlated with the run
phase (the machine escalates to `high` precisely when it stalls). That makes
"tier" and "phase" confounded.

This script re-asks the question on *force-budget* runs, where the tier is
pinned for the whole run, so within a run there is no tier->phase coupling.
We compare, between the pinned-`high` runs and the pinned-`low` runs, the
breakthrough rate as a function of iteration range. If C1 holds even
de-confounded, pinned-`high` should out-breakthrough pinned-`low` in the late
ranges (and roughly tie early); if the original effect was just the auto-budget
confound, the two pinned conditions should look alike.

A *breakthrough* = an iteration whose evaluated candidate strictly improved the
best-so-far passrate (ties broken by average_score), iter 0 (seed) excluded.

Run dirs resolve local-first then from the helios archive. Bootstrap CIs are
over runs (resampling whole runs with replacement).

Usage::

    python scripts/analyze_breakthrough_phase.py
    python scripts/analyze_breakthrough_phase.py --extra-high RUN --extra-low RUN
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

RUN_ROOTS = [
    Path("/data/home/yuhan/MemoMemo/runs"),
    Path("/helios-storage/helios4-data/yuhan/MemoMemo/runs"),
]

# Force-budget runs: tier pinned for the whole run. (Names from
# Optimizer1/docs/experiment_detail.md "Ablation: Force-Budget Runs" -- the
# `budgethigh` / `budgetlow` series.)
FORCE_HIGH = [
    "locomo_memgpt_claudekimi_bandit_v3_budgethigh_docker_iter30_train80_w16_20260503_200314",
    "longmemeval_memgpt_claudekimi_bandit_v3_budgethigh_docker_iter30_train100_w16_20260503_200349",
]
FORCE_LOW = [
    "locomo_memgpt_claudekimi_bandit_v3_budgetlow_docker_iter30_train80_w16_20260502_170954",
    "longmemeval_memgpt_claudekimi_bandit_v3_budgetlow_docker_iter30_train100_w16_20260502_170958",
    "locomo_memgpt_claudekimi_progressive_budgetlow_docker_iter30_train80_20260502_170952",
    "longmemeval_memgpt_claudekimi_progressive_budgetlow_docker_iter30_train100_20260502_170956",
]

# Iteration ranges (inclusive) over a 30-round run.
RANGES = [("01-05", 1, 5), ("06-10", 6, 10), ("11-15", 11, 15),
          ("16-20", 16, 20), ("21-30", 21, 30), ("post-warmup (6-30)", 6, 30)]


def _intact(d: Path) -> bool:
    es = d / "evolution_summary.jsonl"
    if es.exists() and es.stat().st_size > 0:
        return True
    cr = d / "candidate_results"
    return cr.is_dir() and any(cr.iterdir())


def resolve_run(name: str) -> Path | None:
    cands = [r / name for r in RUN_ROOTS if (r / name).is_dir()]
    if not cands:
        return None
    for c in cands:
        if _intact(c):
            return c
    return cands[0]


def load_per_iter(run_dir: Path) -> dict[int, tuple[float | None, float | None]]:
    """iteration -> (passrate, average_score) of that iter's evaluated candidate."""
    out: dict[int, tuple[float | None, float | None]] = {}
    es = run_dir / "evolution_summary.jsonl"
    if es.exists():
        for ln in es.read_text(errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            it = d.get("iteration")
            c = d.get("candidate") or {}
            if it is None or not c:
                continue
            pr = c.get("passrate")
            if it not in out or pr is not None:
                out[it] = (pr, c.get("average_score"))
        if out:
            return out
    idx = run_dir / "iteration_index.json"
    if idx.exists():
        for entry in json.loads(idx.read_text()):
            it = entry.get("iteration")
            best = None
            for cid in entry.get("candidate_ids") or []:
                cr = run_dir / "candidate_results" / f"{cid}.json"
                if cr.exists():
                    try:
                        cc = json.loads(cr.read_text()).get("candidate") or {}
                    except json.JSONDecodeError:
                        continue
                    if best is None or (cc.get("passrate") or -1) > (best[0] or -1):
                        best = (cc.get("passrate"), cc.get("average_score"))
            if best:
                out[it] = best
    return out


def breakthrough_iters(per_iter: dict[int, tuple]) -> set[int]:
    bts: set[int] = set()
    best_pr, best_av = float("-inf"), float("-inf")
    for it in sorted(per_iter):
        pr, av = per_iter[it]
        if pr is None:
            continue
        avv = av if av is not None else float("-inf")
        if pr > best_pr + 1e-12 or (abs(pr - best_pr) <= 1e-12 and avv > best_av + 1e-12):
            if it > 0:
                bts.add(it)
            best_pr, best_av = max(best_pr, pr), max(best_av, avv)
    return bts


def per_run_range_counts(run_dir: Path) -> dict[str, tuple[int, int]]:
    """range -> (breakthroughs in range, evaluated iters in range)."""
    per_iter = load_per_iter(run_dir)
    bts = breakthrough_iters(per_iter)
    evaluated = {it for it, (pr, _) in per_iter.items() if pr is not None and it > 0}
    out: dict[str, tuple[int, int]] = {}
    for label, lo, hi in RANGES:
        rng = set(range(lo, hi + 1))
        out[label] = (len(bts & rng), len(evaluated & rng))
    return out


def bootstrap_rate_ci(per_run: list[dict[str, tuple[int, int]]], label: str,
                      n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap over runs: pooled (sum bt / sum iters) rate, 90% CI."""
    rng = random.Random(seed)
    pooled = []
    k = len(per_run)
    for _ in range(n):
        sample = [per_run[rng.randrange(k)] for _ in range(k)]
        bt = sum(s[label][0] for s in sample)
        it = sum(s[label][1] for s in sample)
        pooled.append(bt / it if it else 0.0)
    pooled.sort()
    bt0 = sum(s[label][0] for s in per_run)
    it0 = sum(s[label][1] for s in per_run)
    return (bt0 / it0 if it0 else 0.0, pooled[int(0.05 * n)], pooled[int(0.95 * n)])


def analyse(high_names: list[str], low_names: list[str]) -> None:
    def collect(names):
        out = []
        for nm in names:
            rd = resolve_run(nm)
            if rd is None:
                print(f"!! {nm}: not found", file=sys.stderr)
                continue
            out.append((nm, rd, per_run_range_counts(rd)))
            print(f".. {rd.name}: {out[-1][2]}", file=sys.stderr)
        return out

    high = collect(high_names)
    low = collect(low_names)
    if not high or not low:
        print("need at least one run on each side", file=sys.stderr)
        return

    print("\n" + "=" * 96)
    print("DE-CONFOUNDED C1: breakthrough rate by iter range, pinned-HIGH vs pinned-LOW context")
    print("=" * 96)
    print(f"  pinned-HIGH runs (n={len(high)}): " + ", ".join(n for n, _, _ in high))
    print(f"  pinned-LOW  runs (n={len(low)}): " + ", ".join(n for n, _, _ in low))
    print()
    print(f"  {'iter range':<22}  {'HIGH bt/iters  rate [90% CI]':<34}  {'LOW bt/iters  rate [90% CI]':<34}  HIGH-LOW")
    hi_counts = [c for _, _, c in high]
    lo_counts = [c for _, _, c in low]
    for label, _, _ in RANGES:
        hb = sum(c[label][0] for c in hi_counts); hi = sum(c[label][1] for c in hi_counts)
        lb = sum(c[label][0] for c in lo_counts); li = sum(c[label][1] for c in lo_counts)
        hr, hlo, hhi = bootstrap_rate_ci(hi_counts, label)
        lr, llo, lhi = bootstrap_rate_ci(lo_counts, label)
        diff = hr - lr
        print(f"  {label:<22}  {hb:>2}/{hi:<3} {hr*100:5.1f}% [{hlo*100:4.1f},{hhi*100:5.1f}]      "
              f"{lb:>2}/{li:<3} {lr*100:5.1f}% [{llo*100:4.1f},{lhi*100:5.1f}]      {diff*100:+5.1f}pp")
    print()
    print("  reading: if C1 survives de-confounding, HIGH should beat LOW in the late ranges")
    print("  (16-20, 21-30) and roughly tie in 01-05; if not, the original C1 effect was the")
    print("  auto-budget tier<->phase confound.  n is small -- CIs are wide; read directionally.")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra-high", action="append", default=[])
    ap.add_argument("--extra-low", action="append", default=[])
    args = ap.parse_args(argv)
    analyse(FORCE_HIGH + args.extra_high, FORCE_LOW + args.extra_low)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
