#!/usr/bin/env python3
"""Reference-iteration reads-per-available-slot, with bootstrap CIs (Finding A3).

The paper's A3 claims -- a `best`-tagged reference directory absorbs 14-22x
more Read calls per available slot than an unmarked reference, and the `worst`
tag is read no more than a random reference; without labels the proposer reads
the 3 most recent iterations 30-220x more per slot than the 3 earliest -- were
reported as raw per-run ranges.  This script re-derives the per-run reads/slot
numbers from the traces and attaches bootstrap CIs over runs, so the headline
can be a number-with-interval instead of a range, and the recency "220x" (a
near-zero denominator) can be replaced by absolute reads/slot.

Two modes:
  * ``bandit``  -- bucket each iter's available ref dirs (from
    ``proposer_calls/iter_N/assignment.json`` -> ``bandit_policy``) into
    best / worst / unlabelled; reads from ``agent/tool_access.json``
    (``files_read`` keys matching ``reference_iterations/iter_M/``).
  * ``recency`` -- bucket the available ref dirs by iter index into
    recent-3 / middle / early-3 (used for the label-free default+direction
    runs).

reads/slot for a bucket in a run = (sum of reads landing in that bucket over
all iters) / (sum of |bucket| over all iters).  Bootstrap is over runs
(resample whole runs with replacement); reported as the across-run mean and a
90% CI.  n is small (3 runs per condition), so CIs are wide -- that is the
honest point.

Run dirs resolve local-first then helios.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

RUN_ROOTS = [
    Path("/data/home/yuhan/MemoMemo/runs"),
    Path("/helios-storage/helios4-data/yuhan/MemoMemo/runs"),
]
REF_ITER_RE = re.compile(r"reference_iterations/iter_0*(\d+)/")

# 3 LongMemEval kimi bandit runs cited in docs/experiment_detail.md (best/worst
# labels live in assignment.json -> bandit_policy).
BANDIT_RUNS = [
    "longmemeval_memgpt_claudekimi_bandit_v3_banditfix_autobudget_docker_iter30_train100_w16_r1_20260505_003416",
    "longmemeval_memgpt_claudekimi_bandit_v3_banditfix_autobudget_docker_iter30_train100_w16_r2_20260505_003416",
    "longmemeval_memgpt_claudekimi_bandit_v4_autobudget_docker_iter30_train100_w16_r1_20260505_040626",
]
# label-free default+direction runs (recency bucketing).
RECENCY_RUNS = [
    "longmemeval_memgpt_claudekimi_default_direction_docker_iter30_train100_20260502_015454",
    "longmemeval_memgpt_claudekimi_default_direction_docker_iter30_train100_20260502_152524",
    "locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_015441",
]


def _intact(d: Path) -> bool:
    es = d / "evolution_summary.jsonl"
    return (es.exists() and es.stat().st_size > 0) or (
        (d / "candidate_results").is_dir() and any((d / "candidate_results").iterdir())
    )


def resolve_run(name: str) -> Path | None:
    cands = [r / name for r in RUN_ROOTS if (r / name).is_dir()]
    if not cands:
        return None
    for c in cands:
        if _intact(c):
            return c
    return cands[0]


def reads_by_target(it_dir: Path) -> dict[int, int]:
    ta = it_dir / "agent" / "tool_access.json"
    out: dict[int, int] = defaultdict(int)
    if not ta.exists():
        return out
    try:
        d = json.loads(ta.read_text())
    except json.JSONDecodeError:
        return out
    # prefer the pre-aggregated files_read; fall back to tool_uses
    fr = d.get("files_read") or {}
    if fr:
        for path, info in fr.items():
            m = REF_ITER_RE.search(path)
            if m:
                out[int(m.group(1))] += int((info or {}).get("reads", 1) or 1)
        return out
    for u in d.get("tool_uses", []):
        if u.get("name") != "Read":
            continue
        path = (u.get("input") or {}).get("file_path", "") or ""
        m = REF_ITER_RE.search(path)
        if m:
            out[int(m.group(1))] += 1
    return out


def per_run_bandit(run_dir: Path) -> dict[str, tuple[int, int]]:
    """bucket -> (total reads landing in bucket, total available slots)."""
    acc: dict[str, list[int]] = {b: [0, 0] for b in ("best", "worst", "unlabelled")}
    pc = run_dir / "proposer_calls"
    for it_dir in sorted(pc.iterdir()):
        m = re.match(r"iter_(\d+)$", it_dir.name)
        if not (it_dir.is_dir() and m):
            continue
        n = int(m.group(1))
        if n == 0:
            continue
        asn_path = it_dir / "assignment.json"
        if not asn_path.exists():
            continue
        try:
            bp = (json.loads(asn_path.read_text()).get("bandit_policy") or {})
        except json.JSONDecodeError:
            continue
        avail = set(int(x) for x in (bp.get("reference_iterations") or []))
        if not avail:
            continue
        best = set(int(x) for x in (bp.get("best_iterations") or [])) & avail
        w = bp.get("worst_iteration")
        worst = ({int(w)} & avail) if w is not None else set()
        unl = avail - best - worst
        rds = reads_by_target(it_dir)
        for name, s in (("best", best), ("worst", worst), ("unlabelled", unl)):
            acc[name][0] += sum(rds.get(i, 0) for i in s)
            acc[name][1] += len(s)
    return {k: (v[0], v[1]) for k, v in acc.items()}


def per_run_recency(run_dir: Path) -> dict[str, tuple[int, int]]:
    acc: dict[str, list[int]] = {b: [0, 0] for b in ("recent3", "middle", "early3")}
    pc = run_dir / "proposer_calls"
    for it_dir in sorted(pc.iterdir()):
        m = re.match(r"iter_(\d+)$", it_dir.name)
        if not (it_dir.is_dir() and m):
            continue
        n = int(m.group(1))
        if n == 0:
            continue
        # available ref dirs: any reference_iterations/iter_M present in the workspace
        wref = it_dir / "workspace" / "reference_iterations"
        if not wref.is_dir():
            # fall back to assignment.json reference_iterations
            asn = it_dir / "assignment.json"
            if asn.exists():
                try:
                    avail = sorted(int(x) for x in (json.loads(asn.read_text()).get("reference_iterations") or []))
                except json.JSONDecodeError:
                    avail = []
            else:
                avail = []
        else:
            avail = sorted(int(re.search(r"iter_0*(\d+)", p.name).group(1))
                           for p in wref.iterdir() if re.match(r"iter_\d+", p.name))
        if len(avail) < 1:
            continue
        recent3 = set(avail[-3:])
        early3 = set(avail[:3])
        middle = set(avail) - recent3 - early3
        rds = reads_by_target(it_dir)
        for name, s in (("recent3", recent3), ("middle", middle), ("early3", early3)):
            acc[name][0] += sum(rds.get(i, 0) for i in s)
            acc[name][1] += len(s)
    return {k: (v[0], v[1]) for k, v in acc.items()}


def bootstrap_ratio_ci(per_run: list[dict[str, tuple[int, int]]], num: str, den: str,
                       n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap (over runs) the ratio (mean reads/slot in `num`) / (... in `den`)."""
    rng = random.Random(seed)
    k = len(per_run)
    vals = []
    for _ in range(n):
        sample = [per_run[rng.randrange(k)] for _ in range(k)]
        rn = sum(s[num][0] for s in sample); sn = sum(s[num][1] for s in sample)
        rd = sum(s[den][0] for s in sample); sd = sum(s[den][1] for s in sample)
        if sn == 0 or sd == 0 or rd == 0:
            continue
        vals.append((rn / sn) / (rd / sd))
    vals.sort()
    rn = sum(s[num][0] for s in per_run); sn = sum(s[num][1] for s in per_run)
    rd = sum(s[den][0] for s in per_run); sd = sum(s[den][1] for s in per_run)
    point = (rn / sn) / (rd / sd) if sn and sd and rd else float("nan")
    if not vals:
        return point, float("nan"), float("nan")
    return point, vals[int(0.05 * len(vals))], vals[int(0.95 * len(vals))]


def bootstrap_mean_ci(per_run: list[dict[str, tuple[int, int]]], bucket: str,
                      n: int = 10000, seed: int = 0) -> tuple[float, float, float, list[float]]:
    rng = random.Random(seed)
    k = len(per_run)
    per_run_rs = [(r[bucket][0] / r[bucket][1]) if r[bucket][1] else 0.0 for r in per_run]
    vals = []
    for _ in range(n):
        s = [per_run_rs[rng.randrange(k)] for _ in range(k)]
        vals.append(sum(s) / len(s))
    vals.sort()
    mean = sum(per_run_rs) / len(per_run_rs)
    return mean, vals[int(0.05 * len(vals))], vals[int(0.95 * len(vals))], per_run_rs


def run_mode(mode: str, names: list[str]) -> None:
    per_run = []
    labels = []
    for nm in names:
        rd = resolve_run(nm)
        if rd is None:
            print(f"!! {nm}: not found", file=sys.stderr)
            continue
        c = per_run_bandit(rd) if mode == "bandit" else per_run_recency(rd)
        per_run.append(c)
        labels.append(rd.name)
        rs = {k: (round(v[0] / v[1], 3) if v[1] else None) for k, v in c.items()}
        print(f".. {rd.name}: reads/slot {rs}  (raw {c})", file=sys.stderr)
    if not per_run:
        return
    buckets = ("best", "worst", "unlabelled") if mode == "bandit" else ("recent3", "middle", "early3")
    print("\n" + "=" * 92)
    print(f"READS PER AVAILABLE SLOT  [{mode}]  (n={len(per_run)} runs; bootstrap-over-runs 90% CI)")
    print("=" * 92)
    for nm in labels:
        print(f"  - {nm}")
    print()
    for b in buckets:
        mean, lo, hi, prs = bootstrap_mean_ci(per_run, b)
        print(f"  {b:<12} reads/slot: mean {mean:6.3f}  [90% CI {lo:6.3f}, {hi:6.3f}]   per-run: {[round(x,3) for x in prs]}")
    print()
    if mode == "bandit":
        r, lo, hi = bootstrap_ratio_ci(per_run, "best", "unlabelled")
        print(f"  RATIO best / unlabelled reads-per-slot:  {r:5.1f}x   [90% CI {lo:4.1f}x, {hi:4.1f}x]")
        rw, lw, hw = bootstrap_ratio_ci(per_run, "worst", "unlabelled")
        print(f"  RATIO worst / unlabelled reads-per-slot: {rw:5.2f}x   [90% CI {lw:4.2f}x, {hw:4.2f}x]   (~1x => worst label ignored)")
    else:
        r, lo, hi = bootstrap_ratio_ci(per_run, "recent3", "early3")
        print(f"  RATIO recent-3 / early-3 reads-per-slot: {r:6.1f}x   [90% CI {lo:5.1f}x, {hi:6.1f}x]   (huge & unstable -- the early-3 denom is ~0)")
        print(f"  -> report ABSOLUTE instead: recent-3 reads/slot mean {bootstrap_mean_ci(per_run,'recent3')[0]:.2f}  vs  early-3 reads/slot mean {bootstrap_mean_ci(per_run,'early3')[0]:.3f}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("bandit", "recency", "both"), default="both")
    ap.add_argument("--bandit-run", action="append", default=[])
    ap.add_argument("--recency-run", action="append", default=[])
    args = ap.parse_args(argv)
    if args.mode in ("bandit", "both"):
        run_mode("bandit", BANDIT_RUNS + args.bandit_run)
    if args.mode in ("recency", "both"):
        run_mode("recency", RECENCY_RUNS + args.recency_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
