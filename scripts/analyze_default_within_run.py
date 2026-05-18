#!/usr/bin/env python3
"""Within-`default`-run behaviour of the coding-agent proposer.

Everything here is measured *inside* the plain `default` loop (fixed patch
base x0, fixed-high context = full revision history exposed every round). No
bandit / progressive / curaii runs are touched -- the only variation used is
the natural one: as a `default` run proceeds, the workspace accumulates one
more ``reference_iterations/iter_N/`` directory each round, so the available
pool grows ~order-of-magnitude over a 30-round run with nothing else changed.

PART A -- "the pool grows, the reading does not (much)".
  For each iteration-range we report, averaged over the `default`-family runs:
  available ref-iter dirs (the pool the loop exposes), and the proposer's
  response -- total tool calls, Read calls, unique files read, read lines, and
  the ref-iter slice specifically (dirs touched, reads, lines) plus the
  *touch rate* = touched / available.

PART B -- "the attention window is on the most recent iterations, not the
  highest-scoring ones".  For each round N in a `default` run we bucket the
  available prior iters by recency (3 most recent) and by quality (3 highest
  passrate in the candidate score table at round N), then report reads per
  available slot for the disjoint buckets recent-only / best-only / both /
  neither.  If recency, not quality, drives attention, recent-only >>
  best-only.

Per-Read targets come from ``proposer_calls/iter_N/agent/tool_access.json``
(``files_read`` for the typed-tool backbone; for the Shell-tool backbone we
extract ``reference_iterations/iter_M`` tokens from the Shell command strings).
Run dirs resolve local-first then from the helios archive.

Usage::

    python scripts/analyze_default_within_run.py            # all default-family groups
    python scripts/analyze_default_within_run.py --group kimi_mem
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

RUN_ROOTS = [
    Path("/data/home/yuhan/MemoMemo/runs"),
    Path("/helios-storage/helios4-data/yuhan/MemoMemo/runs"),
]
REF_ITER_RE = re.compile(r"reference_iterations/iter_0*(\d+)\b")

# `default`-family runs only: `--selection-policy default`, fixed-high context.
# (`+direction` just adds a free-text note; still the Full-history condition.)
GROUPS: dict[str, list[str]] = {
    "kimi_mem": [
        "locomo_memgpt_claudekimi_default_docker_iter30_train80_20260501_204004",
        "locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_015441",
        "locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_154556",
        "longmemeval_memgpt_claudekimi_default_direction_docker_iter30_train100_20260502_015454",
        "longmemeval_memgpt_claudekimi_default_direction_docker_iter30_train100_20260502_152524",
    ],
    "codex_mem": [
        "locomo_memgpt_codex54_default_docker_iter30_train80_20260501_204007",
        "locomo_memgpt_codex54_default_docker_iter30_train80_rerun_20260502_015354",
        "locomo_memgpt_codex54_default_new_docker_iter30_train80_r1_20260506_005632",
        "locomo_memgpt_codex54_default_new_docker_iter30_train80_r2_20260506_005632",
        "locomo_memgpt_codex54_default_new_docker_iter30_train80_r3_20260506_005632",
    ],
    "kimi_swebench": [
        "swebench_miniswe_deepseek_v4_flash_claudekimi_default_direction_iter30_trainfirst30_w10_t900_20260502_015837",
    ],
}

RANGES = [("01-05", 1, 5), ("06-10", 6, 10), ("11-15", 11, 15),
          ("16-20", 16, 20), ("21-30", 21, 30)]


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


def available_ref_iters(it_dir: Path) -> set[int]:
    """Prior-iter dirs the loop placed in this round's workspace."""
    wref = it_dir / "workspace" / "reference_iterations"
    if wref.is_dir():
        out = set()
        for p in wref.iterdir():
            m = re.match(r"iter_0*(\d+)$", p.name)
            if m:
                out.add(int(m.group(1)))
        if out:
            return out
    asn = it_dir / "assignment.json"
    if asn.exists():
        try:
            return set(int(x) for x in (json.loads(asn.read_text()).get("reference_iterations") or []))
        except json.JSONDecodeError:
            pass
    return set()


def reads_by_target(it_dir: Path) -> dict[int, tuple[int, int]]:
    """target iter -> (read calls, read lines) landing in reference_iterations/iter_M/."""
    ta = it_dir / "agent" / "tool_access.json"
    out: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    if not ta.exists():
        return {}
    try:
        d = json.loads(ta.read_text())
    except json.JSONDecodeError:
        return {}
    fr = d.get("files_read") or {}
    if fr:
        for path, info in fr.items():
            m = REF_ITER_RE.search(path)
            if m:
                out[int(m.group(1))][0] += int((info or {}).get("reads", 1) or 1)
                out[int(m.group(1))][1] += int((info or {}).get("lines", 0) or 0)
        return {k: (v[0], v[1]) for k, v in out.items()}
    # Shell-tool backbone: scrape iter tokens from command strings.
    for u in d.get("tool_uses", []):
        inp = u.get("input") or {}
        blob = " ".join(str(v) for v in inp.values())
        for m in REF_ITER_RE.finditer(blob):
            out[int(m.group(1))][0] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def load_metrics(it_dir: Path) -> dict | None:
    m = it_dir / "agent" / "metrics.json"
    if not m.exists():
        m2 = it_dir / "agent" / "attempt_01" / "metrics.json"
        if m2.exists():
            m = m2
        else:
            return None
    try:
        return json.loads(m.read_text())
    except json.JSONDecodeError:
        return None


def passrate_by_iter(run_dir: Path) -> dict[int, float]:
    cst = run_dir / "candidate_score_table.json"
    out: dict[int, float] = {}
    if cst.exists():
        try:
            for r in json.loads(cst.read_text()):
                it = r.get("iteration")
                if it is None:
                    continue
                out[it] = max(out.get(it, -1.0), r.get("passrate") or 0.0)
        except json.JSONDecodeError:
            pass
    return out


# ---------------------------------------------------------------- PART A
def per_run_partA(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    """range -> metric -> list of per-iter values (so we can mean across runs)."""
    out: dict[str, dict[str, list[float]]] = {lbl: defaultdict(list) for lbl, _, _ in RANGES}
    pc = run_dir / "proposer_calls"
    for it_dir in sorted(pc.iterdir()):
        m = re.match(r"iter_(\d+)$", it_dir.name)
        if not (it_dir.is_dir() and m):
            continue
        n = int(m.group(1))
        if n == 0:
            continue
        lbl = next((l for l, lo, hi in RANGES if lo <= n <= hi), None)
        if lbl is None:
            continue
        avail = available_ref_iters(it_dir)
        met = load_metrics(it_dir) or {}
        rds = reads_by_target(it_dir)
        touched = set(rds)
        ref_reads = sum(v[0] for v in rds.values())
        ref_lines = sum(v[1] for v in rds.values())
        rec = out[lbl]
        rec["avail_ref_dirs"].append(len(avail))
        rec["tool_calls"].append(met.get("tool_calls", float("nan")))
        rec["read_calls"].append((met.get("tool_counts") or {}).get("Read", float("nan")))
        rec["unique_files_read"].append(met.get("unique_files_read", float("nan")))
        rec["read_lines"].append(met.get("read_lines", float("nan")))
        rec["duration_s"].append(met.get("duration_s", float("nan")))
        rec["ref_dirs_touched"].append(len(touched))
        rec["ref_reads"].append(ref_reads)
        rec["ref_lines"].append(ref_lines)
        rec["touch_rate"].append(len(touched) / len(avail) if avail else float("nan"))
    return out


def _nanmean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def report_partA(groups: dict[str, list[tuple[str, Path]]]) -> None:
    print("\n" + "=" * 110)
    print("PART A -- within a `default` run: the available pool grows, the proposer's reading does not (much)")
    print("=" * 110)
    cols = ["avail_ref_dirs", "tool_calls", "read_calls", "unique_files_read", "read_lines",
            "ref_dirs_touched", "ref_reads", "ref_lines", "touch_rate", "duration_s"]
    for gname, runs in groups.items():
        if not runs:
            continue
        # aggregate: for each range, pool per-iter values across all runs in the group, then mean
        agg: dict[str, dict[str, list[float]]] = {lbl: defaultdict(list) for lbl, _, _ in RANGES}
        for _, rd in runs:
            pr = per_run_partA(rd)
            for lbl in agg:
                for k, v in pr[lbl].items():
                    agg[lbl][k].extend(v)
        print(f"\n--- group: {gname}  (n_runs={len(runs)}: {', '.join(n for n, _ in runs)})")
        hdr = f"  {'range':<7}"
        for c in cols:
            hdr += f"{c:>18}"
        print(hdr)
        first_avail = None
        last_avail = None
        for lbl, _, _ in RANGES:
            row = f"  {lbl:<7}"
            for c in cols:
                v = _nanmean(agg[lbl][c])
                if c == "touch_rate":
                    row += f"{v*100:>17.1f}%"
                else:
                    row += f"{v:>18.1f}"
            print(row)
            if first_avail is None:
                first_avail = _nanmean(agg[lbl]["avail_ref_dirs"])
            last_avail = _nanmean(agg[lbl]["avail_ref_dirs"])
        # ratio summary 01-05 -> 21-30
        def rat(c):
            a = _nanmean(agg["01-05"][c]); b = _nanmean(agg["21-30"][c])
            return (b / a) if (a and a == a and b == b and a != 0) else float("nan")
        print(f"  >> 01-05 -> 21-30:  avail_ref_dirs x{rat('avail_ref_dirs'):.1f}   "
              f"unique_files_read x{rat('unique_files_read'):.2f}   read_calls x{rat('read_calls'):.2f}   "
              f"ref_dirs_touched x{rat('ref_dirs_touched'):.2f}   ref_reads x{rat('ref_reads'):.2f}   "
              f"touch_rate {_nanmean(agg['01-05']['touch_rate'])*100:.0f}% -> {_nanmean(agg['21-30']['touch_rate'])*100:.0f}%")


# ---------------------------------------------------------------- PART B
def per_run_partB(run_dir: Path, tie_break: str = "recent") -> dict:
    """Per-round bucketing of available prior iters by recency-3 vs top-3-by-passrate.

    tie_break: when passrates tie, prefer the more 'recent' iter into top-3
    ('recent') or the more 'old' one ('old').

    Returns a dict with:
      disjoint[range_label][bucket]     -> [reads, slots]   (bucket in recent_only/best_only/both/neither; range_label includes 'ALL')
      marginal[range_label][m]          -> [reads, slots]   (m in 'in_recent3' / 'in_best3' -- overlapping)
      spread[range_label]               -> list of per-round (max-min) available passrate
    """
    pr_by_iter = passrate_by_iter(run_dir)
    range_labels = ["ALL"] + [l for l, _, _ in RANGES]
    disjoint = {rl: {b: [0, 0] for b in ("recent_only", "best_only", "both", "neither")} for rl in range_labels}
    marginal = {rl: {m: [0, 0] for m in ("in_recent3", "in_best3")} for rl in range_labels}
    spread = {rl: [] for rl in range_labels}
    pc = run_dir / "proposer_calls"
    for it_dir in sorted(pc.iterdir()):
        m = re.match(r"iter_(\d+)$", it_dir.name)
        if not (it_dir.is_dir() and m):
            continue
        n = int(m.group(1))
        if n == 0:
            continue
        rlbl = next((l for l, lo, hi in RANGES if lo <= n <= hi), None)
        targ = ["ALL"] + ([rlbl] if rlbl else [])
        avail = sorted(i for i in available_ref_iters(it_dir) if i < n)
        if len(avail) < 4:
            continue
        recent3 = set(avail[-3:])
        keyfn = (lambda i: (-(pr_by_iter.get(i, 0.0)), -i)) if tie_break == "recent" \
            else (lambda i: (-(pr_by_iter.get(i, 0.0)), i))
        best3 = set(sorted(avail, key=keyfn)[:3])
        prs = [pr_by_iter.get(i, 0.0) for i in avail]
        sp = (max(prs) - min(prs)) if prs else 0.0
        rds = reads_by_target(it_dir)
        for rl in targ:
            spread[rl].append(sp)
            for i in avail:
                in_r, in_b = i in recent3, i in best3
                bucket = ("both" if (in_r and in_b) else "recent_only" if in_r
                          else "best_only" if in_b else "neither")
                rd_cnt = rds.get(i, (0, 0))[0]
                disjoint[rl][bucket][0] += rd_cnt
                disjoint[rl][bucket][1] += 1
                if in_r:
                    marginal[rl]["in_recent3"][0] += rd_cnt; marginal[rl]["in_recent3"][1] += 1
                if in_b:
                    marginal[rl]["in_best3"][0] += rd_cnt; marginal[rl]["in_best3"][1] += 1
    return {"disjoint": disjoint, "marginal": marginal, "spread": spread}


def _rps(pair):  # reads-per-slot
    r, s = pair
    return (r / s) if s else 0.0


def _merge(group_results, key_path):
    """Sum [reads, slots] pairs across a list of per-run result dicts at key_path = (a,b,c)."""
    tot = [0, 0]
    for res in group_results:
        d = res
        for k in key_path:
            d = d[k]
        tot[0] += d[0]; tot[1] += d[1]
    return tuple(tot)


def report_partB(groups: dict[str, list[tuple[str, Path]]]) -> None:
    print("\n" + "=" * 100)
    print("PART B -- within a `default` run: does reference attention track RECENCY or candidate QUALITY?")
    print("  buckets are disjoint: recent_only = in last-3 but NOT top-3-by-passrate;  best_only = top-3 but NOT last-3;")
    print("  recent_only and best_only have EQUAL slot counts by construction, so reads/slot is a fair head-to-head.")
    print("=" * 100)
    for gname, runs in groups.items():
        if not runs:
            continue
        # tie_break = recent (default)
        res_recent = [per_run_partB(rd, "recent") for _, rd in runs]
        res_old = [per_run_partB(rd, "old") for _, rd in runs]
        if sum(_merge(res_recent, ("disjoint", "ALL", b))[0] for b in ("recent_only", "best_only", "both", "neither")) == 0:
            print(f"\n--- group: {gname}: (no per-Read reference-iter targets recoverable -- skip)")
            continue
        print(f"\n--- group: {gname}  (n_runs={len(runs)})")

        # (1) overall disjoint table, tie_break=recent
        print("  [1] overall (all rounds), tie-break=recent:")
        for b in ("both", "best_only", "recent_only", "neither"):
            r, s = _merge(res_recent, ("disjoint", "ALL", b))
            print(f"      {b:<14} reads {r:>6}  slots {s:>6}  reads/slot {_rps((r,s)):>7.3f}")
        bo = _merge(res_recent, ("disjoint", "ALL", "best_only")); ro = _merge(res_recent, ("disjoint", "ALL", "recent_only"))
        ratio = _rps(bo) / _rps(ro) if _rps(ro) else float("inf")
        print(f"      => best_only / recent_only reads/slot = {ratio:.1f}x   ({'QUALITY' if ratio>1 else 'RECENCY'} dominates)")

        # (2) by iter-range
        print("  [2] by iter-range  (recent_only/slot  vs  best_only/slot  ;  n slots each):")
        for lbl, _, _ in RANGES:
            r_ro = _merge(res_recent, ("disjoint", lbl, "recent_only"))
            r_bo = _merge(res_recent, ("disjoint", lbl, "best_only"))
            r_bt = _merge(res_recent, ("disjoint", lbl, "both"))
            print(f"      {lbl}:  recent_only {_rps(r_ro):6.3f} (n={r_ro[1]:>3})   best_only {_rps(r_bo):6.3f} (n={r_bo[1]:>3})   both {_rps(r_bt):6.3f} (n={r_bt[1]:>3})")

        # (3) tie-break robustness: tie_break=old
        print("  [3] tie-break robustness (tie-break=OLD instead of recent):")
        bo2 = _merge(res_old, ("disjoint", "ALL", "best_only")); ro2 = _merge(res_old, ("disjoint", "ALL", "recent_only"))
        r2 = _rps(bo2) / _rps(ro2) if _rps(ro2) else float("inf")
        print(f"      best_only {_rps(bo2):.3f} (n={bo2[1]})   recent_only {_rps(ro2):.3f} (n={ro2[1]})   => ratio {r2:.1f}x")

        # (4) marginal (overlapping) sanity: just "in last-3" vs "in top-3-by-passrate"
        mr = _merge(res_recent, ("marginal", "ALL", "in_recent3"))
        mb = _merge(res_recent, ("marginal", "ALL", "in_best3"))
        print(f"  [4] marginal (overlapping):  in last-3 reads/slot {_rps(mr):.3f} (n={mr[1]})   vs  in top-3-by-passrate reads/slot {_rps(mb):.3f} (n={mb[1]})")

        # (5) is the quality signal informative? per-round spread of available passrates, by range
        print("  [5] quality-signal spread = max-min of available prior-iter passrates, mean per round (if ~0, top-3 is arbitrary):")
        for lbl, _, _ in RANGES:
            sps = [v for res in res_recent for v in res["spread"][lbl]]
            print(f"      {lbl}: {(_nanmean(sps)):.3f}   (n_rounds={len(sps)})", end="   ")
        print()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", default=[], choices=sorted(GROUPS))
    args = ap.parse_args(argv)
    chosen = args.group or list(GROUPS)
    groups: dict[str, list[tuple[str, Path]]] = {}
    for g in chosen:
        rs = []
        for nm in GROUPS[g]:
            rd = resolve_run(nm)
            if rd is None:
                print(f"!! {nm}: not found", file=sys.stderr)
                continue
            rs.append((rd.name, rd))
        groups[g] = rs
    report_partA(groups)
    report_partB(groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
