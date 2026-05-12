#!/usr/bin/env python3
"""Patch-lineage / saturation analysis for the harness-optimizer runs.

Question this answers: in a propose--evaluate run, is the final retained best
patch a *superset* of the earlier breakthrough patches (the search compounded,
even if only by re-derivation against the fixed base) or is it a *different*
edit that merely scores higher (no compounding -- later breakthroughs replace
earlier ones rather than layering on them)?

For one run we:
  * read ``evolution_summary.jsonl`` to get per-iteration candidate passrate;
  * mark a *breakthrough* as an iteration whose candidate strictly improves the
    best-so-far passrate (ties broken by ``average_score``), counting iter 0
    (the seed baseline) as the initial best;
  * identify the *final retained best* iteration (highest passrate among the
    retained frontier in ``best_candidates.json``; falls back to the train-best
    over all evaluated candidates);
  * load each iteration's ``proposer_calls/iter_NNN/diff.patch`` and parse it
    into (a) the multiset of *non-trivial added content lines* (whitespace
    normalised), (b) the set of touched files, (c) the set of added/modified
    *symbols* (``def`` / ``class`` names plus dict-key / CONSTANT assignments);
  * for every breakthrough j <= k* report containment of delta_j in delta_k*
    at the line / file / symbol level, and the aggregate
    ``retained = |delta_k* cap (union_j delta_j)| / |union_j delta_j|``.

Runs are resolved local-first (``/data/home/yuhan/MemoMemo/runs``) then from
the helios archive (``/helios-storage/helios4-data/yuhan/MemoMemo/runs``).

Usage::

    python scripts/analyze_patch_lineage.py RUN [RUN ...] [--json out.json]
    python scripts/analyze_patch_lineage.py --group fixed_base_kimi

``--group`` expands to a curated list (see GROUPS below).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

RUN_ROOTS = [
    Path("/data/home/yuhan/MemoMemo/runs"),
    Path("/helios-storage/helios4-data/yuhan/MemoMemo/runs"),
    Path("/data/home/yuhan/Optimizer1/runs"),
]

# Curated run groups. These names are the canonical run-ids referenced in
# Optimizer1/docs/experiment_detail.md.
GROUPS: dict[str, list[str]] = {
    # fixed patch base (mu_t = delta_x0), kimi proposer, memory benchmarks.
    "fixed_base_kimi": [
        "locomo_memgpt_claudekimi_default_docker_iter30_train80_20260501_204004",
        "locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_015441",
        "locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_154556",
        "locomo_memgpt_claudekimi_default_autobudget_docker_iter30_train80_r1_20260504_162844",
        "longmemeval_memgpt_claudekimi_default_direction_docker_iter30_train100_20260502_015454",
        "longmemeval_memgpt_claudekimi_default_direction_docker_iter30_train100_20260502_152524",
        "longmemeval_memgpt_claudekimi_default_autobudget_docker_iter30_train100_r1_20260504_162844",
    ],
    # rotating patch base: curaii (base resampled from top-3 by passrate).
    "rebase_kimi": [
        "locomo_memgpt_claudekimi_curaii_docker_iter30_train80_20260508_072808",
        "locomo_memgpt_claudekimi_curaii_v2_docker_iter30_train80_20260508_190514",
        "longmemeval_memgpt_claudekimi_curaii_docker_iter30_train100_20260508_072808",
        "longmemeval_memgpt_claudekimi_curaii_v2_docker_iter30_train100_20260508_190514",
    ],
    # bandit / progressive auto-budget (still fixed base, but adaptive context).
    "adaptive_kimi": [
        "locomo_memgpt_claudekimi_bandit_v3_autobudget_docker_iter30_train80_w16_r1_20260504_162844",
        "locomo_memgpt_claudekimi_bandit_v3_autobudget_docker_iter30_train80_w16_r2_20260504_162844",
        "locomo_memgpt_claudekimi_bandit_v4_autobudget_docker_iter30_train80_w16_r1_20260505_040626",
        "locomo_memgpt_claudekimi_progressive_autobudget_docker_iter30_train80_r1_20260504_162844",
        "locomo_memgpt_claudekimi_progressive_autobudget_docker_iter30_train80_r2_20260504_162844",
        "longmemeval_memgpt_claudekimi_bandit_v3_banditfix_autobudget_docker_iter30_train100_w16_r1_20260505_003416",
        "longmemeval_memgpt_claudekimi_bandit_v3_banditfix_autobudget_docker_iter30_train100_w16_r2_20260505_003416",
        "longmemeval_memgpt_claudekimi_bandit_v4_autobudget_docker_iter30_train100_w16_r1_20260505_040626",
        "longmemeval_memgpt_claudekimi_progressive_autobudget_docker_iter30_train100_r1_20260504_162844",
    ],
}

# A line is "trivial" (carries no mechanism information) if, after stripping, it
# is empty, a lone bracket/punctuation run, or a bare keyword like ``else:``.
_TRIVIAL_RE = re.compile(r"^[\s\)\(\]\[\}\{:,\.]*$|^(else|try|except|finally|pass|return|continue|break|raise)\s*:?\s*$")
# Symbols: function / class definitions, and module-level CONSTANT / dict-key
# assignments that the scaffolds use to register mechanisms.
_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)")
_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)")
_ASSIGN_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]{2,})\s*[:=]")
_KEY_RE = re.compile(r"""['"]([A-Za-z_][\w\-]{2,})['"]\s*:""")


def normalise_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_trivial(text: str) -> bool:
    s = text.strip()
    if len(s) < 4:
        return True
    return bool(_TRIVIAL_RE.match(s))


def rel_source_path(diff_b_path: str) -> str:
    """Reduce the ``+++ b/...`` path to something stable across iters/runs."""
    p = diff_b_path
    for marker in ("/project_source/", "/candidate/project_source/", "project_source/"):
        idx = p.find(marker)
        if idx != -1:
            return p[idx + len(marker):]
    # last resort: drop everything up to the last "src/"
    idx = p.rfind("/src/")
    return p[idx + 1:] if idx != -1 else p


@dataclass
class PatchInfo:
    iteration: int
    added: Counter = field(default_factory=Counter)        # non-trivial added content lines
    removed: Counter = field(default_factory=Counter)      # non-trivial removed content lines
    files: set[str] = field(default_factory=set)
    symbols: set[str] = field(default_factory=set)         # added/modified symbol names
    n_added_raw: int = 0
    n_removed_raw: int = 0
    empty: bool = False

    @property
    def added_keys(self) -> set[str]:
        return set(self.added)


def parse_patch(path: Path, iteration: int) -> PatchInfo:
    info = PatchInfo(iteration=iteration)
    if not path.exists() or path.stat().st_size == 0:
        info.empty = True
        return info
    cur_file = ""
    for raw in path.read_text(errors="replace").splitlines():
        if raw.startswith("+++ "):
            cur_file = rel_source_path(raw[4:].strip())
            if cur_file and cur_file != "/dev/null":
                info.files.add(cur_file)
            continue
        if raw.startswith("--- ") or raw.startswith("diff --git") or raw.startswith("index ") or raw.startswith("@@"):
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            content = raw[1:]
            info.n_added_raw += 1
            for rx in (_DEF_RE, _CLASS_RE, _ASSIGN_RE):
                m = rx.match(content)
                if m:
                    info.symbols.add(m.group(1))
            for m in _KEY_RE.finditer(content):
                info.symbols.add(m.group(1))
            if not is_trivial(content):
                info.added[normalise_line(content)] += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            content = raw[1:]
            info.n_removed_raw += 1
            if not is_trivial(content):
                info.removed[normalise_line(content)] += 1
    info.empty = info.n_added_raw == 0 and info.n_removed_raw == 0
    return info


@dataclass
class IterRecord:
    iteration: int
    candidate_id: str | None
    passrate: float | None
    average_score: float | None


def load_iters(run_dir: Path) -> list[IterRecord]:
    """Per-iteration evaluated candidate, from evolution_summary.jsonl."""
    es = run_dir / "evolution_summary.jsonl"
    out: dict[int, IterRecord] = {}
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
            cand = d.get("candidate") or {}
            if it is None or not cand:
                continue
            pr = cand.get("passrate")
            # keep the *last* candidate record per iter that has a numeric passrate
            rec = IterRecord(it, cand.get("candidate_id"), pr, cand.get("average_score"))
            if it not in out or (pr is not None):
                out[it] = rec
        if out:
            return [out[k] for k in sorted(out)]
    # fallback: iteration_index.json + candidate_results/*.json
    idx_path = run_dir / "iteration_index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        for entry in idx:
            it = entry.get("iteration")
            cids = entry.get("candidate_ids") or []
            best = None
            for cid in cids:
                cr = run_dir / "candidate_results" / f"{cid}.json"
                if cr.exists():
                    try:
                        c = json.loads(cr.read_text()).get("candidate") or {}
                    except json.JSONDecodeError:
                        continue
                    if best is None or (c.get("passrate") or -1) > (best.get("passrate") or -1):
                        best = c
            if best is not None:
                out[it] = IterRecord(it, best.get("candidate_id"), best.get("passrate"), best.get("average_score"))
        return [out[k] for k in sorted(out)]
    return []


def find_breakthroughs(iters: list[IterRecord]) -> list[int]:
    bts: list[int] = []
    best_pr = float("-inf")
    best_av = float("-inf")
    for rec in iters:
        if rec.passrate is None:
            continue
        pr = rec.passrate
        av = rec.average_score if rec.average_score is not None else float("-inf")
        if pr > best_pr + 1e-12 or (abs(pr - best_pr) <= 1e-12 and av > best_av + 1e-12):
            if rec.iteration > 0:  # iter 0 is the seed; don't call it a breakthrough
                bts.append(rec.iteration)
            best_pr, best_av = max(best_pr, pr), max(best_av, av)
    return bts


def final_best_iter(run_dir: Path, iters: list[IterRecord]) -> tuple[int | None, float | None]:
    """Highest-passrate iteration among the retained frontier; fall back to
    train-best over all evaluated candidates."""
    bc = run_dir / "best_candidates.json"
    retained_ids: set[str] = set()
    if bc.exists():
        try:
            for c in json.loads(bc.read_text()):
                cid = c.get("candidate_id") or c.get("scaffold_name")
                if cid:
                    retained_ids.add(cid)
        except json.JSONDecodeError:
            pass
    # map candidate_id -> iteration
    cid_to_iter = {r.candidate_id: r.iteration for r in iters if r.candidate_id}
    cand_pool = [r for r in iters if r.passrate is not None and r.iteration > 0]
    if retained_ids:
        retained_iters = [r for r in cand_pool if r.candidate_id in retained_ids]
        if retained_iters:
            best = max(retained_iters, key=lambda r: (r.passrate, r.average_score or 0))
            return best.iteration, best.passrate
    if cand_pool:
        best = max(cand_pool, key=lambda r: (r.passrate, r.average_score or 0))
        return best.iteration, best.passrate
    return None, None


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(sub: set, sup: set) -> float:
    """Fraction of `sub` that is present in `sup`."""
    if not sub:
        return float("nan")
    return len(sub & sup) / len(sub)


def multiset_containment(sub: Counter, sup: Counter) -> float:
    if not sub:
        return float("nan")
    inter = sum(min(sub[k], sup.get(k, 0)) for k in sub)
    return inter / sum(sub.values())


def _run_is_intact(d: Path) -> bool:
    es = d / "evolution_summary.jsonl"
    if es.exists() and es.stat().st_size > 0:
        return True
    cr = d / "candidate_results"
    return cr.is_dir() and any(cr.iterdir())


def resolve_run(name_or_path: str) -> Path | None:
    p = Path(name_or_path)
    if p.is_dir():
        return p
    candidates = [root / name_or_path for root in RUN_ROOTS if (root / name_or_path).is_dir()]
    if not candidates:
        return None
    # prefer a copy that still has its evaluation artifacts (some local copies
    # are gutted husks -- source snapshots survive but results were archived).
    for c in candidates:
        if _run_is_intact(c):
            return c
    return candidates[0]


def analyse_run(run_dir: Path) -> dict:
    iters = load_iters(run_dir)
    if not iters:
        return {"run": run_dir.name, "error": "no iteration records"}
    bts = find_breakthroughs(iters)
    k_star, k_star_pr = final_best_iter(run_dir, iters)
    seed_pr = next((r.passrate for r in iters if r.iteration == 0), None)

    # load patches for the iters we care about: every breakthrough + k*.
    want = sorted(set(bts) | ({k_star} if k_star else set()))
    patches: dict[int, PatchInfo] = {}
    for it in want:
        patches[it] = parse_patch(run_dir / "proposer_calls" / f"iter_{it:03d}" / "diff.patch", it)

    # breakthroughs strictly before or at k*
    pre_bts = [b for b in bts if k_star is not None and b <= k_star]
    pk = patches.get(k_star) if k_star is not None else None

    per_bt = []
    union_added: Counter = Counter()
    union_files: set[str] = set()
    union_syms: set[str] = set()
    for b in pre_bts:
        pb = patches[b]
        if b != k_star:
            union_added.update(pb.added)
            union_files |= pb.files
            union_syms |= pb.symbols
        entry = {
            "iter": b,
            "is_final": b == k_star,
            "added_lines_nontrivial": len(pb.added),
            "files": sorted(pb.files),
            "symbols": sorted(pb.symbols),
            "n_add_raw": pb.n_added_raw,
            "n_rem_raw": pb.n_removed_raw,
        }
        if pk is not None and b != k_star:
            # forward: how much of bt_b survives in the final patch
            entry["line_containment_in_final"] = round(multiset_containment(pb.added, pk.added), 4)
            entry["symbol_containment_in_final"] = round(containment(pb.symbols, pk.symbols), 4) if pb.symbols else None
            # backward: how much of the final patch was already in bt_b
            #   layering   => forward high, backward low (final = bt + more)
            #   replacement=> both moderate (different edit of similar size)
            #   dead-end   => forward low (bt's lines mostly not in final)
            entry["final_line_in_this"] = round(multiset_containment(pk.added, pb.added), 4)
            entry["final_symbol_in_this"] = round(containment(pk.symbols, pb.symbols), 4) if pk.symbols else None
            entry["file_jaccard_with_final"] = round(jaccard(pb.files, pk.files), 4)
            entry["size_ratio_final_over_this"] = round(
                (sum(pk.added.values()) or 0) / (sum(pb.added.values()) or 1), 3)
        per_bt.append(entry)

    retained = {}
    if pk is not None and union_added:
        retained["line_level"] = round(multiset_containment(union_added, pk.added), 4)
    if pk is not None and union_syms:
        retained["symbol_level"] = round(containment(union_syms, pk.symbols), 4)
    if pk is not None and union_files:
        retained["file_level"] = round(containment(union_files, pk.files), 4)

    # also: pairwise consecutive-breakthrough overlap (does bt_{i+1} keep bt_i?)
    consec = []
    bt_patches_all = {b: parse_patch(run_dir / "proposer_calls" / f"iter_{b:03d}" / "diff.patch", b) for b in bts}
    for a, c in zip(bts, bts[1:]):
        pa, pc = bt_patches_all[a], bt_patches_all[c]
        consec.append({
            "from": a, "to": c,
            # fwd = fraction of bt_a's added lines kept in bt_c
            "line_fwd": round(multiset_containment(pa.added, pc.added), 4) if pa.added else None,
            # bwd = fraction of bt_c's added lines already in bt_a
            "line_bwd": round(multiset_containment(pc.added, pa.added), 4) if pc.added else None,
            "file_jaccard": round(jaccard(pa.files, pc.files), 4),
            "symbol_fwd": round(containment(pa.symbols, pc.symbols), 4) if pa.symbols else None,
            "size_ratio": round((sum(pc.added.values()) or 0) / (sum(pa.added.values()) or 1), 3),
        })
    # split consecutive overlap into early vs late half of the breakthrough run
    half = len(consec) // 2
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None
    compounding_profile = {
        "n_consec_pairs": len(consec),
        "early_half_line_fwd_mean": _mean([c["line_fwd"] for c in consec[:half]]) if half else None,
        "late_half_line_fwd_mean": _mean([c["line_fwd"] for c in consec[half:]]) if half else None,
        "early_half_symbol_fwd_mean": _mean([c["symbol_fwd"] for c in consec[:half]]) if half else None,
        "late_half_symbol_fwd_mean": _mean([c["symbol_fwd"] for c in consec[half:]]) if half else None,
    }

    return {
        "run": run_dir.name,
        "n_iters_evaluated": sum(1 for r in iters if r.passrate is not None),
        "seed_passrate": seed_pr,
        "breakthroughs": bts,
        "n_breakthroughs": len(bts),
        "final_best_iter": k_star,
        "final_best_passrate": k_star_pr,
        "pre_kstar_breakthroughs": pre_bts,
        "retained_in_final": retained,
        "per_breakthrough": per_bt,
        "consecutive_breakthrough_overlap": consec,
        "compounding_profile": compounding_profile,
        "final_patch": None if pk is None else {
            "iter": k_star,
            "added_lines_nontrivial": len(pk.added),
            "files": sorted(pk.files),
            "n_symbols": len(pk.symbols),
            "n_add_raw": pk.n_added_raw,
            "n_rem_raw": pk.n_removed_raw,
        },
    }


def fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "  -- "
    return f"{100 * x:5.1f}%"


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 100)
    print("PATCH-LINEAGE / SATURATION SUMMARY")
    print("=" * 100)
    for r in results:
        if "error" in r:
            print(f"\n{r['run']}: ERROR -- {r['error']}")
            continue
        print(f"\n### {r['run']}")
        print(f"  evaluated iters: {r['n_iters_evaluated']}  seed passrate: {r['seed_passrate']}")
        print(f"  breakthroughs ({r['n_breakthroughs']}): {r['breakthroughs']}")
        print(f"  final retained best: iter {r['final_best_iter']}  passrate {r['final_best_passrate']}")
        ret = r["retained_in_final"]
        print(f"  RETAINED in final patch -- line: {fmt_pct(ret.get('line_level'))}  "
              f"symbol: {fmt_pct(ret.get('symbol_level'))}  file: {fmt_pct(ret.get('file_level'))}   "
              f"(of the union of pre-k* breakthrough patches)")
        if r["final_patch"]:
            fp = r["final_patch"]
            print(f"  final patch shape: +{fp['n_add_raw']} / -{fp['n_rem_raw']} raw lines, "
                  f"{fp['added_lines_nontrivial']} non-trivial added, {len(fp['files'])} files, {fp['n_symbols']} symbols")
        if r["per_breakthrough"]:
            print("  per pre-k* breakthrough vs final patch  (fwd = bt's lines kept in final; bwd = final's lines already in bt):")
            print("    iter  fin  +raw  -raw  add*   fwd_line  bwd_line   fwd_sym  bwd_sym   x_size  fileJ")
            for e in r["per_breakthrough"]:
                if e["is_final"]:
                    print(f"    {e['iter']:>4}  *   {e['n_add_raw']:>4}  {e['n_rem_raw']:>4}  {e['added_lines_nontrivial']:>4}   (final patch: {len(e['files'])} files, {len(e['symbols'])} symbols)")
                    continue
                print(f"    {e['iter']:>4}      {e['n_add_raw']:>4}  {e['n_rem_raw']:>4}  {e['added_lines_nontrivial']:>4}   "
                      f"{fmt_pct(e.get('line_containment_in_final'))}  {fmt_pct(e.get('final_line_in_this'))}   "
                      f"{fmt_pct(e.get('symbol_containment_in_final'))}  {fmt_pct(e.get('final_symbol_in_this'))}   "
                      f"{e.get('size_ratio_final_over_this', float('nan')):>5.1f}x  {fmt_pct(e.get('file_jaccard_with_final'))}")
        if r["consecutive_breakthrough_overlap"]:
            cp = r["compounding_profile"]
            print(f"  consecutive breakthrough overlap (fwd = bt_i's lines kept in bt_(i+1)):")
            for c in r["consecutive_breakthrough_overlap"]:
                print(f"    {c['from']:>3} -> {c['to']:<3}  fwd_line {fmt_pct(c['line_fwd'])}  bwd_line {fmt_pct(c['line_bwd'])}  "
                      f"fwd_sym {fmt_pct(c['symbol_fwd'])}  x_size {c['size_ratio']:>5.2f}  fileJ {fmt_pct(c['file_jaccard'])}")
            print(f"  COMPOUNDING PROFILE: early-half mean fwd_line {fmt_pct(cp['early_half_line_fwd_mean'])} / sym {fmt_pct(cp['early_half_symbol_fwd_mean'])}   "
                  f"-> late-half mean fwd_line {fmt_pct(cp['late_half_line_fwd_mean'])} / sym {fmt_pct(cp['late_half_symbol_fwd_mean'])}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run dir paths or canonical run-ids")
    ap.add_argument("--group", action="append", default=[], choices=sorted(GROUPS), help="expand a curated run group")
    ap.add_argument("--json", type=Path, help="write per-run results JSON here")
    args = ap.parse_args(argv)

    names: list[str] = list(args.runs)
    for g in args.group:
        names.extend(GROUPS[g])
    if not names:
        ap.error("no runs given (pass run-ids or --group)")

    results = []
    for name in names:
        rd = resolve_run(name)
        if rd is None:
            results.append({"run": name, "error": "run dir not found in any known root"})
            print(f"!! {name}: not found", file=sys.stderr)
            continue
        print(f".. analysing {rd}", file=sys.stderr)
        results.append(analyse_run(rd))

    print_summary(results)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
