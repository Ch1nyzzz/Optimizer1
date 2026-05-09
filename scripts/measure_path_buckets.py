#!/usr/bin/env python3
"""Bucket every Shell tool call's file-path references across codex54 runs.

For each Shell tool call in ``proposer_calls/iter_NNN/agent/tool_access.json``, we
extract every file-path token matching
``(summaries|reference_iterations|source_snapshot)/[\\w./_-]+`` and assign it
to one of seven exclusive buckets:

- summ-agg:        summaries/{best_candidates|evolution_summary|candidate_score_table|
                              retrieval_diagnostics_summary|iteration_index|diff_summary}*
- summ-oth:        any other summaries/*
- ref-iter-diff:   reference_iterations/iter_NNN/(diff.patch|diff_digest.md|pending_eval.json)
- ref-iter-traces: reference_iterations/iter_NNN/traces/...
- ref-iter-source: reference_iterations/iter_NNN/source_snapshot/...
- ref-iter-other:  any other reference_iterations/iter_NNN/*
- clean-source:    source_snapshot/candidate/... (NOT under reference_iterations)

This complements ``measure_top1_attention.py`` (which buckets by Read-call target
iter under the kimi proposer). Codex54 traces wrap every operation as a single
Shell tool call so per-Read bucketing is unavailable, but the path-reference
extraction here gives a coarse but cross-policy proxy for *what kinds of
reference-iteration artifacts the proposer inspects*.

Sample: 3 default + 3 recent3 codex54 runs per benchmark on LoCoMo and
LongMemEval (LongMemEval default has only 2 retained 30-iter runs; the third
crashed early). Output is per-run shares plus a (benchmark, policy) aggregate.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RUNS_DIR = Path("/data/home/yuhan/MemoMemo/runs")

GROUPS = {
    "LoCoMo / default": [
        "locomo_memgpt_codex54_default_codexlogin_autobudget_docker_iter30_train80_r1_20260504_163640",
        "locomo_memgpt_codex54_default_codexlogin_autobudget_docker_iter30_train80_r3_20260505_005403",
        "locomo_memgpt_codex54_default_docker_iter30_train80_20260501_204007",
    ],
    "LoCoMo / recent3": [
        "locomo_memgpt_codex54_recent3_fileaccess_docker_iter30_train80_r1_20260505_152300",
        "locomo_memgpt_codex54_recent3_fileaccess_docker_iter30_train80_r2_20260505_152300",
        "locomo_memgpt_codex54_recent3_fileaccess_docker_iter30_train80_r3_20260505_152300",
    ],
    "LongMemEval / default": [
        "longmemeval_memgpt_codex54_default_codexlogin_autobudget_docker_iter30_train100_r1_20260504_163640",
        "longmemeval_memgpt_codex54_default_codexlogin_autobudget_docker_iter30_train100_r3_retryenv_20260505_005443",
    ],
    "LongMemEval / recent3": [
        "longmemeval_memgpt_codex54_recent3_fileaccess_docker_iter30_train100_r1_20260505_152300",
        "longmemeval_memgpt_codex54_recent3_fileaccess_docker_iter30_train100_r2_20260505_152300",
        "longmemeval_memgpt_codex54_recent3_fileaccess_docker_iter30_train100_r3_20260505_152300",
    ],
}

PATH_RX = re.compile(r"(?:summaries|reference_iterations|source_snapshot)/[\w./_-]+")
SUMMARIES_AGG_NAMES = {
    "best_candidates", "evolution_summary", "candidate_score_table",
    "retrieval_diagnostics_summary", "iteration_index", "diff_summary",
}


def bucket_for(path: str) -> str:
    if path.startswith("source_snapshot/candidate/") or path.startswith("source_snapshot/"):
        return "clean-source"
    if path.startswith("summaries/"):
        rest = path[len("summaries/"):]
        head = rest.split(".")[0].split("/")[0]
        return "summ-agg" if head in SUMMARIES_AGG_NAMES else "summ-oth"
    if path.startswith("reference_iterations/"):
        m = re.match(r"reference_iterations/iter_\d+/(.+)$", path)
        if not m:
            return "ref-iter-other"
        sub = m.group(1)
        if sub.startswith("traces/"):
            return "ref-iter-traces"
        if sub.startswith("source_snapshot/"):
            return "ref-iter-source"
        head = sub.split("/")[0]
        if head in {"diff.patch", "diff_digest.md", "pending_eval.json"}:
            return "ref-iter-diff"
        return "ref-iter-other"
    return "other"


def analyze_run(run_dir: Path) -> tuple[Counter, int, int]:
    bucket: Counter = Counter()
    n_shell = 0
    n_iters_with_data = 0
    pc_dir = run_dir / "proposer_calls"
    if not pc_dir.exists():
        return bucket, 0, 0
    for it_dir in sorted(pc_dir.iterdir()):
        m = re.match(r"iter_(\d+)", it_dir.name)
        if not it_dir.is_dir() or not m or int(m.group(1)) == 0:
            continue
        ta = it_dir / "agent" / "tool_access.json"
        if not ta.exists():
            continue
        d = json.load(ta.open())
        any_match = False
        for u in d.get("tool_uses", []):
            if u.get("name") not in {"Shell", "Read", "Bash"}:
                continue
            inp = u.get("input") or {}
            text = inp.get("command", "") or inp.get("file_path", "")
            paths = PATH_RX.findall(text)
            n_shell += 1
            for p in paths:
                bucket[bucket_for(p)] += 1
                any_match = True
        if any_match:
            n_iters_with_data += 1
    return bucket, n_shell, n_iters_with_data


def main() -> None:
    cols = ["summ-agg", "summ-oth", "ref-iter-diff", "ref-iter-trace",
            "ref-iter-source", "ref-iter-other", "clean-source"]

    print("Per-run shares (%) of file-path references across Shell tool calls")
    print("=" * 135)
    hdr = (
        f"{'cell':<22} {'run':<6} {'shells':>6} {'paths':>6} | "
        + " ".join(f"{c[:8]:>9}" for c in cols)
    )
    print(hdr)
    print("-" * 135)

    agg: dict[str, dict] = defaultdict(lambda: {"b": Counter(), "shell": 0, "paths": 0, "n": 0})
    for label, names in GROUPS.items():
        for name in names:
            rd = RUNS_DIR / name
            if not rd.exists():
                continue
            b, n_shell, _ = analyze_run(rd)
            n_paths = sum(b.values())
            run_lbl = re.search(r"_r(\d)_", name)
            run_lbl = run_lbl.group(1) if run_lbl else "—"
            shares = " ".join(f"{b[c] / max(n_paths, 1) * 100:>8.1f}%" for c in cols)
            print(f"{label[:21]:<22} r{run_lbl:<5} {n_shell:>6} {n_paths:>6} | {shares}")
            a = agg[label]
            a["b"] += b
            a["shell"] += n_shell
            a["paths"] += n_paths
            a["n"] += 1

    print()
    print("Aggregated by (bench, policy)")
    print("=" * 135)
    print(hdr)
    print("-" * 135)
    for label, info in GROUPS.items():
        a = agg[label]
        if a["paths"] == 0:
            continue
        shares = " ".join(f"{a['b'][c] / a['paths'] * 100:>8.1f}%" for c in cols)
        print(f"{label[:21]:<22} n={a['n']}    {a['shell']:>6} {a['paths']:>6} | {shares}")


if __name__ == "__main__":
    main()
