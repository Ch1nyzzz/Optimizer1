#!/usr/bin/env python3
"""Run LongMemEval test-frontier evaluation for completed optimizer runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from optimizer1.longmemeval import (
    DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL,
    DEFAULT_LONGMEMEVAL_JUDGE_MODEL,
)
from optimizer1.longmemeval_optimizer import (
    LongMemEvalOptimizer,
    LongMemEvalOptimizerConfig,
)
from optimizer1.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from optimizer1.schemas import CandidateResult


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_candidates(path: Path) -> list[CandidateResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_items = payload if isinstance(payload, list) else payload.get("candidates", [])
    candidates = [CandidateResult.from_dict(item) for item in raw_items]
    if not candidates:
        raise ValueError(f"no candidates in {path}")
    return candidates


def _config_for_run(run_dir: Path, args: argparse.Namespace) -> LongMemEvalOptimizerConfig:
    summary = _load_json(run_dir / "run_summary.json")
    judge = summary.get("judge") if isinstance(summary.get("judge"), dict) else {}
    return LongMemEvalOptimizerConfig(
        run_id=run_dir.name,
        out_dir=run_dir,
        iterations=0,
        split=str(summary.get("split") or "train"),
        limit=int(summary.get("limit") or 0),
        dataset_variant=str(summary.get("variant") or args.variant),
        data_path=args.data_path,
        split_path=args.split_path,
        judge_model=str(judge.get("model") or args.judge_model),
        judge_base_url=str(judge.get("base_url") or args.judge_base_url),
        judge_api_key=args.judge_api_key,
        use_llm_judge=not args.no_llm_judge,
        model=str(summary.get("model") or args.model),
        base_url=str(summary.get("base_url") or args.base_url),
        api_key=str(summary.get("api_key") or args.api_key),
        eval_timeout_s=args.timeout_s,
        dry_run=False,
        max_context_chars=int(summary.get("max_context_chars") or args.max_context_chars),
        max_eval_workers=args.eval_workers,
        test_frontier=True,
        test_limit=args.limit,
        pareto_quality_threshold=float(summary.get("pareto_quality_threshold") or 0.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--variant", default="s")
    parser.add_argument("--data-path", type=Path, default=Path("data/longmemeval/longmemeval_s_cleaned.json"))
    parser.add_argument("--split-path", type=Path, default=Path("data/longmemeval/splits_s.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--eval-workers", type=int, default=16)
    parser.add_argument("--judge-model", default=DEFAULT_LONGMEMEVAL_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--no-llm-judge", action="store_true")
    args = parser.parse_args()

    summaries = []
    for run_dir in args.runs:
        run_dir = run_dir.resolve()
        print(json.dumps({"event": "start_run", "run": str(run_dir)}, ensure_ascii=False), flush=True)
        try:
            candidates = _load_candidates(run_dir / "best_candidates.json")
            optimizer = LongMemEvalOptimizer(_config_for_run(run_dir, args))
            summary = optimizer._run_test_frontier(candidates)
            summaries.append({"run": str(run_dir), "summary": summary})
        except Exception as exc:  # noqa: BLE001 - keep evaluating remaining runs
            summaries.append({"run": str(run_dir), "error": str(exc)})
        print(json.dumps(summaries[-1], indent=2, ensure_ascii=False), flush=True)

    print(json.dumps({"runs": summaries}, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
