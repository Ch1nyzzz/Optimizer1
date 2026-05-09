#!/usr/bin/env python3
"""Evaluate one generated candidate on LongMemEval (companion to evaluate_candidate_json.py)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from memomemo.dynamic import load_candidate_scaffold
from memomemo.evaluation import EvaluationRunner
from memomemo.longmemeval import (
    DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL,
    DEFAULT_LONGMEMEVAL_JUDGE_MODEL,
    LongMemEvalJudge,
    default_data_path,
    load_longmemeval_examples,
    prepare_longmemeval,
    select_split,
)
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from memomemo.scaffolds.base import ScaffoldConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("warmup", "train", "test"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variant", default="s", choices=("s", "m", "oracle"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--judge-model", default=DEFAULT_LONGMEMEVAL_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-timeout-s", type=int, default=300)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--eval-workers", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_path = args.data_path or default_data_path(args.variant)
    if not data_path.exists():
        prepare_longmemeval(variant=args.variant, dest=data_path)
    examples = load_longmemeval_examples(
        data_path=data_path,
        variant=args.variant,
        question_types=tuple(),
    )
    examples = select_split(
        examples,
        split=args.split,
        variant=args.variant,
        split_path=args.split_path,
    )
    if args.limit:
        examples = examples[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    project_root = Path.cwd()
    scaffold = load_candidate_scaffold(candidate, project_root=project_root)
    config = ScaffoldConfig(
        top_k=int(candidate.get("top_k", 12)),
        window=int(candidate.get("window", 1)),
        extra=dict(candidate.get("extra") or {}),
    )

    judge_api_key = args.judge_api_key or os.environ.get("TOGETHER_API_KEY", "")
    judge = LongMemEvalJudge(
        model=args.judge_model,
        base_url=args.judge_base_url,
        api_key=judge_api_key,
        timeout_s=args.judge_timeout_s,
    )
    runner = EvaluationRunner(
        examples=examples,
        out_dir=args.out,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_s=args.timeout_s,
        max_context_chars=args.max_context_chars,
        max_eval_workers=args.eval_workers,
        force=args.force,
        score_run=judge.score_run,
    )
    result = runner.evaluate_scaffold(
        scaffold=scaffold,
        scaffold_name=str(candidate["name"]),
        config=config,
        candidate_id=str(candidate["candidate_id"]),
    )
    summary = {
        "benchmark": "longmemeval",
        "variant": args.variant,
        "split": args.split,
        "limit": args.limit,
        "count": len(examples),
        "model": args.model,
        "base_url": args.base_url,
        "judge_model": args.judge_model,
        "max_context_chars": args.max_context_chars,
        "max_eval_workers": args.eval_workers,
        "candidate": result.to_dict(),
        "candidate_spec": candidate,
    }
    (args.out / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
