from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from optimizer1.dynamic import load_candidate_scaffold
from optimizer1.evaluation import EvaluationRunner
from optimizer1.longmemeval import (
    DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL,
    DEFAULT_LONGMEMEVAL_JUDGE_MODEL,
    LongMemEvalJudge,
    _fallback_score_run,
    load_longmemeval_examples,
    select_split,
)
from optimizer1.scaffolds.base import ScaffoldConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--variant", default="s")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="/data/home/yuhan/model_zoo/Qwen3-8B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--judge-model", default=DEFAULT_LONGMEMEVAL_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=None)
    args = parser.parse_args()

    payload = json.loads(args.pending_eval.read_text(encoding="utf-8"))
    raw = (payload.get("candidates") or [])[0]
    raw.setdefault("candidate_root", str(args.pending_eval.parent.parent.parent / "generated"))
    scaffold = load_candidate_scaffold(raw, project_root=Path.cwd())

    examples = select_split(
        load_longmemeval_examples(data_path=args.data_path, variant=args.variant),
        split=args.split,
        variant=args.variant,
        split_path=args.split_path,
    )
    if args.limit:
        examples = examples[: args.limit]

    judge_key = args.judge_api_key or os.environ.get("TOGETHER_API_KEY", "")
    judge = LongMemEvalJudge(
        model=args.judge_model,
        base_url=args.judge_base_url,
        api_key=judge_key,
        timeout_s=300,
    ) if judge_key else None

    extra = dict(raw.get("extra") or {})
    for key in (
        "build_tag", "class", "cost_level", "factory", "module", "module_path",
        "project_source_path", "source_base_dir", "source_family", "source_path",
        "source_project_path", "upstream_source_path", "mem0_source_path",
        "memgpt_source_path", "membank_source_path", "optimization_target",
    ):
        if key in raw and key not in extra:
            extra[key] = raw[key]
    extra.setdefault("benchmark", "longmemeval")
    extra.setdefault("scoring_method", "longmemeval_llm_judge" if judge else "token_f1")
    if judge:
        extra.setdefault("judge_model", args.judge_model)

    config = ScaffoldConfig(
        top_k=int(raw.get("top_k", 8)),
        window=int(raw.get("window", 1)),
        extra=extra,
    )
    candidate_name = str(raw.get("name") or scaffold.name)
    candidate_id = f"{candidate_name}_top{config.top_k}"

    args.out.mkdir(parents=True, exist_ok=True)
    runner = EvaluationRunner(
        examples=examples,
        out_dir=args.out,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_s=args.timeout_s,
        dry_run=False,
        max_context_chars=args.max_context_chars,
        max_eval_workers=args.eval_workers,
        force=True,
        score_run=judge.score_run if judge else _fallback_score_run,
    )
    result = runner.evaluate_scaffold(
        scaffold=scaffold,
        scaffold_name=candidate_name,
        config=config,
        candidate_id=candidate_id,
    )
    summary = {
        "benchmark": "longmemeval",
        "split": args.split,
        "count": len(examples),
        "candidate": result.to_dict(),
        "pending_eval": str(args.pending_eval),
        "loaded_scaffold_class": scaffold.__class__.__module__ + "." + scaffold.__class__.__qualname__,
    }
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
