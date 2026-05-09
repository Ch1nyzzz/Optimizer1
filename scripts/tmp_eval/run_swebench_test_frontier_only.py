#!/usr/bin/env python3
"""Evaluate the current SWE-bench train frontier on the full verified test set."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from optimizer1.pareto import ParetoPoint, save_frontier
from optimizer1.schemas import CandidateResult
from optimizer1.swebench import (
    DEFAULT_MINI_SWE_AGENT_NAME,
    MiniSweAgentSourceRunner,
    load_swebench_instances,
)


RUN_ID = "swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_fixedsource_iter20_trainfirst30_w10_t900_20260430_233750"
PROJECT_ROOT = Path("/data/home/yuhan/MemoMemo")
RUN_DIR = Path("/helios-storage/helios4-data/yuhan/MemoMemo/runs") / RUN_ID
DATA_PATH = PROJECT_ROOT / "data/swebench_verified_full.json"
TIMEOUT_S = 900
EVAL_WORKERS = 10

RUN_COMMAND = (
    f"python {PROJECT_ROOT / 'scripts/run_miniswe_swebench_single.py'} run "
    "--source-path {source_path} "
    "--instance-path {instance_path} "
    "--patch-path {patch_path} "
    "--task-dir {task_dir} "
    "--model openai/deepseek-v4-flash "
    "--base-url https://api.deepseek.com/v1 "
    "--max-tokens 4096"
)
EVAL_COMMAND = (
    f"python {PROJECT_ROOT / 'scripts/run_miniswe_swebench_single.py'} eval "
    "--source-path {source_path} "
    "--instance-path {instance_path} "
    "--patch-path {patch_path} "
    "--task-dir {task_dir}"
)


def source_path_for(candidate_id: str) -> str:
    iteration = candidate_id[4:7]
    path = (
        RUN_DIR
        / "proposer_calls"
        / f"iter_{iteration}"
        / "source_snapshot/candidate/upstream_source/mini-swe-agent"
    )
    if not path.exists():
        raise FileNotFoundError(f"missing source snapshot for {candidate_id}: {path}")
    return str(path)


def candidate_config(candidate: CandidateResult) -> dict[str, Any]:
    config = dict(candidate.config)
    extra = dict(config.get("extra") or {})
    source_path = source_path_for(candidate.candidate_id)
    extra["source_project_path"] = source_path
    config["extra"] = extra
    config["source_project_path"] = source_path
    config["agent_name"] = DEFAULT_MINI_SWE_AGENT_NAME
    config["scaffold_name"] = DEFAULT_MINI_SWE_AGENT_NAME
    config["source_family"] = DEFAULT_MINI_SWE_AGENT_NAME
    config["command"] = RUN_COMMAND
    config["eval_command"] = EVAL_COMMAND
    return config


def main() -> int:
    os.chdir(PROJECT_ROOT)
    candidates_payload = json.loads((RUN_DIR / "best_candidates.json").read_text(encoding="utf-8"))
    candidates = [CandidateResult.from_dict(item) for item in candidates_payload]
    test_examples = load_swebench_instances(DATA_PATH, split="test", limit=0)

    test_dir = RUN_DIR / "test_frontier"
    specs_dir = test_dir / "candidate_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    runner = MiniSweAgentSourceRunner(
        instances=test_examples,
        out_dir=test_dir,
        timeout_s=TIMEOUT_S,
        max_eval_workers=EVAL_WORKERS,
        dry_run=False,
        force=False,
    )

    rows: list[dict[str, Any]] = []
    test_results: list[CandidateResult] = []
    failures: list[dict[str, Any]] = []
    for candidate in candidates:
        config = candidate_config(candidate)
        test_candidate_id = f"test_{candidate.candidate_id}"
        spec_path = specs_dir / f"{test_candidate_id}.json"
        spec_path.write_text(
            json.dumps(
                {
                    "original_candidate_id": candidate.candidate_id,
                    "candidate_id": test_candidate_id,
                    **config,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"evaluating {test_candidate_id} on {len(test_examples)} test instances", flush=True)
        try:
            result = runner.evaluate_candidate(
                candidate=config,
                candidate_id=test_candidate_id,
                agent_name=DEFAULT_MINI_SWE_AGENT_NAME,
            )
        except Exception as exc:  # noqa: BLE001
            failure = {
                "original_candidate_id": candidate.candidate_id,
                "test_candidate_id": test_candidate_id,
                "candidate_spec_path": str(spec_path),
                "error": str(exc),
            }
            failures.append(failure)
            rows.append({"original_candidate": candidate.to_dict(), **failure})
            print(f"failed {test_candidate_id}: {exc}", flush=True)
            continue
        test_results.append(result)
        rows.append(
            {
                "original_candidate": candidate.to_dict(),
                "candidate_spec_path": str(spec_path),
                "test_candidate": result.to_dict(),
            }
        )
        print(f"finished {test_candidate_id}: passrate={result.passrate:.4f}", flush=True)

    results_path = test_dir / "test_results.json"
    results_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    test_frontier_path = test_dir / "test_pareto_frontier.json"
    save_frontier(
        test_frontier_path,
        [
            ParetoPoint(
                candidate_id=item.candidate_id,
                scaffold_name=item.scaffold_name,
                passrate=item.passrate,
                token_consuming=item.token_consuming,
                avg_token_consuming=item.avg_token_consuming,
                average_score=item.average_score,
                result_path=item.result_path,
                config=item.config,
            )
            for item in test_results
        ],
        quality_gap_threshold=0.125,
    )
    summary = {
        "split": "test",
        "limit": 0,
        "count": len(test_examples),
        "train_frontier_count": len(candidates),
        "evaluated_count": len(test_results),
        "failed_count": len(failures),
        "test_dir": str(test_dir),
        "test_results_path": str(results_path),
        "test_pareto_frontier_path": str(test_frontier_path),
        "candidate_spec_dir": str(specs_dir),
        "failures": failures,
    }
    summary_path = RUN_DIR / "test_frontier_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
