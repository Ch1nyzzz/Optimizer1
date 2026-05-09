#!/usr/bin/env python3
"""Resume the DeepSeek v4 Flash SWE-bench bandit run and evaluate test frontier."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from optimizer1.pareto import ParetoPoint, save_frontier
from optimizer1.post_eval import write_post_eval_artifacts
from optimizer1.schemas import CandidateResult
from optimizer1.swebench import (
    DEFAULT_MINI_SWE_AGENT_NAME,
    MiniSweAgentSourceRunner,
    load_swebench_instances,
)
from optimizer1.swebench_optimizer import SwebenchOptimizer, SwebenchOptimizerConfig


RUN_ID = "swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_fixedsource_iter20_trainfirst30_w10_t900_20260430_233750"
PROJECT_ROOT = Path("/data/home/yuhan/MemoMemo")
RUN_DIR = Path("/helios-storage/helios4-data/yuhan/MemoMemo/runs") / RUN_ID
OLD_RUN_PREFIX = f"/data/home/yuhan/MemoMemo/runs/{RUN_ID}"
NEW_RUN_PREFIX = str(RUN_DIR)
DATA_PATH = PROJECT_ROOT / "data/swebench_verified_full.json"
MINI_SOURCE = PROJECT_ROOT / "references/vendor/mini-swe-agent"
TRAIN_LIMIT = 30
EVAL_WORKERS = 10
TIMEOUT_S = 900

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


def remap_paths(value: Any) -> Any:
    if isinstance(value, str):
        helios_project = "/helios-storage/helios4-data/yuhan/MemoMemo"
        value = value.replace(f"{helios_project}//{NEW_RUN_PREFIX}", NEW_RUN_PREFIX)
        value = value.replace(f"{helios_project}/{NEW_RUN_PREFIX}", NEW_RUN_PREFIX)
        if value.startswith(OLD_RUN_PREFIX):
            value = NEW_RUN_PREFIX + value[len(OLD_RUN_PREFIX):]
        relative_run = f"runs/{RUN_ID}"
        if value.startswith(relative_run):
            value = NEW_RUN_PREFIX + value[len(relative_run):]
        return value
    if isinstance(value, list):
        return [remap_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: remap_paths(item) for key, item in value.items()}
    return value


def normalize_candidate(candidate: CandidateResult) -> CandidateResult:
    payload = candidate.to_dict()
    payload["config"] = normalize_candidate_config(payload.get("config") or {})
    payload["result_path"] = remap_paths(payload.get("result_path", ""))
    return CandidateResult.from_dict(payload)


def normalize_candidate_config(config: dict[str, Any]) -> dict[str, Any]:
    out = remap_paths(dict(config))
    out.setdefault("command", RUN_COMMAND)
    out.setdefault("eval_command", EVAL_COMMAND)
    out.setdefault("agent_name", DEFAULT_MINI_SWE_AGENT_NAME)
    out.setdefault("source_family", DEFAULT_MINI_SWE_AGENT_NAME)
    extra = out.get("extra") if isinstance(out.get("extra"), dict) else {}
    out["extra"] = extra
    if "source_project_path" not in out and "source_project_path" not in extra:
        out["source_project_path"] = str(MINI_SOURCE)
    return out


def load_candidates(optimizer: SwebenchOptimizer) -> list[CandidateResult]:
    return [normalize_candidate(item) for item in optimizer._load_existing_candidates()]


def pending_candidates(iteration: int) -> list[dict[str, Any]]:
    pending_path = RUN_DIR / "proposer_calls" / f"iter_{iteration:03d}" / "pending_eval.json"
    if not pending_path.exists():
        return []
    payload = json.loads(pending_path.read_text(encoding="utf-8"))
    raw = payload.get("candidates", []) if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        return []
    return [normalize_candidate_config(dict(item)) for item in raw if isinstance(item, dict)]


def candidate_iteration(candidate: CandidateResult) -> int | None:
    text = candidate.candidate_id
    if not text.startswith("iter") or len(text) < 7:
        return None
    try:
        return int(text[4:7])
    except ValueError:
        return None


def keep_training_history(candidate: CandidateResult) -> bool:
    iteration = candidate_iteration(candidate)
    return iteration is None or iteration <= 14


def candidates_for_iteration(
    candidates: list[CandidateResult],
    iteration: int,
) -> list[CandidateResult]:
    return [item for item in candidates if candidate_iteration(item) == iteration]


def candidates_before_iteration(
    candidates: list[CandidateResult],
    iteration: int,
) -> list[CandidateResult]:
    out: list[CandidateResult] = []
    for item in candidates:
        item_iter = candidate_iteration(item)
        if item_iter is None or item_iter < iteration:
            out.append(item)
    return out


def rebuild_bandit_state(
    optimizer: SwebenchOptimizer,
    candidates: list[CandidateResult],
    *,
    through_iteration: int,
) -> None:
    if optimizer.bandit_state_path.exists():
        backup = optimizer.bandit_state_path.with_suffix(".pre_clean_resume.json")
        shutil.copy2(optimizer.bandit_state_path, backup)
        optimizer.bandit_state_path.unlink()
        print(f"backed up old bandit state to {backup}", flush=True)

    for iteration in range(1, through_iteration + 1):
        previous_candidates = candidates_before_iteration(candidates, iteration)
        evaluated = candidates_for_iteration(candidates, iteration)
        optimizer._update_bandit_state(
            iteration=iteration,
            previous_best_passrate=optimizer._best_passrate(previous_candidates),
            previous_best_quality=optimizer._best_quality_value(previous_candidates),
            evaluated=evaluated,
            call_dir=optimizer._iteration_dir(iteration),
        )


def write_iteration_artifacts(
    optimizer: SwebenchOptimizer,
    iteration: int,
    candidates: list[CandidateResult],
    evaluated: list[CandidateResult],
) -> None:
    optimizer._save_best_candidates(candidates)
    optimizer._refresh_run_indexes(candidates)
    write_post_eval_artifacts(
        run_dir=optimizer.run_dir,
        call_dir=None,
        iteration=iteration,
        candidates=evaluated,
        frontier_ids=optimizer._quality_frontier_ids(candidates),
    )


def evaluate_test_frontier(
    optimizer: SwebenchOptimizer,
    candidates: list[CandidateResult],
) -> dict[str, Any]:
    frontier = optimizer._quality_frontier(candidates)
    test_dir = RUN_DIR / "test_frontier"
    specs_dir = test_dir / "candidate_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    test_examples = load_swebench_instances(DATA_PATH, split="test", limit=0)
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
    for candidate in frontier:
        config = normalize_candidate_config(candidate.config)
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
        try:
            result = runner.evaluate_candidate(
                candidate=config,
                candidate_id=test_candidate_id,
                agent_name=str(config.get("agent_name") or DEFAULT_MINI_SWE_AGENT_NAME),
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
            continue
        test_results.append(result)
        rows.append(
            {
                "original_candidate": candidate.to_dict(),
                "candidate_spec_path": str(spec_path),
                "test_candidate": result.to_dict(),
            }
        )

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
        quality_gap_threshold=optimizer.config.pareto_quality_threshold,
    )
    summary = {
        "split": "test",
        "limit": 0,
        "count": len(test_examples),
        "train_frontier_count": len(frontier),
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
    return summary


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(PROJECT_ROOT)
    config = SwebenchOptimizerConfig(
        run_id=RUN_ID,
        out_dir=RUN_DIR,
        iterations=20,
        split="train",
        limit=TRAIN_LIMIT,
        data_path=DATA_PATH,
        mini_swe_agent_source_path=MINI_SOURCE,
        mini_swe_agent_command=RUN_COMMAND,
        mini_swe_agent_eval_command=EVAL_COMMAND,
        eval_timeout_s=TIMEOUT_S,
        proposer_agent="kimi",
        propose_timeout_s=2400,
        max_eval_workers=EVAL_WORKERS,
        skip_scaffold_eval=True,
        selection_policy="bandit",
        pareto_quality_threshold=0.125,
        proposer_sandbox="docker",
        proposer_docker_image="docker-claude-kimi:latest",
        proposer_docker_env=("KIMI_API_KEY",),
        proposer_docker_user="1023:1023",
        proposer_docker_home="/tmp",
        test_frontier=False,
    )
    optimizer = SwebenchOptimizer(config)
    examples = optimizer._load_examples()
    candidates = [item for item in load_candidates(optimizer) if keep_training_history(item)]
    write_iteration_artifacts(optimizer, 0, candidates, [])

    for iteration in (10, 11, 14):
        if any(candidate_iteration(item) == iteration for item in candidates):
            print(f"skip iter{iteration:03d}: result already present", flush=True)
            continue
        proposed = pending_candidates(iteration)
        if not proposed:
            print(f"skip iter{iteration:03d}: no pending candidate", flush=True)
            continue
        print(f"evaluating pending iter{iteration:03d}", flush=True)
        evaluated = optimizer._evaluate_proposed(iteration, proposed, examples)
        candidates.extend(normalize_candidate(item) for item in evaluated)
        write_iteration_artifacts(optimizer, iteration, candidates, evaluated)

    candidates = [item for item in candidates if keep_training_history(item)]
    write_iteration_artifacts(optimizer, 14, candidates, [])
    rebuild_bandit_state(optimizer, candidates, through_iteration=14)

    for iteration in (15, 16, 17, 18, 19, 20):
        if any(candidate_iteration(item) == iteration for item in candidates):
            print(f"skip iter{iteration:03d}: result already present", flush=True)
            continue
        print(f"running proposer/eval iter{iteration:03d}", flush=True)
        previous_best_passrate = optimizer._best_passrate(candidates)
        previous_best_quality = optimizer._best_quality_value(candidates)
        bandit_policy = optimizer._bandit_policy_for_workspace(
            iteration=iteration,
            candidates=candidates,
        )
        budget = str(bandit_policy.get("budget") or "low")
        evaluated = optimizer._run_progressive_proposer_iteration(
            iteration,
            candidates,
            examples,
            budget=budget,
            adaptive=True,
            selection_policy="bandit",
            bandit_policy=bandit_policy,
        )
        candidates.extend(normalize_candidate(item) for item in evaluated)
        write_iteration_artifacts(optimizer, iteration, candidates, evaluated)
        optimizer._update_bandit_state(
            iteration=iteration,
            previous_best_passrate=previous_best_passrate,
            previous_best_quality=previous_best_quality,
            evaluated=evaluated,
            call_dir=optimizer._iteration_dir(iteration),
        )

    optimizer_summary = {
        "run_id": RUN_ID,
        "out_dir": str(RUN_DIR),
        "iterations": 20,
        "candidate_count": len(candidates),
        "best_candidates_path": str(optimizer.frontier_path),
        "selection_policy": "bandit",
        "bandit_state_path": str(optimizer.bandit_state_path),
    }
    (RUN_DIR / "optimizer_summary.json").write_text(
        json.dumps(optimizer_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("running test frontier", flush=True)
    test_summary = evaluate_test_frontier(optimizer, candidates)
    optimizer_summary["test_frontier"] = test_summary
    (RUN_DIR / "optimizer_summary.json").write_text(
        json.dumps(optimizer_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(optimizer_summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
