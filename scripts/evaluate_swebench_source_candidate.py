#!/usr/bin/env python3
"""Evaluate one source-backed mini-SWE-agent candidate on local SWE-bench rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimizer1.swebench import (
    DEFAULT_MINI_SWE_AGENT_NAME,
    MiniSweAgentSourceRunner,
    load_swebench_instances,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--step-limit", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_path = args.source_path.resolve()
    root = Path(__file__).resolve().parents[1]
    command = (
        f"python {root / 'scripts' / 'run_miniswe_swebench_single.py'} run "
        "--source-path {source_path} "
        "--instance-path {instance_path} "
        "--patch-path {patch_path} "
        "--task-dir {task_dir} "
        f"--model {args.model} "
        f"--base-url {args.base_url} "
        f"--step-limit {args.step_limit} "
        f"--max-tokens {args.max_tokens}"
    )
    eval_command = (
        f"python {root / 'scripts' / 'run_miniswe_swebench_single.py'} eval "
        "--source-path {source_path} "
        "--instance-path {instance_path} "
        "--patch-path {patch_path} "
        "--task-dir {task_dir}"
    )
    candidate = {
        "name": args.candidate_id,
        "agent_name": DEFAULT_MINI_SWE_AGENT_NAME,
        "source_project_path": str(source_path),
        "command": command,
        "eval_command": eval_command,
        "model": args.model,
        "base_url": args.base_url,
    }
    instances = load_swebench_instances(args.data_path, split=args.split, limit=args.limit)
    runner = MiniSweAgentSourceRunner(
        instances=instances,
        out_dir=args.out,
        timeout_s=args.timeout_s,
        max_eval_workers=args.eval_workers,
        force=args.force,
    )
    result = runner.evaluate_candidate(
        candidate=candidate,
        candidate_id=args.candidate_id,
        agent_name=DEFAULT_MINI_SWE_AGENT_NAME,
    )
    payload = result.to_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
