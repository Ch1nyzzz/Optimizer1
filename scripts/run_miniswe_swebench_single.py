#!/usr/bin/env python3
"""Run or evaluate one mini-SWE-agent SWE-bench instance.

This bridges MemoMemo's per-instance optimizer runner with mini-SWE-agent's
batch-oriented SWE-bench CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "openai/Qwen3.5-35B-A3B-FP8"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL_CLASS = "qwen35_miniswe_model.Qwen35TextModel"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "eval"))
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--instance-path", required=True, type=Path)
    parser.add_argument("--patch-path", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the model API key.",
    )
    parser.add_argument("--step-limit", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    instance = json.loads(args.instance_path.read_text(encoding="utf-8"))
    instance_id = str(instance.get("task_id") or instance.get("instance_id") or "").strip()
    if not instance_id:
        raise SystemExit("instance_path is missing task_id/instance_id")

    if args.mode == "run":
        return run_agent(args, root=root, instance_id=instance_id)
    return eval_patch(args, root=root, instance_id=instance_id)


def run_agent(args: argparse.Namespace, *, root: Path, instance_id: str) -> int:
    source_path = args.source_path.resolve()
    output = args.task_dir / "miniswe_run"
    output.mkdir(parents=True, exist_ok=True)
    args.patch_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_paths(
        env.get("PYTHONPATH", ""),
        [str(root), str(args.source_path / "src")],
    )
    env["MSWEA_COST_TRACKING"] = "ignore_errors"

    cmd = [
        "uvx",
        "--from",
        str(source_path),
        "python",
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        "verified",
        "--split",
        "test",
        "--filter",
        f"^{re.escape(instance_id)}$",
        "--output",
        str(output),
        "--workers",
        "1",
        "--model",
        args.model,
        "--model-class",
        DEFAULT_MODEL_CLASS,
        "--config",
        "swebench_backticks.yaml",
        "--config",
        f"model.model_kwargs.api_base={args.base_url}",
        "--config",
        "model.model_kwargs.temperature=0",
        "--config",
        f"model.model_kwargs.max_tokens={args.max_tokens}",
        "--config",
        "agent.cost_limit=0",
        "--config",
        f"agent.step_limit={args.step_limit}",
        "--redo-existing",
    ]
    api_key = args.api_key
    if not api_key and args.api_key_env:
        api_key = env.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"{args.api_key_env} is not set")
    if api_key:
        cmd.extend(["--config", f"model.model_kwargs.api_key={api_key}"])
    completed = subprocess.run(
        cmd,
        cwd=args.source_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (args.task_dir / "miniswe_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.task_dir / "miniswe_stderr.txt").write_text(completed.stderr, encoding="utf-8")

    patch = ""
    preds_path = output / "preds.json"
    if preds_path.exists():
        preds = json.loads(preds_path.read_text(encoding="utf-8"))
        prediction = preds.get(instance_id) or {}
        if isinstance(prediction, dict):
            patch = str(prediction.get("model_patch") or "")
        else:
            patch = str(prediction or "")
    args.patch_path.write_text(patch, encoding="utf-8")
    return completed.returncode


def eval_patch(args: argparse.Namespace, *, root: Path, instance_id: str) -> int:
    args.task_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.task_dir / "single_pred.json"
    pred_path.write_text(
        json.dumps(
            {
                instance_id: {
                    "model_name_or_path": "memomemo_candidate",
                    "instance_id": instance_id,
                    "model_patch": args.patch_path.read_text(encoding="utf-8", errors="ignore"),
                }
            }
        ),
        encoding="utf-8",
    )
    report_id = f"memomemo_{_safe_id(instance_id)}"
    cmd = [
        "uvx",
        "--from",
        "swebench",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "-d",
        "princeton-nlp/SWE-Bench_Verified",
        "-s",
        "test",
        "-i",
        instance_id,
        "-p",
        str(pred_path),
        "--max_workers",
        "1",
        "--cache_level",
        "instance",
        "--clean",
        "True",
        "-id",
        report_id,
        "--report_dir",
        str(args.task_dir / "eval_report"),
    ]
    completed = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    (args.task_dir / "official_eval_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.task_dir / "official_eval_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    report = _find_report(root, args.task_dir, report_id)
    if report is None:
        return 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    return 0 if int(payload.get("resolved_instances") or 0) == 1 else 1


def _find_report(root: Path, task_dir: Path, report_id: str) -> Path | None:
    candidates = [
        *task_dir.glob(f"**/*.{report_id}.json"),
        *root.glob(f"*.{report_id}.json"),
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _prepend_paths(existing: str, paths: list[str]) -> str:
    values = [item for item in paths if item]
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    sys.exit(main())
