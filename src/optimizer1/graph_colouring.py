"""Graph-colouring benchmark support — source-backed C++ heuristic mutation.

The proposer mutates the upstream graph-colouring repo (cloned under
references/vendor/graph-colouring/, see scripts/fetch_reference_repos.sh).
Each candidate edits files under src/algorithms/ and src/benchmark_runner.cpp
in its own workspace. Evaluation compiles the candidate's tree, runs
build/benchmark_runner under the configured target algorithm on a fixed
DIMACS subset, parses the per-instance CSV row, and aggregates two axes:

- passrate (high=good): mean(known_optimal / colors_used) over instances.
  Stays in (0, 1] for valid heuristics; lower colors_used pushes passrate up.
- token_consuming (low=good): int(round(mean(runtime_ms))).

The exact (colors_used, runtime_ms, known_optimal) per instance lives on each
TaskResult.metadata so the lex comparator (added in pareto.py) can rank
candidates by colors first and runtime as tiebreaker.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from optimizer1.graph_colouring_overlay import apply_overlay as apply_graph_colouring_overlay
from optimizer1.pareto import ParetoPoint, save_lex_frontier
from optimizer1.schemas import CandidateResult, TaskResult


DEFAULT_GRAPH_COLOURING_SOURCE_PATH = Path("references/vendor/graph-colouring")
DEFAULT_GRAPH_COLOURING_NAME = "graph_colouring_source"
DEFAULT_TARGET_ALGORITHM = "evolved"
DEFAULT_TIME_BUDGET_S = 30
DEFAULT_COMPILE_TIMEOUT_S = 120
RESULTS_CSV_HEADER = (
    "algorithm",
    "graph_name",
    "vertices",
    "edges",
    "colors_used",
    "known_optimal",
    "runtime_ms",
)

# Curated DIMACS subsets with known chromatic numbers. All files exist under
# data/dimacs/ in the upstream graph-colouring repo and have known_optimal
# entries in data/metadata-dimacs.csv. Sizes range from anna (138v) to
# le450_5a (450v) so a full eval pass stays within ~minutes.
DEFAULT_TRAIN_INSTANCES: tuple[str, ...] = (
    "anna.col",
    "myciel6.col",
    "queen8_8.col",
    "miles750.col",
    "mulsol.i.1.col",
    "zeroin.i.1.col",
    "fpsol2.i.1.col",
    "le450_5a.col",
)
DEFAULT_TEST_INSTANCES: tuple[str, ...] = (
    "myciel7.col",
    "queen9_9.col",
    "miles1000.col",
    "le450_5b.col",
    "flat300_20_0.col",
)


@dataclass(frozen=True)
class GraphInstance:
    """One DIMACS-format graph with its ground-truth chromatic number."""

    name: str  # filename without extension, e.g. "myciel6"
    file_name: str  # filename with extension, e.g. "myciel6.col"
    path: Path  # absolute path to the .col file
    vertices: int
    edges: int
    known_optimal: int  # chromatic number; must be > 0
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return self.name


@dataclass(frozen=True)
class GraphColouringRun:
    """One algorithm run on a single graph instance."""

    colors_used: int
    runtime_ms: float
    passed: bool  # True iff colors_used == known_optimal
    score: float  # known_optimal / colors_used, clamped to [0, 1]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_graph_colouring_instances(
    *,
    source_dir: Path,
    instance_names: tuple[str, ...] | None = None,
    split: str = "train",
) -> list[GraphInstance]:
    """Resolve a curated subset of DIMACS graphs against the upstream repo.

    Reads vertices/edges/known_optimal from data/metadata-dimacs.csv and
    resolves the .col path against data/dimacs/. Skips any name missing from
    metadata or absent on disk, raising on an empty result so the harness
    fails fast instead of evaluating against zero instances.
    """

    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(
            "graph-colouring source dir does not exist: "
            f"{source_dir} (run scripts/fetch_reference_repos.sh)"
        )
    metadata_csv = source_dir / "data" / "metadata-dimacs.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata CSV missing: {metadata_csv}")

    if instance_names is None:
        instance_names = (
            DEFAULT_TRAIN_INSTANCES if split == "train" else DEFAULT_TEST_INSTANCES
        )

    metadata = _read_metadata_csv(metadata_csv)
    dimacs_dir = source_dir / "data" / "dimacs"
    instances: list[GraphInstance] = []
    for file_name in instance_names:
        row = metadata.get(file_name)
        if row is None:
            continue
        path = dimacs_dir / file_name
        if not path.exists():
            continue
        try:
            vertices = int(row["vertices"])
            edges = int(row["edges"])
            known_optimal = int(row["known_optimal"])
        except (KeyError, ValueError):
            continue
        if known_optimal <= 0:
            continue
        instances.append(
            GraphInstance(
                name=Path(file_name).stem,
                file_name=file_name,
                path=path.resolve(),
                vertices=vertices,
                edges=edges,
                known_optimal=known_optimal,
                split=split,
                metadata={
                    "source": row.get("source", ""),
                    "graph_type": row.get("graph_type", ""),
                    "density": row.get("density", ""),
                },
            )
        )
    if not instances:
        raise ValueError(
            f"No graph instances resolved for split={split!r} from {dimacs_dir}. "
            "Check instance_names and metadata-dimacs.csv."
        )
    return instances


class GraphColouringSourceRunner:
    """Compile-and-run evaluator for source-backed graph-colouring candidates.

    Each candidate carries a source_project_path pointing at the workspace the
    proposer (or seed) wrote. The runner copies src/ + Makefile into a
    candidate-scoped workspace under {out_dir}/candidate_workspaces/<id>/,
    runs `make all`, then invokes build/benchmark_runner once per instance
    with --algorithm <target>, parses the appended CSV row, and aggregates.
    Compile/runtime failures degrade gracefully to worst-case scores so the
    Pareto frontier and lex comparator both stay well-defined.
    """

    def __init__(
        self,
        *,
        instances: list[GraphInstance],
        out_dir: Path,
        target_algorithm: str = DEFAULT_TARGET_ALGORITHM,
        time_budget_s: int = DEFAULT_TIME_BUDGET_S,
        compile_timeout_s: int = DEFAULT_COMPILE_TIMEOUT_S,
        max_eval_workers: int = 1,
        dry_run: bool = False,
        force: bool = False,
        default_source_path: Path = DEFAULT_GRAPH_COLOURING_SOURCE_PATH,
    ) -> None:
        self.instances = instances
        self.out_dir = Path(out_dir)
        self.target_algorithm = target_algorithm
        self.time_budget_s = max(1, int(time_budget_s))
        self.compile_timeout_s = max(1, int(compile_timeout_s))
        self.max_eval_workers = max(1, int(max_eval_workers))
        self.dry_run = dry_run
        self.force = force
        self.default_source_path = Path(default_source_path)

    def evaluate_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        agent_name: str = DEFAULT_GRAPH_COLOURING_NAME,
    ) -> CandidateResult:
        candidate_dir = self.out_dir / "candidate_results"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        result_path = candidate_dir / f"{candidate_id}.json"
        if not self.force:
            existing = _load_candidate_result(
                result_path,
                candidate_id=candidate_id,
                agent_name=agent_name,
                config=dict(candidate),
            )
            if existing is not None:
                return existing

        source_path = _candidate_source_path(candidate) or self.default_source_path
        if not source_path.exists():
            raise FileNotFoundError(
                f"graph-colouring source path does not exist: {source_path}"
            )

        workspace = self.out_dir / "candidate_workspaces" / candidate_id
        compile_error = ""
        if self.dry_run:
            task_results = [
                self._build_dry_run_task(instance) for instance in self.instances
            ]
        else:
            workspace, compile_error = self._prepare_workspace(
                source_path=source_path, workspace=workspace
            )
            if compile_error:
                task_results = [
                    self._build_failed_task(
                        instance, reason="compile_failed", detail=compile_error
                    )
                    for instance in self.instances
                ]
            else:
                task_results = self._run_instances(
                    workspace=workspace, candidate_id=candidate_id
                )

        count = len(task_results)
        passrate = (
            sum(item.score for item in task_results) / count if count else 0.0
        )
        average_score = passrate
        runtime_total = sum(
            float(item.metadata.get("runtime_ms", 0.0)) for item in task_results
        )
        token_consuming = int(round(runtime_total / count)) if count else 0
        result = CandidateResult(
            candidate_id=candidate_id,
            scaffold_name=agent_name,
            passrate=passrate,
            average_score=average_score,
            token_consuming=token_consuming,
            avg_token_consuming=float(token_consuming),
            avg_prompt_tokens=0.0,
            avg_completion_tokens=0.0,
            count=count,
            config=dict(candidate),
            result_path=str(result_path),
        )
        payload = {
            "candidate": result.to_dict(),
            "tasks": [item.to_dict() for item in task_results],
            "score_breakdown": _score_breakdown(task_results),
            "compile_error": compile_error,
            "target_algorithm": self.target_algorithm,
            "time_budget_s": self.time_budget_s,
        }
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    def _prepare_workspace(
        self, *, source_path: Path, workspace: Path
    ) -> tuple[Path, str]:
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        src_dir = source_path / "src"
        makefile = source_path / "Makefile"
        if not src_dir.is_dir():
            return workspace, f"missing src/ at {source_path}"
        if not makefile.exists():
            return workspace, f"missing Makefile at {source_path}"
        shutil.copytree(src_dir, workspace / "src", symlinks=False)
        shutil.copy2(makefile, workspace / "Makefile")

        # Inject the `evolved` algorithm slot used by the eval harness. The
        # overlay is idempotent, so it is also safe to apply on top of a
        # workspace that already received the overlay (e.g. a snapshot the
        # proposer copied forward from a previous iteration).
        try:
            apply_graph_colouring_overlay(workspace)
        except Exception as exc:  # noqa: BLE001 - propagate as compile failure
            return workspace, f"overlay failed: {exc}"

        try:
            completed = subprocess.run(
                ["make", "all"],
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=self.compile_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return workspace, _truncate(_timeout_output_to_text(exc.stderr))
        (workspace / "build_stdout.txt").write_text(
            completed.stdout or "", encoding="utf-8"
        )
        (workspace / "build_stderr.txt").write_text(
            completed.stderr or "", encoding="utf-8"
        )
        if completed.returncode != 0:
            return workspace, _truncate(completed.stderr or completed.stdout or "")
        runner_bin = workspace / "build" / "benchmark_runner"
        if not runner_bin.exists():
            return workspace, "make all succeeded but build/benchmark_runner missing"
        return workspace, ""

    def _run_instances(
        self, *, workspace: Path, candidate_id: str
    ) -> list[TaskResult]:
        runner_bin = (workspace / "build" / "benchmark_runner").resolve()
        task_root = self.out_dir / "agent_runs" / candidate_id
        task_root.mkdir(parents=True, exist_ok=True)

        def _one(instance: GraphInstance) -> TaskResult:
            task_dir = task_root / instance.task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            results_csv = task_dir / "results.csv"
            coloring_out = task_dir / f"{instance.name}.col"
            if results_csv.exists():
                results_csv.unlink()
            run = self._run_one_instance(
                runner_bin=runner_bin,
                workspace=workspace,
                instance=instance,
                results_csv=results_csv,
                coloring_out=coloring_out,
                task_dir=task_dir,
            )
            return _task_result_from_run(instance, run)

        if self.max_eval_workers == 1 or len(self.instances) <= 1:
            return [_one(instance) for instance in self.instances]
        workers = min(self.max_eval_workers, len(self.instances))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_one, self.instances))

    def _run_one_instance(
        self,
        *,
        runner_bin: Path,
        workspace: Path,
        instance: GraphInstance,
        results_csv: Path,
        coloring_out: Path,
        task_dir: Path,
    ) -> GraphColouringRun:
        # The subprocess runs with cwd=workspace, so any relative path in the
        # CLI args resolves against the workspace tree. Force absolute paths
        # for outputs the harness owns; instance.path is already resolved.
        coloring_out_abs = coloring_out.resolve()
        results_csv_abs = results_csv.resolve()
        cmd = [
            str(runner_bin),
            "--algorithm",
            self.target_algorithm,
            "--input",
            str(instance.path),
            "--graph-name",
            instance.name,
            "--output",
            str(coloring_out_abs),
            "--results",
            str(results_csv_abs),
        ]
        started = time.time()
        try:
            completed = subprocess.run(
                cmd,
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=self.time_budget_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            (task_dir / "stdout.txt").write_text(
                _timeout_output_to_text(exc.stdout), encoding="utf-8"
            )
            (task_dir / "stderr.txt").write_text(
                _timeout_output_to_text(exc.stderr), encoding="utf-8"
            )
            return _worst_case_run(
                instance,
                reason="runner_timeout",
                detail=f"timeout after {self.time_budget_s}s",
                duration_s=time.time() - started,
            )
        duration_s = time.time() - started
        (task_dir / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
        (task_dir / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            return _worst_case_run(
                instance,
                reason="runner_nonzero_exit",
                detail=_truncate(completed.stderr or completed.stdout or ""),
                duration_s=duration_s,
                returncode=completed.returncode,
            )
        row = _parse_last_csv_row(results_csv)
        if row is None:
            return _worst_case_run(
                instance,
                reason="csv_missing",
                detail=f"benchmark_runner exit 0 but {results_csv.name} unparseable",
                duration_s=duration_s,
                returncode=completed.returncode,
            )
        try:
            colors_used = int(row["colors_used"])
            runtime_ms = float(row["runtime_ms"])
        except (KeyError, ValueError) as exc:
            return _worst_case_run(
                instance,
                reason="csv_malformed",
                detail=f"row parse error: {exc}; row={row!r}",
                duration_s=duration_s,
                returncode=completed.returncode,
            )
        if colors_used <= 0:
            return _worst_case_run(
                instance,
                reason="invalid_colors",
                detail=f"colors_used={colors_used} <= 0",
                duration_s=duration_s,
                returncode=completed.returncode,
            )
        score = min(1.0, instance.known_optimal / colors_used)
        return GraphColouringRun(
            colors_used=colors_used,
            runtime_ms=runtime_ms,
            passed=colors_used == instance.known_optimal,
            score=score,
            metadata={
                "benchmark": "graph_colouring",
                "algorithm": self.target_algorithm,
                "graph_name": instance.name,
                "vertices": instance.vertices,
                "edges": instance.edges,
                "known_optimal": instance.known_optimal,
                "colors_used": colors_used,
                "runtime_ms": runtime_ms,
                "duration_s": duration_s,
                "returncode": completed.returncode,
                "reason": "ok",
            },
        )

    def _build_dry_run_task(self, instance: GraphInstance) -> TaskResult:
        run = GraphColouringRun(
            colors_used=instance.known_optimal,
            runtime_ms=0.0,
            passed=True,
            score=1.0,
            metadata={
                "benchmark": "graph_colouring",
                "algorithm": self.target_algorithm,
                "graph_name": instance.name,
                "vertices": instance.vertices,
                "edges": instance.edges,
                "known_optimal": instance.known_optimal,
                "colors_used": instance.known_optimal,
                "runtime_ms": 0.0,
                "dry_run": True,
                "reason": "dry_run",
            },
        )
        return _task_result_from_run(instance, run)

    def _build_failed_task(
        self, instance: GraphInstance, *, reason: str, detail: str
    ) -> TaskResult:
        run = _worst_case_run(instance, reason=reason, detail=detail)
        return _task_result_from_run(instance, run)


def run_graph_colouring_frontier(
    *,
    out_dir: Path,
    source_path: Path | None = None,
    instance_names: tuple[str, ...] | None = None,
    split: str = "train",
    target_algorithm: str = DEFAULT_TARGET_ALGORITHM,
    time_budget_s: int = DEFAULT_TIME_BUDGET_S,
    compile_timeout_s: int = DEFAULT_COMPILE_TIMEOUT_S,
    max_eval_workers: int = 1,
    dry_run: bool = False,
    force: bool = False,
    pareto_quality_threshold: float = 0.05,
) -> dict[str, object]:
    """Evaluate the upstream graph-colouring repo as the seed Pareto point."""

    resolved_source = (
        Path(source_path).expanduser()
        if source_path is not None
        else DEFAULT_GRAPH_COLOURING_SOURCE_PATH
    )
    instances = load_graph_colouring_instances(
        source_dir=resolved_source,
        instance_names=instance_names,
        split=split,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate: dict[str, Any] = {
        "name": DEFAULT_GRAPH_COLOURING_NAME,
        "agent_name": DEFAULT_GRAPH_COLOURING_NAME,
        "source_project_path": str(resolved_source),
        "target_algorithm": target_algorithm,
    }
    runner = GraphColouringSourceRunner(
        instances=instances,
        out_dir=out_dir,
        target_algorithm=target_algorithm,
        time_budget_s=time_budget_s,
        compile_timeout_s=compile_timeout_s,
        max_eval_workers=max_eval_workers,
        dry_run=dry_run,
        force=force,
        default_source_path=resolved_source,
    )
    result = runner.evaluate_candidate(
        candidate=candidate,
        candidate_id=DEFAULT_GRAPH_COLOURING_NAME,
        agent_name=DEFAULT_GRAPH_COLOURING_NAME,
    )
    frontier_path = out_dir / "pareto_frontier.json"
    save_lex_frontier(
        frontier_path,
        [
            ParetoPoint(
                candidate_id=result.candidate_id,
                scaffold_name=result.scaffold_name,
                passrate=result.passrate,
                token_consuming=result.token_consuming,
                avg_token_consuming=result.avg_token_consuming,
                average_score=result.average_score,
                result_path=result.result_path,
                config=result.config,
            )
        ],
    )
    del pareto_quality_threshold  # graph-colouring frontier uses lex tiebreaker
    summary = {
        "benchmark": "graph_colouring",
        "target_system": DEFAULT_GRAPH_COLOURING_NAME,
        "target_algorithm": target_algorithm,
        "split": split,
        "count": len(instances),
        "instance_names": [inst.file_name for inst in instances],
        "dry_run": dry_run,
        "force": force,
        "candidate_count": 1,
        "candidates": [result.to_dict()],
        "pareto_frontier_path": str(frontier_path),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _read_metadata_csv(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("graph_name") or "").strip()
            if not name:
                continue
            rows[name] = {k: (v or "").strip() for k, v in row.items()}
    return rows


def _parse_last_csv_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    reader = csv.DictReader(text.splitlines())
    last: dict[str, str] | None = None
    for row in reader:
        last = {k: (v or "").strip() for k, v in row.items() if k}
    return last


def _candidate_source_path(candidate: Mapping[str, Any]) -> Path | None:
    extra = candidate.get("extra") if isinstance(candidate.get("extra"), Mapping) else {}
    for key in (
        "source_project_path",
        "graph_colouring_source_path",
        "upstream_source_path",
    ):
        value = candidate.get(key) or (extra.get(key) if extra else None)
        if value:
            return Path(str(value)).expanduser()
    return None


def _task_result_from_run(
    instance: GraphInstance, run: GraphColouringRun
) -> TaskResult:
    prediction = (
        f"colors_used={run.colors_used}, runtime_ms={run.runtime_ms:.3f}"
    )
    return TaskResult(
        task_id=instance.task_id,
        question=f"{instance.name}@{instance.vertices}v/{instance.edges}e",
        gold_answer=str(instance.known_optimal),
        prediction=prediction,
        score=run.score,
        passed=run.passed,
        prompt_tokens=0,
        completion_tokens=0,
        retrieved=[],
        metadata=run.metadata,
    )


def _worst_case_run(
    instance: GraphInstance,
    *,
    reason: str,
    detail: str = "",
    duration_s: float | None = None,
    returncode: int | None = None,
) -> GraphColouringRun:
    """Penalise compile/timeout/crash by assigning the worst feasible colouring."""

    return GraphColouringRun(
        colors_used=instance.vertices,
        runtime_ms=float(instance.vertices),
        passed=False,
        score=instance.known_optimal / max(1, instance.vertices),
        metadata={
            "benchmark": "graph_colouring",
            "graph_name": instance.name,
            "vertices": instance.vertices,
            "edges": instance.edges,
            "known_optimal": instance.known_optimal,
            "colors_used": instance.vertices,
            "runtime_ms": float(instance.vertices),
            "duration_s": duration_s,
            "returncode": returncode,
            "reason": reason,
            "detail": detail,
        },
    )


def _score_breakdown(task_results: list[TaskResult]) -> dict[str, dict[str, object]]:
    if not task_results:
        return {"all": {"count": 0, "passrate": 0.0, "average_score": 0.0}}
    count = len(task_results)
    return {
        "all": {
            "count": count,
            "passrate": sum(item.score for item in task_results) / count,
            "average_score": sum(item.score for item in task_results) / count,
            "optimal_hits": sum(1 for item in task_results if item.passed),
            "mean_colors_used": sum(
                float(item.metadata.get("colors_used", 0)) for item in task_results
            )
            / count,
            "mean_runtime_ms": sum(
                float(item.metadata.get("runtime_ms", 0.0)) for item in task_results
            )
            / count,
        }
    }


def _truncate(text: str, *, limit: int = 2000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _timeout_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _load_candidate_result(
    result_path: Path,
    *,
    candidate_id: str,
    agent_name: str,
    config: dict[str, Any],
) -> CandidateResult | None:
    if not result_path.exists():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        candidate = CandidateResult.from_dict(payload["candidate"])
    except Exception:
        return None
    if (
        candidate.candidate_id != candidate_id
        or candidate.scaffold_name != agent_name
        or candidate.config != config
    ):
        return None
    return candidate
