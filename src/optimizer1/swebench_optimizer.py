"""SWE-bench optimization entry point for source-backed coding agents."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# When mirroring the repo-root eval-gate script into a candidate snapshot we
# prepend this banner so the proposer sees a clear note that edits are inert.
EVAL_SCRIPT_BANNER = (
    "# READ-ONLY REFERENCE — DO NOT EDIT.\n"
    "# This file is the SWE-bench eval gate. The real grading subprocess runs\n"
    "# the trusted copy at <repo>/scripts/run_miniswe_swebench_single.py via\n"
    "# an absolute path. Edits to this in-candidate copy have no effect and\n"
    "# will be flagged as a reward_hack_attempt in candidate_score_table.\n"
    "# To improve agent behaviour, edit files under src/minisweagent/* and\n"
    "# express your contract in pending_eval.json — do not touch this script.\n"
    "\n"
)


def _eval_script_repo_path(project_root: Path) -> Path:
    return project_root / "scripts" / "run_miniswe_swebench_single.py"


def _eval_script_sha256(text: str) -> str:
    """Hash the eval script after removing any banner we may have injected.

    Strips a leading occurrence of ``EVAL_SCRIPT_BANNER`` so the hash compares
    apples to apples regardless of whether the file we are looking at has
    been mirrored from the trusted source (banner injected) or is the
    pristine trusted source itself.
    """

    payload = text
    if payload.startswith(EVAL_SCRIPT_BANNER):
        payload = payload[len(EVAL_SCRIPT_BANNER):]
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mirror_eval_script_into_candidate(
    *,
    candidate_mini_root: Path,
    project_root: Path,
) -> None:
    """Force-mirror the trusted eval script into a candidate snapshot.

    The trusted file lives at ``<repo>/scripts/run_miniswe_swebench_single.py``.
    The on-disk copy under the candidate
    (``<candidate>/scripts/run_miniswe_swebench_single.py``) is a read-only
    reference; the grading subprocess never uses it because eval-command
    invocation is rewritten to the absolute repo-root path. We always create
    ``scripts/`` (even if the vendored mini-swe-agent has no such directory)
    so the trusted copy is in place before the proposer's session starts —
    if the proposer later overwrites it, sha256 detection will flag it.

    No-op when ``candidate_mini_root`` itself does not exist (the upstream
    mini-swe-agent copy wasn't materialized for this candidate).
    """

    if not candidate_mini_root.exists():
        return
    trusted = _eval_script_repo_path(project_root)
    if not trusted.is_file():
        # Defensive — repo missing the eval script is a setup error, not a
        # snapshot-build error. Surface it explicitly.
        raise FileNotFoundError(
            "Trusted eval gate script is missing at "
            f"{trusted}. Cannot mirror into candidate snapshot."
        )
    target_dir = candidate_mini_root / "scripts"
    target_dir.mkdir(parents=True, exist_ok=True)
    trusted_text = trusted.read_text(encoding="utf-8")
    target = target_dir / "run_miniswe_swebench_single.py"
    target.write_text(EVAL_SCRIPT_BANNER + trusted_text, encoding="utf-8")


def detect_eval_script_tampering(
    candidate_mini_root: Path,
    project_root: Path,
) -> bool:
    """Return True iff the candidate's eval-script copy differs from trusted.

    Compares the candidate copy (banner-stripped) against the repo-root truth
    by sha256. Missing candidate copy is treated as tampering — the snapshot
    builder always mirrors a banner-prefixed copy, so absence means a proposer
    deleted or moved it.
    """

    trusted = _eval_script_repo_path(project_root)
    candidate_copy = candidate_mini_root / "scripts" / "run_miniswe_swebench_single.py"
    if not trusted.is_file():
        return False
    if not candidate_copy.is_file():
        return True
    try:
        trusted_hash = _eval_script_sha256(trusted.read_text(encoding="utf-8"))
        candidate_hash = _eval_script_sha256(
            candidate_copy.read_text(encoding="utf-8", errors="replace")
        )
    except OSError:
        return True
    return trusted_hash != candidate_hash

from optimizer1.benchmark_workspaces import BenchmarkWorkspaceSpec, SWEBENCH_WORKSPACE_SPEC
from optimizer1.optimizer import LocomoOptimizer, OptimizerConfig
from optimizer1.pareto import ParetoPoint, save_frontier
from optimizer1.schemas import CandidateResult
from optimizer1.swebench import (
    DEFAULT_MINI_SWE_AGENT_NAME,
    DEFAULT_MINI_SWE_AGENT_SOURCE_PATH,
    MiniSweAgentSourceRunner,
    SwebenchInstance,
    load_swebench_instances,
    run_swebench_frontier,
)


@dataclass(frozen=True)
class SwebenchOptimizerConfig(OptimizerConfig):
    """Configuration for source-backed mini-SWE-agent optimization."""

    data_path: Path | None = None
    mini_swe_agent_source_path: Path = DEFAULT_MINI_SWE_AGENT_SOURCE_PATH
    mini_swe_agent_command: str = ""
    mini_swe_agent_eval_command: str = ""
    force: bool = False
    scaffolds: tuple[str, ...] = (DEFAULT_MINI_SWE_AGENT_NAME,)
    progressive_target_system: str = DEFAULT_MINI_SWE_AGENT_NAME


class SwebenchOptimizer(LocomoOptimizer):
    """Proposer loop for SWE-bench-style coding-agent candidates."""

    workspace_spec: BenchmarkWorkspaceSpec = SWEBENCH_WORKSPACE_SPEC
    config: SwebenchOptimizerConfig

    def __init__(self, config: SwebenchOptimizerConfig) -> None:
        super().__init__(config)

    def _load_examples(self) -> list[SwebenchInstance]:
        return load_swebench_instances(
            self.config.data_path,
            split=self.config.split,
            limit=self.config.limit,
        )

    def _run_seed_frontier(self) -> dict[str, Any]:
        return run_swebench_frontier(
            out_dir=self.run_dir,
            data_path=self.config.data_path,
            split=self.config.split,
            limit=self.config.limit,
            source_project_path=self.config.mini_swe_agent_source_path,
            command=self.config.mini_swe_agent_command,
            eval_command=self.config.mini_swe_agent_eval_command,
            timeout_s=self.config.eval_timeout_s,
            max_eval_workers=self.config.max_eval_workers,
            dry_run=self.config.dry_run,
            force=self.config.force,
            pareto_quality_threshold=self.config.pareto_quality_threshold,
        )

    def _benchmark_prompt_name(self) -> str:
        return "SWE-bench coding-agent issue resolution"

    def _raw_data_policy_name(self) -> str:
        return "SWE-bench gold patches, test patches, and evaluation results"

    def _run_test_frontier(self, candidates: list[CandidateResult]) -> dict[str, Any]:
        full_frontier = self._quality_frontier(candidates)
        candidate_limit = max(0, int(self.config.test_frontier_candidate_limit or 0))
        frontier = full_frontier[:candidate_limit] if candidate_limit else full_frontier
        test_dir = self.run_dir / "test_frontier"
        specs_dir = test_dir / "candidate_specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        examples = load_swebench_instances(
            self.config.data_path,
            split=self.config.test_split,
            limit=self.config.test_limit,
        )
        runner = MiniSweAgentSourceRunner(
            instances=examples,
            out_dir=test_dir,
            timeout_s=self.config.eval_timeout_s,
            max_eval_workers=self.config.max_eval_workers,
            dry_run=self.config.dry_run,
            force=self.config.force,
            project_root=self.project_root,
        )

        rows: list[dict[str, Any]] = []
        test_results: list[CandidateResult] = []
        failures: list[dict[str, Any]] = []
        for candidate in frontier:
            spec = self._swebench_test_spec(candidate)
            spec_path = specs_dir / f"{spec['candidate_id']}.json"
            spec_path.write_text(
                json.dumps(spec, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            try:
                result = runner.evaluate_candidate(
                    candidate=spec,
                    candidate_id=str(spec["candidate_id"]),
                    agent_name=str(spec.get("agent_name") or DEFAULT_MINI_SWE_AGENT_NAME),
                )
            except Exception as exc:  # noqa: BLE001 - keep testing the rest of the frontier
                failure = {
                    "original_candidate_id": candidate.candidate_id,
                    "test_candidate_id": spec["candidate_id"],
                    "candidate_spec_path": str(spec_path),
                    "error": str(exc),
                }
                failures.append(failure)
                rows.append(
                    {
                        "original_candidate": candidate.to_dict(),
                        "candidate_spec_path": str(spec_path),
                        "error": str(exc),
                    }
                )
                self._append_event({"event": "test_frontier_candidate_failed", **failure})
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
        results_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
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
            quality_gap_threshold=self.config.pareto_quality_threshold,
        )
        summary = {
            "benchmark": "swebench",
            "split": self.config.test_split,
            "limit": self.config.test_limit,
            "count": len(examples),
            "train_frontier_total_count": len(full_frontier),
            "candidate_limit": candidate_limit,
            "train_frontier_count": len(frontier),
            "evaluated_count": len(test_results),
            "failed_count": len(failures),
            "test_dir": str(test_dir),
            "test_results_path": str(results_path),
            "test_pareto_frontier_path": str(test_frontier_path),
            "candidate_spec_dir": str(specs_dir),
            "failures": failures,
        }
        summary_path = self.run_dir / "test_frontier_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_path)
        return summary

    def _swebench_test_spec(self, candidate: CandidateResult) -> dict[str, Any]:
        config = dict(candidate.config) if isinstance(candidate.config, dict) else {}
        extra = config.get("extra") if isinstance(config.get("extra"), dict) else {}
        spec = dict(config)
        spec["candidate_id"] = self._test_candidate_id(candidate.candidate_id)
        spec["original_candidate_id"] = candidate.candidate_id
        spec["agent_name"] = str(
            spec.get("agent_name")
            or spec.get("scaffold_name")
            or candidate.scaffold_name
            or DEFAULT_MINI_SWE_AGENT_NAME
        )
        spec["scaffold_name"] = DEFAULT_MINI_SWE_AGENT_NAME
        spec["source_family"] = DEFAULT_MINI_SWE_AGENT_NAME
        if "name" not in spec:
            spec["name"] = candidate.scaffold_name or DEFAULT_MINI_SWE_AGENT_NAME
        if "source_project_path" not in spec and extra.get("source_project_path"):
            spec["source_project_path"] = str(extra["source_project_path"])
        if "source_project_path" not in spec:
            spec["source_project_path"] = str(self.config.mini_swe_agent_source_path)
        spec["command"] = self.config.mini_swe_agent_command
        spec["eval_command"] = self.config.mini_swe_agent_eval_command
        return spec

    def _evaluate_proposed(
        self,
        iteration: int,
        proposed: list[dict[str, Any]],
        examples: list[SwebenchInstance],
    ) -> list[CandidateResult]:
        runner = MiniSweAgentSourceRunner(
            instances=examples,
            out_dir=self.run_dir,
            timeout_s=self.config.eval_timeout_s,
            max_eval_workers=self.config.max_eval_workers,
            dry_run=self.config.dry_run,
            force=self.config.force,
            project_root=self.project_root,
        )
        results: list[CandidateResult] = []
        for raw in proposed:
            if not isinstance(raw, dict):
                continue
            candidate = dict(raw)
            agent_name = str(
                candidate.get("agent_name")
                or candidate.get("scaffold_name")
                or DEFAULT_MINI_SWE_AGENT_NAME
            )
            candidate.setdefault("agent_name", agent_name)
            candidate.setdefault("source_family", DEFAULT_MINI_SWE_AGENT_NAME)
            self._normalize_candidate_source_project_path(candidate)
            candidate["command"] = self.config.mini_swe_agent_command
            candidate["eval_command"] = self.config.mini_swe_agent_eval_command

            violations = self._candidate_code_policy_violations(candidate)
            if violations:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_policy_rejected",
                        "candidate": candidate,
                        "violations": violations,
                    }
                )
                continue

            candidate_name = str(candidate.get("name") or agent_name)
            candidate_id = f"iter{iteration:03d}_{candidate_name}"
            try:
                result = runner.evaluate_candidate(
                    candidate=candidate,
                    candidate_id=candidate_id,
                    agent_name=agent_name,
                )
            except Exception as exc:  # noqa: BLE001 - log and continue
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_eval_failed",
                        "candidate": candidate,
                        "candidate_id": candidate_id,
                        "error": str(exc),
                    }
                )
                continue
            self._tag_reward_hack_attempt(
                iteration=iteration,
                candidate=candidate,
                candidate_id=candidate_id,
                result=result,
            )
            results.append(result)
            self._append_summary(iteration=iteration, candidate=result, proposal=candidate)
        return results

    def _tag_reward_hack_attempt(
        self,
        *,
        iteration: int,
        candidate: dict[str, Any],
        candidate_id: str,
        result: CandidateResult,
    ) -> None:
        """Audit (don't punish) candidates that tampered with the eval script,
        and immediately reset the in-snapshot copy to the trusted version so
        the next iteration's parent-snapshot copy starts clean.

        Sets ``result.config["reward_hack_attempt"] = True`` and logs an
        evolution event. ``passrate`` stays untouched — the grading
        subprocess always invokes the trusted repo-root copy via an
        absolute path, so tampered candidate copies could not have biased
        the score in the first place. The flag is a signal for downstream
        analysis that the proposer attempted to game the gate.

        After flagging, the tampered file is overwritten with the trusted
        banner-prefixed version. This breaks the implicit-contagion path
        where a later curaii-style iteration copies the parent snapshot's
        scripts/ directory and inherits a hacked eval script.
        """

        mini_root = self._candidate_mini_source_path(candidate)
        if mini_root is None:
            return
        tampered = detect_eval_script_tampering(mini_root, self.project_root)
        # result.config is a dict (CandidateResult is frozen, but dicts mutate).
        result.config["reward_hack_attempt"] = bool(tampered)
        if not tampered:
            return
        eval_script = mini_root / "scripts" / "run_miniswe_swebench_single.py"
        logger.warning(
            "iter %d candidate %s tampered with eval script at %s; resetting",
            iteration,
            candidate_id,
            eval_script,
        )
        # Overwrite back to the trusted copy. Best-effort: a write failure
        # here is logged but does not propagate (eval already completed and
        # passrate is unaffected; worst case the next iteration's copy
        # still sees the tampered version and gets re-flagged + reset).
        try:
            _mirror_eval_script_into_candidate(
                candidate_mini_root=mini_root,
                project_root=self.project_root,
            )
        except OSError as exc:
            logger.warning(
                "failed to reset tampered eval script for %s: %s",
                candidate_id,
                exc,
            )
        self._append_event(
            {
                "iteration": iteration,
                "event": "candidate_reward_hack_attempt",
                "candidate_id": candidate_id,
                "candidate_mini_root": str(mini_root),
                "note": (
                    "candidate edited the in-snapshot copy of the eval gate; "
                    "grading uses the trusted repo-root path, so passrate is "
                    "unaffected. flagged for audit and the copy has been "
                    "overwritten back to the trusted version."
                ),
            }
        )

    def _normalize_candidate_source_project_path(self, candidate: dict[str, Any]) -> None:
        """Keep proposer-edited mini-SWE-agent snapshots ahead of the default source."""

        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        if candidate.get("source_project_path"):
            return
        for key in ("source_project_path", "upstream_source_path", "mini_swe_agent_source_path"):
            if extra.get(key):
                candidate["source_project_path"] = str(extra[key])
                return
        candidate["source_project_path"] = str(self.config.mini_swe_agent_source_path)

    def _copy_upstream_source_context(self, source_family: str, dest_dir: Path) -> None:
        super()._copy_upstream_source_context(source_family, dest_dir)
        if source_family != DEFAULT_MINI_SWE_AGENT_NAME:
            return
        source = self.config.mini_swe_agent_source_path
        if not source.exists() or not source.is_dir():
            return
        candidate_mini_root = dest_dir / "upstream_source" / "mini-swe-agent"
        self._copy_tree_if_exists(source, candidate_mini_root)
        # Lock down the eval-gate copy: mirror the trusted repo-root version
        # with a banner. The grading subprocess invokes the absolute path so
        # any later proposer edit to this file is inert; we still keep a copy
        # visible so the proposer can study how passrate is computed.
        _mirror_eval_script_into_candidate(
            candidate_mini_root=candidate_mini_root,
            project_root=self.project_root,
        )

    def _candidate_policy_scan_paths(self, candidate: dict[str, Any]) -> list[Path]:
        out = super()._candidate_policy_scan_paths(candidate)
        source_path = self._candidate_mini_source_path(candidate)
        if source_path is not None and source_path.exists():
            # The mirrored eval-gate script is the trusted repo-root copy
            # injected by _mirror_eval_script_into_candidate so the proposer
            # can read it. It legitimately references the SWE-bench scorer
            # (it *is* the scorer entry). Skip it during code-policy scan;
            # eval-gate integrity is handled separately via sha256 detection.
            eval_gate_copy = (source_path / "scripts" / "run_miniswe_swebench_single.py").resolve()
            for path in sorted(source_path.rglob("*.py")):
                if path.resolve() == eval_gate_copy:
                    continue
                out.append(path)
        return sorted(set(out))

    def _candidate_code_policy_violations(self, candidate: Any) -> list[dict[str, str]]:
        violations = super()._candidate_code_policy_violations(candidate)
        if not isinstance(candidate, dict):
            return violations
        forbidden = {
            "test_patch": "runtime code must not read SWE-bench gold test patches",
            "gold_patch": "runtime code must not read SWE-bench gold patches",
            "swebench.harness": "candidate code must not call or modify the SWE-bench scorer",
            "candidate_results": "runtime code must not read previous candidate results",
        }
        for path in self._candidate_policy_scan_paths(candidate):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            for marker, reason in forbidden.items():
                if marker.lower() in text:
                    violations.append(
                        {
                            "path": str(path),
                            "marker": marker,
                            "reason": reason,
                        }
                    )
        return violations

    def _candidate_mini_source_path(self, candidate: dict[str, Any]) -> Path | None:
        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        for key in ("source_project_path", "upstream_source_path", "mini_swe_agent_source_path"):
            value = candidate.get(key) or extra.get(key)
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            return path
        return None

    def _build_source_snapshot_workspace(
        self,
        *,
        iteration: int,
        source_family: str,
        call_dir: Path,
        target_system: str | None = None,
        snapshot_root: Path | None = None,
        generated_dir: Path | None = None,
        base_iter: int | None = None,
    ) -> Path:
        snapshot_root = super()._build_source_snapshot_workspace(
            iteration=iteration,
            source_family=source_family,
            call_dir=call_dir,
            target_system=target_system,
            snapshot_root=snapshot_root,
            generated_dir=generated_dir,
            base_iter=base_iter,
        )
        candidate_dir = snapshot_root / "candidate"
        upstream = candidate_dir / "upstream_source" / "mini-swe-agent"
        # CuraII: when a parent base is supplied, replace the freshly
        # baseline-seeded mini-swe-agent source with the parent iteration's
        # archived candidate source so the proposer edits on top of a
        # previously evaluated candidate rather than restarting from baseline.
        if base_iter is not None:
            parent_upstream = (
                self._iteration_dir(base_iter)
                / "source_snapshot"
                / "candidate"
                / "upstream_source"
                / "mini-swe-agent"
            )
            if parent_upstream.exists():
                if upstream.exists():
                    shutil.rmtree(upstream)
                shutil.copytree(
                    parent_upstream,
                    upstream,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
                # Re-lock the eval gate after copying from the parent, in case
                # an earlier iteration's snapshot contains a tampered copy.
                _mirror_eval_script_into_candidate(
                    candidate_mini_root=upstream,
                    project_root=self.project_root,
                )
        readme = candidate_dir / "SNAPSHOT.md"
        readme.write_text(
            "\n".join(
                [
                    "# mini-SWE-agent Source Snapshot Candidate",
                    "",
                    f"Iteration: {iteration}",
                    f"Target system: {target_system or source_family}",
                    "",
                    "This directory is a writable candidate-specific source snapshot.",
                    "Edit `upstream_source/mini-swe-agent` to optimize the coding agent.",
                    "Do not edit evaluator/scorer files or read gold patches/test patches.",
                    "",
                    "Write `pending_eval.json` with exactly one candidate. Point",
                    "`source_project_path` at `source_snapshot/candidate/upstream_source/mini-swe-agent`",
                    "or the edited absolute path visible in the workspace.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        manifest_path = snapshot_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["mini_swe_agent_source"] = str(upstream)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        call_manifest_path = call_dir / "source_snapshot_manifest.json"
        if call_manifest_path.exists():
            call_manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return snapshot_root
