"""Per-candidate rationale writer — LLM-generated narrative for one
optimization iteration.

Each call produces a structured JSON record (hypothesis / change /
outcome / diagnosis / next_signal), pretty-prints it as Markdown under
``traces/rationale/iter_NNN/<candidate>.md``, and indexes the parsed
fields into the ``rationales`` table.

Failures are deliberately non-fatal: an LLM error or a malformed
response writes a placeholder file plus a row with raw_yaml populated
and parsed fields empty. Callers should never depend on a successful
rationale to make progress.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from optimizer1.model import LocalModelClient


_PROMPT_SYSTEM = (
    "You summarize one iteration of a code-optimization loop. "
    "You must output ONE JSON object and nothing else — no prose, "
    "no Markdown fences, no commentary before or after."
)

_PROMPT_USER_TEMPLATE = """Inputs:
- Candidate's stated intent (pending_eval excerpt):
{pending_eval}

- Diff that was applied (truncated):
{diff}

- Evaluation summary:
  passrate: {passrate}
  mean_score: {mean_score}
  task counts: passed={n_pass}, failed={n_fail}
  failed task_ids (sample): {failed_sample}
  passed task_ids (sample): {passed_sample}

- Baseline diff classification (counts vs harness baseline):
  regressed={c_regressed}, breakthrough={c_breakthrough},
  persistent_fail={c_pfail}, stable_pass={c_stable}

Output exactly this JSON schema (no extra keys, no commentary):

{{
  "hypothesis": "<one sentence: what this change was supposed to fix and why>",
  "change": "<one line: which files were modified and what they did>",
  "outcome": {{
    "passrate_delta": <number; pass `null` if not derivable>,
    "regressed_tasks": [<at most 5 task_ids>],
    "breakthrough_tasks": [<at most 5 task_ids>]
  }},
  "diagnosis": "<one sentence; MUST cite at least one concrete task_id, number, or filename>",
  "next_hypothesis_signal": "<one sentence with a concrete next-step idea, or empty string>"
}}"""


@dataclass(frozen=True)
class RationaleResult:
    path: Path
    hypothesis: str | None
    diagnosis: str | None
    next_signal: str | None
    raw_response: str
    parsed: dict[str, Any] | None


# Type for an LLM caller. Defaults to LocalModelClient.chat-style call;
# tests inject a stub.
LLMCaller = Callable[[list[dict[str, str]]], str]


class RationaleWriter:
    def __init__(
        self,
        *,
        root: Path,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_s: int = 60,
        max_diff_chars: int = 4000,
        max_pending_eval_chars: int = 2000,
        llm_caller: LLMCaller | None = None,
    ) -> None:
        self.root = Path(root)
        self.model = model
        self.max_diff_chars = max_diff_chars
        self.max_pending_eval_chars = max_pending_eval_chars
        self._llm_caller: LLMCaller
        if llm_caller is not None:
            self._llm_caller = llm_caller
        else:
            client = LocalModelClient(
                model=model,
                base_url=base_url,
                api_key=api_key or "EMPTY",
                timeout_s=timeout_s,
            )

            def _call(messages: list[dict[str, str]]) -> str:
                return client.chat(messages, max_tokens=600, temperature=0.0).content

            self._llm_caller = _call

    def write(
        self,
        *,
        iteration: int,
        candidate_id: str,
        pending_eval: dict[str, Any] | str | None,
        tasks: list[dict[str, Any]],
        diff_text: str,
        status_counts: dict[str, int],
        passrate: float | None,
        mean_score: float | None,
    ) -> RationaleResult:
        """Generate, persist, and return the rationale for one candidate.

        Always writes a markdown file (placeholder if the LLM call or
        parse fails) and never raises on LLM/parse errors.
        """

        prompt = self._build_prompt(
            pending_eval=pending_eval,
            tasks=tasks,
            diff_text=diff_text,
            status_counts=status_counts,
            passrate=passrate,
            mean_score=mean_score,
        )
        messages = [
            {"role": "system", "content": _PROMPT_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        raw = ""
        parsed: dict[str, Any] | None = None
        error: str | None = None
        try:
            raw = self._llm_caller(messages) or ""
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            error = f"LLM call failed: {exc!r}"

        if raw and error is None:
            parsed_or_err = _parse_json_response(raw)
            if isinstance(parsed_or_err, dict):
                parsed = parsed_or_err
            else:
                error = parsed_or_err  # type: ignore[assignment]

        path = self._write_markdown(
            iteration=iteration,
            candidate_id=candidate_id,
            parsed=parsed,
            raw=raw,
            error=error,
        )

        return RationaleResult(
            path=path,
            hypothesis=(parsed or {}).get("hypothesis") if parsed else None,
            diagnosis=(parsed or {}).get("diagnosis") if parsed else None,
            next_signal=(parsed or {}).get("next_hypothesis_signal") if parsed else None,
            raw_response=raw,
            parsed=parsed,
        )

    # ---- prompt --------------------------------------------------

    def _build_prompt(
        self,
        *,
        pending_eval: dict[str, Any] | str | None,
        tasks: list[dict[str, Any]],
        diff_text: str,
        status_counts: dict[str, int],
        passrate: float | None,
        mean_score: float | None,
    ) -> str:
        pe_excerpt = self._format_pending_eval(pending_eval)
        diff_excerpt = self._truncate(diff_text or "(no diff)", self.max_diff_chars)

        passed_ids: list[str] = []
        failed_ids: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            tid = str(task.get("task_id") or "")
            if not tid:
                continue
            if task.get("passed"):
                passed_ids.append(tid)
            else:
                failed_ids.append(tid)

        return _PROMPT_USER_TEMPLATE.format(
            pending_eval=pe_excerpt,
            diff=diff_excerpt,
            passrate=_fmt_number(passrate),
            mean_score=_fmt_number(mean_score),
            n_pass=len(passed_ids),
            n_fail=len(failed_ids),
            failed_sample=", ".join(failed_ids[:8]) or "(none)",
            passed_sample=", ".join(passed_ids[:8]) or "(none)",
            c_regressed=int(status_counts.get("regressed", 0)),
            c_breakthrough=int(status_counts.get("breakthrough", 0)),
            c_pfail=int(status_counts.get("persistent_fail", 0)),
            c_stable=int(status_counts.get("stable_pass", 0)),
        )

    def _format_pending_eval(
        self, pending_eval: dict[str, Any] | str | None
    ) -> str:
        if pending_eval is None:
            return "(none)"
        if isinstance(pending_eval, str):
            return self._truncate(pending_eval, self.max_pending_eval_chars)
        try:
            text = json.dumps(pending_eval, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(pending_eval)
        return self._truncate(text, self.max_pending_eval_chars)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... (truncated, original {len(text)} chars)"

    # ---- write markdown ------------------------------------------

    def _write_markdown(
        self,
        *,
        iteration: int,
        candidate_id: str,
        parsed: dict[str, Any] | None,
        raw: str,
        error: str | None,
    ) -> Path:
        out_dir = self.root / "rationale" / f"iter_{iteration:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{candidate_id}.md"

        lines = [f"# iter_{iteration:03d} / {candidate_id} rationale", ""]
        if parsed is not None:
            outcome = parsed.get("outcome") or {}
            lines += [
                f"**Hypothesis**: {parsed.get('hypothesis') or '(missing)'}",
                "",
                f"**Change**: {parsed.get('change') or '(missing)'}",
                "",
                "**Outcome**:",
                f"- passrate_delta: {outcome.get('passrate_delta')}",
                f"- regressed_tasks: {outcome.get('regressed_tasks') or []}",
                f"- breakthrough_tasks: {outcome.get('breakthrough_tasks') or []}",
                "",
                f"**Diagnosis**: {parsed.get('diagnosis') or '(missing)'}",
                "",
                f"**Next signal**: {parsed.get('next_hypothesis_signal') or ''}",
                "",
            ]
        else:
            lines += [
                "_(rationale unavailable — see error below)_",
                "",
                f"**Error**: {error or 'unknown'}",
                "",
            ]
        lines += ["---", f"*Generated by RationaleWriter, model={self.model}.*", ""]
        if raw:
            lines += ["```", "Raw model response:", raw, "```", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


# ---- helpers ----------------------------------------------------


_FENCE_RX = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_json_response(raw: str) -> dict[str, Any] | str:
    """Return parsed dict on success, error string on failure."""

    text = raw.strip()
    # Strip a single ```json ... ``` fence if the model added one.
    fenced = _FENCE_RX.search(text)
    if fenced:
        text = fenced.group(1).strip()
    # Last-ditch: take the substring from first { to last }.
    if not text.startswith("{"):
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return f"no JSON object found in response (len={len(raw)})"
        text = text[first : last + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"JSON parse failed: {exc}"
    if not isinstance(parsed, dict):
        return f"top-level JSON is not an object: {type(parsed).__name__}"
    return parsed


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.4f}"
