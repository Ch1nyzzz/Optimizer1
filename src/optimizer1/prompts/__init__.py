"""Proposer skill loading for the optimizer.

Every benchmark has one self-contained proposer skill at
``skills/<benchmark>/SKILL.md``. The skill carries the proposer's full
contract for that benchmark — role, objective, generalization rules, search
space, workflow, quality gate, edit scope, and ``pending_eval.json``
conventions. It is delivered to the proposer agent in one of two ways:

  - Claude Code path — the skill text is passed via ``--append-system-prompt``.
  - Codex path — the skill text is written into ``<workspace>/AGENTS.md``,
    which Codex auto-prepends to its system prompt at session start.

A skill file may carry mode-specific blocks delimited by
``<!-- MODE:<name> -->`` / ``<!-- END MODE:<name> -->``. ``load_proposer_skill``
keeps the block for the active mode and drops the others. Modes:

  - ``default`` — cumulative-summary evidence workflow.
  - ``organized`` — state.md + RunStore tools, summaries withheld.
  - ``organized-summaries`` — organized tools with summaries kept as orientation.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent
_SKILLS_DIR = _PROMPTS_DIR / "skills"
_FRONTMATTER_DELIM = "---\n"

_MODE_BLOCK_RE = re.compile(
    r"[ \t]*<!-- MODE:([\w-]+) -->\n(.*?)\n[ \t]*<!-- END MODE:\1 -->[ \t]*\n?",
    re.DOTALL,
)

PROPOSER_SKILL_MODES = ("default", "organized", "organized-summaries")


def _strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block if present.

    Skill files start with ``---\\n...---\\n`` for documentation parity with
    upstream skills; the injected proposer contract must not carry that
    metadata.
    """

    if not text.startswith(_FRONTMATTER_DELIM):
        return text
    end = text.find("\n" + _FRONTMATTER_DELIM, len(_FRONTMATTER_DELIM))
    if end < 0:
        return text
    return text[end + len("\n" + _FRONTMATTER_DELIM):]


def _resolve_mode_blocks(text: str, mode: str) -> str:
    """Keep the active mode's block, drop the others, strip the markers."""

    def repl(match: re.Match[str]) -> str:
        block_mode = match.group(1)
        body = match.group(2)
        if block_mode == mode:
            return body.strip("\n") + "\n"
        return ""

    return _MODE_BLOCK_RE.sub(repl, text)


def benchmark_skill_name(*, benchmark_name: str, target_system: str) -> str:
    """Return the proposer skill key (``skills/<key>/SKILL.md``).

    Resolution keys off the human benchmark name and the target-system
    identifier so it works regardless of which one the caller has handy.
    """

    benchmark = benchmark_name.lower()
    target = target_system.lower()
    if "terminal-bench" in benchmark or "terminus" in target:
        return "terminus"
    if "swe-bench" in benchmark or "mini_swe" in target or "miniswe" in target:
        return "swebench"
    if (
        "graph" in benchmark
        or "colour" in benchmark
        or "color" in benchmark
        or "graph" in target
    ):
        return "graph_colouring"
    if "longmemeval" in benchmark:
        return "longmemeval"
    if "locomo" in benchmark:
        return "locomo"
    # Default to the LoCoMo skill: the base optimizer is the LoCoMo loop.
    return "locomo"


@lru_cache(maxsize=32)
def load_proposer_skill(skill_key: str, mode: str = "default") -> str:
    """Return the resolved proposer skill text for a benchmark + mode.

    YAML frontmatter is stripped; the active ``mode``'s block is kept and the
    other mode blocks are dropped. Raises ``FileNotFoundError`` with a helpful
    message if the skill is missing.
    """

    path = _SKILLS_DIR / skill_key / "SKILL.md"
    if not path.exists():
        available = ", ".join(
            sorted(p.parent.name for p in _SKILLS_DIR.glob("*/SKILL.md"))
        )
        raise FileNotFoundError(
            f"Proposer skill {skill_key!r} not found at {path}. "
            f"Available: {available}"
        )
    raw = path.read_text(encoding="utf-8")
    body = _strip_frontmatter(raw)
    return _resolve_mode_blocks(body, mode).rstrip() + "\n"


def proposer_skill_path(skill_key: str) -> Path:
    """Return the absolute path to a benchmark's ``SKILL.md`` (unresolved)."""

    return _SKILLS_DIR / skill_key / "SKILL.md"


__all__ = [
    "PROPOSER_SKILL_MODES",
    "benchmark_skill_name",
    "load_proposer_skill",
    "proposer_skill_path",
]
