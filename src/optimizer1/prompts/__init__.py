"""Agent role definitions for the optimizer.

Every named agent in the optimization loop (proposer, diagnoser, ...)
has its role contract — what it is, what it must produce, what it
must not do — in a sibling Markdown file named ``<agent>.md``. These
files are the single source of truth and are consumed in two
different ways depending on the runtime path:

  - **Claude Code subagent path** (``--proposer-agent claude`` +
    ``--diagnose``): the file is copied verbatim into the workspace
    so Claude Code can load it via its native protocol —
    ``proposer.md`` becomes ``<workspace>/CLAUDE.md`` (auto-loaded
    main-conversation system prompt) and ``diagnoser.md`` becomes
    ``<workspace>/.claude/agents/diagnoser.md`` (named subagent).
  - **Legacy prompt-string path** (everything else): the file is
    read by Python via ``load_role_prompt`` and concatenated into the
    user message. ``load_role_prompt`` strips any YAML frontmatter so
    Claude-specific subagent metadata never leaks into a raw prompt.

The two layers are maintained independently:

  - editing wording / responsibilities / style → edit the .md
  - changing which fields get injected per iteration → edit the .py
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent
_FRONTMATTER_DELIM = "---\n"


def _strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block if present.

    Claude Code subagent files start with ``---\\n...---\\n``; legacy
    string-injection paths must not see that metadata.
    """

    if not text.startswith(_FRONTMATTER_DELIM):
        return text
    end = text.find("\n" + _FRONTMATTER_DELIM, len(_FRONTMATTER_DELIM))
    if end < 0:
        return text
    return text[end + len("\n" + _FRONTMATTER_DELIM):]


@lru_cache(maxsize=8)
def load_role_prompt(name: str) -> str:
    """Return the contents of ``<name>.md`` as a string, frontmatter
    stripped. Cached because the file does not change at runtime and
    is read on every iteration. Raises ``FileNotFoundError`` with a
    helpful message if the prompt is missing.
    """

    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Role prompt {name!r} not found at {path}. Available: "
            + ", ".join(sorted(p.stem for p in _PROMPTS_DIR.glob("*.md")))
        )
    raw = path.read_text(encoding="utf-8")
    return _strip_frontmatter(raw).rstrip() + "\n"


def role_prompt_path(name: str) -> Path:
    """Return the absolute path to ``<name>.md``. Used by deployment
    code that copies files verbatim into a workspace (frontmatter and
    all)."""

    return _PROMPTS_DIR / f"{name}.md"


__all__ = ["load_role_prompt", "role_prompt_path"]
