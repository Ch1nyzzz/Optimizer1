"""Overlay that injects the `evolved` algorithm slot into a graph-colouring tree.

The upstream `Rmehta-sudo/graph-colouring` does not ship an algorithm slot
named `evolved`, and its dispatch table also forgets to register the existing
`tabu_search` algorithm. The harness invokes `--algorithm evolved` on every
candidate (seed and proposer-mutated alike), so we patch every workspace
copy we hand the compiler:

  - Add src/algorithms/evolved.{cpp,h} (the proposer-editable slot, seeded
    to delegate to colour_with_tabu so iter 0 = upstream TabuCol baseline).
  - Patch src/benchmark_runner.cpp to include the new headers and register
    both `evolved` and `tabu_search` in the algorithm dispatch table.
  - Patch the Makefile to include evolved.cpp in SOURCES.

apply_overlay() is idempotent: it skips any file that already carries the
overlay marker (so re-applying on a workspace that already has overlayed
content is safe and preserves proposer edits to evolved.cpp).
"""

from __future__ import annotations

from pathlib import Path

OVERLAY_MARKER = "// optimizer1-overlay-evolved"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def apply_overlay(target_dir: Path) -> None:
    """Inject the `evolved` slot into a graph-colouring source tree.

    Args:
        target_dir: Root of the graph-colouring tree to patch (must contain
            `src/` and `Makefile` at the top level — i.e. the same structure
            as the upstream repo).

    Raises:
        FileNotFoundError: If target_dir does not look like a graph-colouring
            workspace.
    """

    target_dir = Path(target_dir)
    src = target_dir / "src"
    makefile = target_dir / "Makefile"
    if not src.is_dir():
        raise FileNotFoundError(f"missing src/ at {target_dir}")
    if not makefile.exists():
        raise FileNotFoundError(f"missing Makefile at {target_dir}")

    _ensure_evolved_files(src)
    _patch_benchmark_runner(src / "benchmark_runner.cpp")
    _patch_makefile(makefile)


def _ensure_evolved_files(src: Path) -> None:
    """Materialise evolved.{cpp,h} from packaged templates iff absent.

    Preserves any proposer-authored content already on disk so iterative
    mutation is not silently overwritten.
    """

    algorithms_dir = src / "algorithms"
    algorithms_dir.mkdir(parents=True, exist_ok=True)
    for name in ("evolved.h", "evolved.cpp"):
        target = algorithms_dir / name
        if target.exists():
            continue
        target.write_text(
            (_TEMPLATES_DIR / name).read_text(encoding="utf-8"), encoding="utf-8"
        )


def _patch_benchmark_runner(runner_path: Path) -> None:
    """Inject `evolved` and `tabu_search` includes + dispatch entries.

    Idempotent via OVERLAY_MARKER. We patch two anchors in the upstream
    file: the include block immediately after `algorithms/exact_solver.h`
    and the dispatch table's `exact_solver` entry. Both anchors are
    upstream-stable.
    """

    text = runner_path.read_text(encoding="utf-8")
    if OVERLAY_MARKER in text:
        return

    include_anchor = '#include "algorithms/exact_solver.h"'
    if include_anchor not in text:
        raise ValueError(
            f"benchmark_runner.cpp missing include anchor "
            f"{include_anchor!r}; upstream layout changed?"
        )
    text = text.replace(
        include_anchor,
        (
            f"{include_anchor}\n"
            f'#include "algorithms/tabu.h"  {OVERLAY_MARKER}\n'
            f'#include "algorithms/evolved.h"  {OVERLAY_MARKER}'
        ),
        1,
    )

    dispatch_anchor = (
        '\t\t{"exact_solver", [](const Graph &graph) '
        "{ return colour_with_exact(graph); }},"
    )
    if dispatch_anchor not in text:
        raise ValueError(
            "benchmark_runner.cpp missing dispatch anchor for exact_solver; "
            "upstream layout changed?"
        )
    text = text.replace(
        dispatch_anchor,
        (
            f"{dispatch_anchor}\n"
            f'\t\t{{"tabu_search", [](const Graph &graph) '
            f"{{ return colour_with_tabu(graph); }}}},  {OVERLAY_MARKER}\n"
            f'\t\t{{"evolved", [](const Graph &graph) '
            f"{{ return colour_with_evolved(graph); }}}},  {OVERLAY_MARKER}"
        ),
        1,
    )
    runner_path.write_text(text, encoding="utf-8")


def _patch_makefile(makefile_path: Path) -> None:
    """Add evolved.cpp to the Makefile SOURCES list. Idempotent."""

    text = makefile_path.read_text(encoding="utf-8")
    if "$(SRC_DIR)/algorithms/evolved.cpp" in text:
        return
    anchor = "$(SRC_DIR)/algorithms/tabu.cpp"
    if anchor not in text:
        raise ValueError(
            "Makefile missing tabu.cpp SOURCES entry; upstream layout changed?"
        )
    text = text.replace(
        anchor,
        f"{anchor} \\\n\t$(SRC_DIR)/algorithms/evolved.cpp",
        1,
    )
    makefile_path.write_text(text, encoding="utf-8")
