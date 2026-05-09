"""Memory scaffold registry."""

from __future__ import annotations

from optimizer1.scaffolds.base import MemoryScaffold, RetrievalMemoryScaffold, ScaffoldConfig, ScaffoldRun
from optimizer1.scaffolds.memgpt_scaffold import MemGPTSourceScaffold


SCAFFOLD_REGISTRY: dict[str, type[MemoryScaffold]] = {
    MemGPTSourceScaffold.name: MemGPTSourceScaffold,
}

DEFAULT_EVOLUTION_SEED_SCAFFOLDS = (
    MemGPTSourceScaffold.name,
)

DEFAULT_BASELINE_SCAFFOLDS = DEFAULT_EVOLUTION_SEED_SCAFFOLDS

DEFAULT_MEMORY_SCAFFOLDS = DEFAULT_EVOLUTION_SEED_SCAFFOLDS

DEFAULT_SCAFFOLD_TOP_KS = {
    MemGPTSourceScaffold.name: 12,
}


def available_scaffolds() -> tuple[str, ...]:
    return tuple(sorted(SCAFFOLD_REGISTRY))


def build_scaffold(name: str) -> MemoryScaffold:
    try:
        return SCAFFOLD_REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(available_scaffolds())
        raise ValueError(f"unknown scaffold {name!r}; available: {available}") from exc


__all__ = [
    "MemoryScaffold",
    "RetrievalMemoryScaffold",
    "ScaffoldConfig",
    "ScaffoldRun",
    "DEFAULT_BASELINE_SCAFFOLDS",
    "DEFAULT_EVOLUTION_SEED_SCAFFOLDS",
    "DEFAULT_MEMORY_SCAFFOLDS",
    "DEFAULT_SCAFFOLD_TOP_KS",
    "available_scaffolds",
    "build_scaffold",
]
