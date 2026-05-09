"""Adapter registry — one per benchmark.

Each benchmark provides a `BenchmarkTraceAdapter` that turns a raw `task`
dict (as produced by that benchmark's evaluator and stored under
`candidate_results/<candidate>.json`) into the unified `Trace`.

Adapters are registered at import time. Built-in adapters are wired up in
`optimizer1.traces.__init__` so callers only need to import the package.
"""

from __future__ import annotations

from typing import Any, Protocol

from .schema import Trace


class BenchmarkTraceAdapter(Protocol):
    name: str

    def build_trace(
        self,
        *,
        iteration: int,
        candidate_id: str,
        task: dict[str, Any],
    ) -> Trace: ...


_REGISTRY: dict[str, BenchmarkTraceAdapter] = {}


def register_adapter(adapter: BenchmarkTraceAdapter) -> None:
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> BenchmarkTraceAdapter:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"No trace adapter registered for benchmark {name!r}. "
            f"Available: {available}"
        )
    return _REGISTRY[name]


def has_adapter(name: str) -> bool:
    return name in _REGISTRY
