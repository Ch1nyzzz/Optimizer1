"""Trace harness — pluggable per-iteration trace recording.

Every optimizer run records structured traces under `runs/<run>/traces/`:
a JSONL per (iteration, candidate), a manifest, a SQLite index of diffs
vs the baseline, and pre-rendered diagnostic markdown for each iteration.

The package auto-registers built-in benchmark adapters at import time.
"""

from __future__ import annotations

from .adapter import (
    BenchmarkTraceAdapter,
    get_adapter,
    has_adapter,
    register_adapter,
)
from .baseline import Baseline
from .diff import (
    ALL_STATUSES,
    STATUS_BASELINE,
    STATUS_BREAKTHROUGH,
    STATUS_NO_BASELINE,
    STATUS_PERSISTENT_FAIL,
    STATUS_REGRESSED,
    STATUS_STABLE_PASS,
    BaselineEntry,
    classify,
)
from .harness import BACKEND_VERSION, TraceHarness
from .indexer import Indexer
from .query import TraceQuery
from .recorder import Recorder
from .renderer import RenderConfig, Renderer
from .schema import (
    SCHEMA_VERSION,
    Span,
    Trace,
    read_jsonl,
    trace_from_dict,
    trace_to_dict,
    write_jsonl,
)

# Auto-register built-in adapters so importing the package is enough.
from .adapters.longmemeval import LongMemEvalAdapter
from .adapters.locomo import LocomoAdapter
from .adapters.swebench import SwebenchAdapter
from .adapters.graph_colouring import GraphColouringAdapter

register_adapter(LongMemEvalAdapter())
register_adapter(LocomoAdapter())
register_adapter(SwebenchAdapter())
register_adapter(GraphColouringAdapter())

__all__ = [
    "ALL_STATUSES",
    "BACKEND_VERSION",
    "Baseline",
    "BaselineEntry",
    "BenchmarkTraceAdapter",
    "Indexer",
    "RenderConfig",
    "Renderer",
    "Recorder",
    "SCHEMA_VERSION",
    "STATUS_BASELINE",
    "STATUS_BREAKTHROUGH",
    "STATUS_NO_BASELINE",
    "STATUS_PERSISTENT_FAIL",
    "STATUS_REGRESSED",
    "STATUS_STABLE_PASS",
    "Span",
    "Trace",
    "TraceHarness",
    "TraceQuery",
    "classify",
    "get_adapter",
    "has_adapter",
    "read_jsonl",
    "register_adapter",
    "trace_from_dict",
    "trace_to_dict",
    "write_jsonl",
]
