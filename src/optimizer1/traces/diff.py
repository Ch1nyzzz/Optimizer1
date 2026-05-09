"""Diff classification — pure functions, no I/O.

A trace's `status` is computed by comparing the current run's pass/score
to a `BaselineEntry` for the same task_id. There are six possible
statuses:

  - ``baseline``         — this trace is itself part of a baseline run
                           (no `--trace-baseline` was supplied). Set by
                           callers, not by `classify`.
  - ``regressed``        — passed in baseline, fails now.
  - ``breakthrough``     — failed in baseline, passes now.
  - ``stable_pass``      — passed in both.
  - ``persistent_fail``  — failed in both.
  - ``no_baseline``      — baseline run has no trace for this task_id.

The reasoning is intentionally minimal — anything beyond pass/fail
flips (e.g. score-delta thresholds or failure-mode tagging) belongs at
the renderer or downstream tooling, where it can be tuned per
benchmark without invalidating the index.
"""

from __future__ import annotations

from dataclasses import dataclass


STATUS_BASELINE = "baseline"
STATUS_REGRESSED = "regressed"
STATUS_BREAKTHROUGH = "breakthrough"
STATUS_STABLE_PASS = "stable_pass"
STATUS_PERSISTENT_FAIL = "persistent_fail"
STATUS_NO_BASELINE = "no_baseline"

ALL_STATUSES = (
    STATUS_BASELINE,
    STATUS_REGRESSED,
    STATUS_BREAKTHROUGH,
    STATUS_STABLE_PASS,
    STATUS_PERSISTENT_FAIL,
    STATUS_NO_BASELINE,
)


@dataclass(frozen=True)
class BaselineEntry:
    trace_id: str
    task_id: str
    passed: bool
    score: float


def classify(*, curr_passed: bool, baseline: BaselineEntry | None) -> str:
    """Compare curr trace pass/fail vs the baseline entry for the same task.

    Callers handle the ``baseline`` status separately by writing it
    directly when the harness is recording a baseline run; this function
    only handles the diff cases.
    """

    if baseline is None:
        return STATUS_NO_BASELINE
    if baseline.passed and not curr_passed:
        return STATUS_REGRESSED
    if not baseline.passed and curr_passed:
        return STATUS_BREAKTHROUGH
    if baseline.passed and curr_passed:
        return STATUS_STABLE_PASS
    return STATUS_PERSISTENT_FAIL
