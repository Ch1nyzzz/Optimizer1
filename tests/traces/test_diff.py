"""diff.classify — pure function, exhaustive truth table."""

from __future__ import annotations

from optimizer1.traces import (
    STATUS_BREAKTHROUGH,
    STATUS_NO_BASELINE,
    STATUS_PERSISTENT_FAIL,
    STATUS_REGRESSED,
    STATUS_STABLE_PASS,
    BaselineEntry,
    classify,
)


def _entry(passed: bool, score: float = 0.0) -> BaselineEntry:
    return BaselineEntry(
        trace_id="bl_x",
        task_id="t",
        passed=passed,
        score=score,
    )


def test_classify_returns_no_baseline_when_baseline_is_none():
    assert classify(curr_passed=True, baseline=None) == STATUS_NO_BASELINE
    assert classify(curr_passed=False, baseline=None) == STATUS_NO_BASELINE


def test_classify_regressed_pass_to_fail():
    assert (
        classify(curr_passed=False, baseline=_entry(passed=True))
        == STATUS_REGRESSED
    )


def test_classify_breakthrough_fail_to_pass():
    assert (
        classify(curr_passed=True, baseline=_entry(passed=False))
        == STATUS_BREAKTHROUGH
    )


def test_classify_stable_pass():
    assert (
        classify(curr_passed=True, baseline=_entry(passed=True))
        == STATUS_STABLE_PASS
    )


def test_classify_persistent_fail():
    assert (
        classify(curr_passed=False, baseline=_entry(passed=False))
        == STATUS_PERSISTENT_FAIL
    )
