from __future__ import annotations

from memomemo.benchmark_tasks import normalize_task_name, task_spec


def test_benchmark_task_aliases_normalize_to_canonical_slugs() -> None:
    assert normalize_task_name(None) == "locomo"
    assert normalize_task_name("locomo_subset") == "locomo"
    assert normalize_task_name("lme") == "longmemeval"
    assert normalize_task_name("coding") == "swebench"


def test_task_specs_carry_base_agent_mapping() -> None:
    assert task_spec("locomo").base_agent_system == "locomo_memory_scaffolds"
    assert task_spec("longmemeval").base_agent_system == task_spec("locomo").base_agent_system
    assert task_spec("swebench").base_agent_system == "mini_swe_agent_source"
