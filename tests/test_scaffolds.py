from memomemo.evaluation import make_initial_candidate_grid
from memomemo.scaffolds import (
    DEFAULT_BASELINE_SCAFFOLDS,
    DEFAULT_EVOLUTION_SEED_SCAFFOLDS,
    DEFAULT_MEMORY_SCAFFOLDS,
    DEFAULT_SCAFFOLD_TOP_KS,
    available_scaffolds,
    build_scaffold,
)
from memomemo.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig
from memomemo.schemas import ConversationTurn, LocomoExample


def example():
    turns = (
        ConversationTurn("session_1", "2024-01-01", "D1", "Alice", "I adopted a dog named Max.", 0),
        ConversationTurn("session_2", "2024-01-02", "D2", "Alice", "Max likes blue frisbees.", 1),
    )
    return LocomoExample(
        task_id="T1",
        sample_id="S1",
        question="What is Alice's dog named?",
        answer="Max",
        category=1,
        evidence=("D1",),
        conversation=turns,
    )


def test_lightweight_retrieval_scaffold_retrieves_relevant_memory():
    ex = example()
    scaffold = build_scaffold("memgpt_source")
    assert isinstance(scaffold, RetrievalMemoryScaffold)
    state = scaffold.build(ex, ScaffoldConfig(top_k=2))
    hits = scaffold.retrieve(state, ex.question, ScaffoldConfig(top_k=2))
    assert hits
    assert any("Max" in hit.text for hit in hits)


def test_default_experiment_scaffolds_exclude_controls():
    assert DEFAULT_EVOLUTION_SEED_SCAFFOLDS == ("memgpt_source",)
    assert DEFAULT_MEMORY_SCAFFOLDS == DEFAULT_EVOLUTION_SEED_SCAFFOLDS
    assert DEFAULT_BASELINE_SCAFFOLDS == ("memgpt_source",)
    assert DEFAULT_SCAFFOLD_TOP_KS["memgpt_source"] == 12
    assert "no_memory" not in available_scaffolds()
    assert "mem0_source" not in available_scaffolds()
    assert "memgpt_source" in available_scaffolds()
    assert "membank_source" not in available_scaffolds()
    assert "bm25" not in available_scaffolds()
    assert "amem_source" not in available_scaffolds()
    assert "amem" not in available_scaffolds()
    assert "mem0" not in available_scaffolds()
    assert [item[2] for item in make_initial_candidate_grid()] == [
        "memgpt_source_top12",
    ]


def test_memgpt_source_scaffold_exposes_memory_hierarchy():
    scaffold = build_scaffold("memgpt_source")
    state = scaffold.build(example(), ScaffoldConfig(top_k=4, extra={"context_window_turns": 1}))
    hits = scaffold.retrieve(state, "What does Max like?", ScaffoldConfig(top_k=4))

    tiers = {hit.metadata.get("memory_tier") for hit in hits}
    assert {"core", "archival", "recall"} <= tiers
    assert any(hit.metadata.get("tool") == "archival_memory_search" for hit in hits)
    assert any(hit.metadata.get("tool") == "conversation_search" for hit in hits)
    assert any("<memory_blocks>" in hit.text for hit in hits)
    assert any("blue frisbees" in hit.text for hit in hits)

