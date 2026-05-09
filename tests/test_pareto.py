from memomemo.pareto import ParetoPoint, pareto_frontier
from memomemo.optimizer import OptimizerConfig


def point(name, passrate, tokens, average_score=None):
    return ParetoPoint(
        candidate_id=name,
        scaffold_name=name.split("_")[0],
        passrate=passrate,
        token_consuming=tokens,
        avg_token_consuming=tokens,
        average_score=passrate if average_score is None else average_score,
        result_path=f"{name}.json",
        config={},
    )


def test_pareto_keeps_passrate_score_tradeoffs_and_drops_dominated():
    frontier = pareto_frontier(
        [
            point("weak_low_score", 0.4, 200, 0.4),
            point("high_passrate_low_score", 0.8, 300, 0.6),
            point("same_quality_cheaper", 0.8, 100, 0.6),
            point("lower_passrate_higher_score", 0.7, 80, 0.75),
        ]
    )
    assert [item.candidate_id for item in frontier] == [
        "same_quality_cheaper",
        "lower_passrate_higher_score",
    ]


def test_pareto_quality_threshold_drops_lower_score_when_passrate_gap_is_large():
    frontier = pareto_frontier(
        [
            point("strong_expensive", 0.8, 300, 0.7),
            point("cheap_much_weaker", 0.74, 50, 0.7),
        ],
        quality_gap_threshold=0.03,
    )
    assert [item.candidate_id for item in frontier] == ["strong_expensive"]


def test_pareto_keeps_score_tradeoff_even_with_more_tokens():
    frontier = pareto_frontier(
        [
            point("strong_lower_score", 0.8, 300, 0.6),
            point("weaker_higher_score", 0.78, 50, 0.75),
        ],
        quality_gap_threshold=0.03,
    )
    assert [item.candidate_id for item in frontier] == [
        "strong_lower_score",
        "weaker_higher_score",
    ]


def test_optimizer_default_pareto_quality_threshold_is_full_train_friendly(tmp_path):
    config = OptimizerConfig(run_id="r", out_dir=tmp_path)

    assert config.pareto_quality_threshold == 0.125
