from optimizer1.pareto import FRONTIER_TOP_K, ParetoPoint, dominates, pareto_frontier
from optimizer1.optimizer import OptimizerConfig


def point(name, passrate, tokens=100, average_score=None):
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


def test_dominates_strictly_by_passrate():
    a = point("a", 0.8)
    b = point("b", 0.7)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_ignores_other_fields():
    # Equal passrate — neither dominates, even if average_score / tokens differ.
    a = point("a", 0.8, tokens=300, average_score=0.5)
    b = point("b", 0.8, tokens=50, average_score=0.9)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominates_quality_gap_threshold_is_deadcode():
    # quality_gap_threshold no longer affects the result; passrate is the
    # only signal.
    a = point("a", 0.8, average_score=0.6)
    b = point("b", 0.7, average_score=0.9)
    assert dominates(a, b, quality_gap_threshold=0.5)
    assert not dominates(b, a, quality_gap_threshold=0.5)


def test_frontier_takes_strict_top_k_by_passrate():
    frontier = pareto_frontier(
        [
            point("p_low", 0.4),
            point("p_high1", 0.8),
            point("p_mid", 0.6),
            point("p_high2", 0.8),
        ]
    )
    assert [item.candidate_id for item in frontier] == ["p_high2", "p_high1", "p_mid"]


def test_frontier_breaks_ties_by_candidate_id_desc():
    frontier = pareto_frontier(
        [
            point("alpha", 0.5),
            point("charlie", 0.5),
            point("bravo", 0.5),
            point("delta", 0.4),
        ]
    )
    # Three points share passrate=0.5; tie-break picks larger candidate_id
    # first. "delta" never enters because top-3 fills with charlie/bravo/alpha.
    assert [item.candidate_id for item in frontier] == ["charlie", "bravo", "alpha"]


def test_frontier_keeps_iter_number_ordering_via_string_compare():
    # iter010_xxx > iter003_xxx lexicographically because '0' < '1' at
    # position 4 — so later iters win ties under the chosen rule.
    frontier = pareto_frontier(
        [
            point("iter003_run", 0.7),
            point("iter010_run", 0.7),
            point("iter002_run", 0.7),
            point("iter001_run", 0.5),
        ]
    )
    assert [item.candidate_id for item in frontier] == [
        "iter010_run",
        "iter003_run",
        "iter002_run",
    ]


def test_frontier_returns_pool_when_smaller_than_top_k():
    frontier = pareto_frontier(
        [
            point("only_one", 0.5),
            point("only_two", 0.4),
        ]
    )
    assert [item.candidate_id for item in frontier] == ["only_one", "only_two"]


def test_frontier_returns_empty_for_empty_pool():
    assert pareto_frontier([]) == []


def test_frontier_drops_dominated_lower_passrate_points():
    # Single best, then two ties; the 0.4 point is excluded because we
    # already have 3 stronger or equal points.
    frontier = pareto_frontier(
        [
            point("best", 0.9),
            point("midA", 0.6),
            point("midB", 0.6),
            point("loser", 0.4),
        ]
    )
    assert [item.candidate_id for item in frontier] == ["best", "midB", "midA"]


def test_frontier_top_k_argument_overrides_default():
    frontier = pareto_frontier(
        [
            point("a", 0.9),
            point("b", 0.8),
            point("c", 0.7),
            point("d", 0.6),
        ],
        top_k=2,
    )
    assert [item.candidate_id for item in frontier] == ["a", "b"]


def test_frontier_default_top_k_is_three():
    assert FRONTIER_TOP_K == 3


def test_frontier_quality_gap_threshold_is_deadcode():
    # threshold no longer filters; passing it should not change output.
    no_threshold = pareto_frontier(
        [
            point("hi", 0.8, average_score=0.6),
            point("lo", 0.74, average_score=0.7),
        ]
    )
    with_threshold = pareto_frontier(
        [
            point("hi", 0.8, average_score=0.6),
            point("lo", 0.74, average_score=0.7),
        ],
        quality_gap_threshold=0.5,
    )
    assert [item.candidate_id for item in no_threshold] == [
        item.candidate_id for item in with_threshold
    ]


def test_optimizer_config_still_carries_pareto_quality_threshold(tmp_path):
    # Field is deadcode at the pareto layer but still on the config object;
    # other call sites (run_initial_frontier, save_frontier) accept it.
    config = OptimizerConfig(run_id="r", out_dir=tmp_path)
    assert config.pareto_quality_threshold == 0.125
