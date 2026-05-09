"""OptiHarness benchmark optimization harness.

The historical Python package name remains ``optimizer1`` for backward
compatibility. New user-facing entry points should use ``optiharness``.
"""

from optimizer1.evaluation import EvaluationRunner, run_initial_frontier
from optimizer1.pareto import ParetoPoint, pareto_frontier

__all__ = [
    "EvaluationRunner",
    "ParetoPoint",
    "pareto_frontier",
    "run_initial_frontier",
]
