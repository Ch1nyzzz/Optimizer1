from __future__ import annotations

import optiharness
from optiharness.cli import main as optiharness_main
from optimizer1.cli import main as optimizer1_main


def test_optiharness_public_alias_imports_cli() -> None:
    assert optiharness_main is optimizer1_main
    assert hasattr(optiharness, "EvaluationRunner")
