from __future__ import annotations

import ab_pro
import abliterator
from abliterator.cli import build_parser


def test_ab_pro_api_preserves_compatibility_namespace() -> None:
    assert ab_pro.__version__ == abliterator.__version__
    assert ab_pro.ABProError is abliterator.AbliteratorError
    assert ab_pro.ModelLoadConfig is abliterator.ModelLoadConfig


def test_ab_pro_is_primary_cli_brand() -> None:
    parser = build_parser()
    assert parser.prog == "abpro"
    assert "AB Pro" in parser.description
