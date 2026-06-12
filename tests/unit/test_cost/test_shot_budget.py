"""Shot-budget estimator tests."""
import pytest

from qb_compiler.cost.shot_budget import shots_for_expectation, shots_for_rate


def test_expectation_scales_inverse_square():
    a = shots_for_expectation(0.01)
    b = shots_for_expectation(0.005)
    assert 3.8 < b / a < 4.2


def test_expectation_l1_scaling():
    assert shots_for_expectation(0.01, observable_l1=2.0) > shots_for_expectation(0.01)


def test_rate_monotone_in_width():
    assert shots_for_rate(0.01, rel_width=0.1) > shots_for_rate(0.01, rel_width=0.5)


def test_rare_rate_is_expensive():
    assert shots_for_rate(0.001, rel_width=0.2) > 90_000


def test_validation():
    with pytest.raises(ValueError):
        shots_for_expectation(0.0)
    with pytest.raises(ValueError):
        shots_for_rate(0.5, rel_width=0.2, confidence=0.97)
