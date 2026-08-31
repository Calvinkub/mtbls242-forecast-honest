"""The old solver's failure mode was silent non-convergence. These tests catch it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monotone_logistic import fit_monotone_logistic  # noqa: E402


def _easy_problem(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    logits = 2.0 * x[:, 0] - 1.5 * x[:, 1]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    return x, y


def test_solver_reports_convergence():
    x, y = _easy_problem()
    model = fit_monotone_logistic(x, y, np.zeros(3, dtype=int), l2=1e-4)
    assert model.converged


def test_unconstrained_fit_recovers_the_true_coefficient_signs():
    x, y = _easy_problem()
    model = fit_monotone_logistic(x, y, np.zeros(3, dtype=int), l2=1e-4)
    assert model.weights[0] > 0.5
    assert model.weights[1] < -0.5


def test_coefficients_are_not_stuck_near_zero():
    """The regression this whole module exists to prevent.

    The superseded projected-gradient implementation returned a maximum standardized
    coefficient of 0.018 on real data and was mistaken for a fitted model.
    """
    x, y = _easy_problem()
    model = fit_monotone_logistic(x, y, np.zeros(3, dtype=int), l2=1e-4)
    assert np.max(np.abs(model.weights)) > 0.5


def test_positive_constraints_are_never_violated():
    x, y = _easy_problem()
    constraints = np.array([1, 1, 1])
    model = fit_monotone_logistic(x, y, constraints, l2=1e-4)
    assert np.all(model.weights >= -1e-12)


def test_negative_constraints_are_never_violated():
    x, y = _easy_problem()
    constraints = np.array([-1, -1, -1])
    model = fit_monotone_logistic(x, y, constraints, l2=1e-4)
    assert np.all(model.weights <= 1e-12)


def test_a_constraint_opposing_the_data_is_pinned_to_zero_and_flagged():
    """Feature 0 truly has a positive effect; forcing it negative must bind at 0."""
    x, y = _easy_problem()
    constraints = np.array([-1, 0, 0])
    model = fit_monotone_logistic(x, y, constraints, l2=1e-4)
    assert model.weights[0] == pytest.approx(0.0, abs=1e-8)
    assert model.binding_constraints(constraints)[0]


def test_a_constraint_agreeing_with_the_data_does_not_bind():
    x, y = _easy_problem()
    constraints = np.array([1, 0, 0])
    model = fit_monotone_logistic(x, y, constraints, l2=1e-4)
    assert model.weights[0] > 0.5
    assert not model.binding_constraints(constraints)[0]


def test_penalty_meaning_is_stable_across_sample_size():
    """`l2` is applied to the mean loss, so doubling n must not shrink the fit."""
    small_x, small_y = _easy_problem(n=300, seed=1)
    big_x, big_y = _easy_problem(n=3000, seed=1)
    small = fit_monotone_logistic(small_x, small_y, np.zeros(3, dtype=int), l2=0.01)
    big = fit_monotone_logistic(big_x, big_y, np.zeros(3, dtype=int), l2=0.01)
    assert big.weights[0] == pytest.approx(small.weights[0], rel=0.35)


def test_heavier_penalty_shrinks_coefficients():
    x, y = _easy_problem()
    light = fit_monotone_logistic(x, y, np.zeros(3, dtype=int), l2=1e-4)
    heavy = fit_monotone_logistic(x, y, np.zeros(3, dtype=int), l2=10.0)
    assert np.max(np.abs(heavy.weights)) < np.max(np.abs(light.weights))


def test_predicted_probabilities_are_valid_and_ordered_by_the_signal():
    x, y = _easy_problem()
    model = fit_monotone_logistic(x, y, np.array([1, -1, 0]), l2=1e-4)
    prob = model.predict_proba(x)
    assert np.all((prob > 0.0) & (prob < 1.0))
    assert np.corrcoef(prob, x[:, 0])[0, 1] > 0.5


def test_intercept_alone_reproduces_prevalence_when_features_are_useless():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2000, 3))
    y = (rng.random(2000) < 0.3).astype(int)
    model = fit_monotone_logistic(x, y, np.zeros(3, dtype=int), l2=1.0)
    assert model.predict_proba(x).mean() == pytest.approx(0.3, abs=0.03)
