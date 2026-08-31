"""Each control is tested against a case whose answer is known in advance."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controls import (  # noqa: E402
    average_precision,
    bootstrap_delta_ci,
    label_permutation_null,
    pairing_destroyed_null,
    repeated_stratified_cv,
    residualize_change,
    trivial_signed_baseline_score,
)


# --------------------------------------------------------------------------
# average_precision
# --------------------------------------------------------------------------

def test_average_precision_is_one_when_ranking_is_perfect():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.9, 0.8])
    assert average_precision(y, score) == pytest.approx(1.0)


def test_average_precision_of_a_constant_score_averages_to_the_prevalence():
    """A score carrying no information scores the no-skill baseline in expectation."""
    rng = np.random.default_rng(99)
    scores = []
    for _ in range(300):
        y = (rng.random(200) < 0.4).astype(int)
        scores.append(average_precision(y, np.zeros(200)))
    assert np.mean(scores) == pytest.approx(0.4, abs=0.02)


# --------------------------------------------------------------------------
# trivial_signed_baseline_score
# --------------------------------------------------------------------------

def test_trivial_scorer_separates_perfectly_separable_data():
    """A no-learning scorer must reach AP 1.0 when the signed markers separate."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    # Positives are pushed up on marker 0 and down on marker 1.
    y = np.zeros(60, dtype=int)
    y[:30] = 1
    x[:30, 0] += 12.0
    x[:30, 1] -= 12.0
    score = trivial_signed_baseline_score(x, [0, 1], np.array([1, -1]))
    assert average_precision(y, score) == pytest.approx(1.0)


def test_trivial_scorer_on_random_data_averages_to_the_prevalence():
    """Averaged over draws, because a single AP estimate is very noisy at n=400."""
    rng = np.random.default_rng(1)
    scores = []
    for _ in range(200):
        x = rng.normal(size=(400, 5))
        y = (rng.random(400) < 0.4).astype(int)
        score = trivial_signed_baseline_score(x, [0, 1, 2], np.array([1, 1, -1]))
        scores.append(average_precision(y, score))
    assert np.mean(scores) == pytest.approx(0.4, abs=0.02)


def test_trivial_scorer_is_invariant_to_feature_scale():
    """Scores are z-scored, so re-scaling a column must not change the ranking."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 3))
    a = trivial_signed_baseline_score(x, [0, 1], np.array([1, -1]))
    scaled = x.copy()
    scaled[:, 0] *= 1000.0
    b = trivial_signed_baseline_score(scaled, [0, 1], np.array([1, -1]))
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# residualize_change
# --------------------------------------------------------------------------

def test_residualized_change_is_orthogonal_to_baseline():
    """The whole point: residuals must carry no linear baseline dependence."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 4)) * 3.0
    # Pure regression to the mean: the future pulls halfway back to the grand mean.
    y = x + (x.mean(axis=0) - x) * 0.5 + rng.normal(size=(200, 4)) * 0.1
    fit = residualize_change(x, y, train_idx=np.arange(200))
    resid = fit.transform(x, y)
    for j in range(4):
        assert abs(np.corrcoef(x[:, j], resid[:, j])[0, 1]) < 0.05


def test_residualize_fitted_on_train_applies_to_heldout():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 3))
    y = x * 0.4 + rng.normal(size=(120, 3)) * 0.2
    train = np.arange(80)
    test = np.arange(80, 120)
    fit = residualize_change(x, y, train_idx=train)
    resid = fit.transform(x, y)
    # Held-out residuals are computed, finite, and not merely the raw change.
    assert np.isfinite(resid[test]).all()
    assert not np.allclose(resid[test], (y - x)[test])


def test_residualize_uses_only_training_rows():
    """Corrupting held-out rows must not move the fitted coefficients."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 2))
    y = x * 0.5 + rng.normal(size=(100, 2)) * 0.1
    train = np.arange(50)
    clean = residualize_change(x, y, train_idx=train)
    corrupted_y = y.copy()
    corrupted_y[50:] += 999.0
    dirty = residualize_change(x, corrupted_y, train_idx=train)
    assert np.allclose(clean.slope, dirty.slope)
    assert np.allclose(clean.intercept, dirty.intercept)


# --------------------------------------------------------------------------
# pairing_destroyed_null
# --------------------------------------------------------------------------

def _rtm_outcome(x, y):
    """Responder = at least 3 of 4 markers fell. Mechanically baseline-driven."""
    return (np.sum((y - x) < 0, axis=1) >= 3).astype(int)


def test_pairing_destroyed_null_is_not_cleared_by_pure_regression_to_the_mean():
    """On pure-RTM data the real score must sit inside the null distribution."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=(150, 4)) * 3.0
    y = x + (x.mean(axis=0) - x) * 0.6 + rng.normal(size=(150, 4)) * 0.5

    def scorer(xx):
        return trivial_signed_baseline_score(xx, [0, 1, 2, 3], np.array([1, 1, 1, 1]))

    real = average_precision(_rtm_outcome(x, y), scorer(x))
    null = pairing_destroyed_null(x, y, _rtm_outcome, scorer, n_reps=150, seed=0)
    # Real lift over prevalence is no better than the null's own lift.
    assert real / _rtm_outcome(x, y).mean() <= np.percentile(null.lift, 95)


def test_pairing_destroyed_null_is_cleared_by_genuine_injected_signal():
    """A real subject-level signal must beat the null it is compared against."""
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(size=(n, 4))
    latent = rng.normal(size=n)          # per-subject responsiveness
    x[:, 0] += latent * 2.0              # ...visible at baseline in marker 0
    y = x.copy()
    y[:, 1:] -= latent[:, None] * 3.0    # ...and it drives the future change

    def outcome(xx, yy):
        return (np.sum((yy - xx)[:, 1:] < 0, axis=1) >= 2).astype(int)

    def scorer(xx):
        return trivial_signed_baseline_score(xx, [0], np.array([1]))

    real = average_precision(outcome(x, y), scorer(x))
    null = pairing_destroyed_null(x, y, outcome, scorer, n_reps=150, seed=0)
    assert real / outcome(x, y).mean() > np.percentile(null.lift, 95)


def test_pairing_destroyed_null_reports_prevalence_per_replicate():
    """Prevalence shifts when pairing is destroyed, so it must be reported."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=(80, 4))
    y = x + rng.normal(size=(80, 4))

    def scorer(xx):
        return trivial_signed_baseline_score(xx, [0, 1], np.array([1, 1]))

    null = pairing_destroyed_null(x, y, _rtm_outcome, scorer, n_reps=40, seed=0)
    assert len(null.prevalence) == len(null.average_precision) == len(null.lift)
    assert np.all(null.prevalence > 0) and np.all(null.prevalence < 1)


# --------------------------------------------------------------------------
# label_permutation_null
# --------------------------------------------------------------------------

def test_permutation_p_value_is_small_for_a_strong_ranking():
    rng = np.random.default_rng(9)
    y = np.repeat([0, 1], 60)
    score = y + rng.normal(size=120) * 0.2
    assert label_permutation_null(y, score, n_reps=500, seed=0).p_value < 0.01


def test_permutation_p_value_is_large_for_a_meaningless_ranking():
    rng = np.random.default_rng(10)
    y = (rng.random(120) < 0.4).astype(int)
    score = rng.normal(size=120)
    assert label_permutation_null(y, score, n_reps=500, seed=0).p_value > 0.05


# --------------------------------------------------------------------------
# bootstrap_delta_ci
# --------------------------------------------------------------------------

def test_bootstrap_ci_spans_zero_when_two_models_are_equivalent():
    rng = np.random.default_rng(11)
    errors = rng.random(80)
    result = bootstrap_delta_ci(errors, errors.copy(), n_reps=800, seed=0)
    assert result.low <= 0.0 <= result.high


def test_bootstrap_ci_excludes_zero_when_one_model_is_clearly_better():
    rng = np.random.default_rng(12)
    baseline = rng.random(200) + 1.0
    better = baseline - 0.5
    result = bootstrap_delta_ci(better, baseline, n_reps=800, seed=0)
    assert result.high < 0.0
    assert result.point == pytest.approx(-0.5, abs=0.01)


# --------------------------------------------------------------------------
# repeated_stratified_cv
# --------------------------------------------------------------------------

def test_repeated_cv_never_puts_a_subject_in_train_and_test():
    y = np.repeat([0, 1], 40)
    for _, _, train_idx, test_idx in repeated_stratified_cv(y, 5, 4, seed=0):
        assert not set(train_idx) & set(test_idx)


def test_repeated_cv_covers_every_subject_exactly_once_per_repeat():
    y = np.repeat([0, 1], 40)
    seen = {}
    for repeat, _, _, test_idx in repeated_stratified_cv(y, 5, 3, seed=0):
        seen.setdefault(repeat, []).extend(test_idx.tolist())
    for repeat, idx in seen.items():
        assert sorted(idx) == list(range(80)), f"repeat {repeat} is not a partition"


def test_repeated_cv_folds_are_stratified():
    y = np.array([0] * 70 + [1] * 30)
    for _, _, _, test_idx in repeated_stratified_cv(y, 5, 2, seed=0):
        assert y[test_idx].mean() == pytest.approx(0.3, abs=0.12)


def test_repeated_cv_repeats_differ_but_are_deterministic():
    y = np.repeat([0, 1], 40)
    runs = [
        [t.tolist() for _, _, _, t in repeated_stratified_cv(y, 5, 2, seed=0)]
        for _ in range(2)
    ]
    assert runs[0] == runs[1], "same seed must give the same folds"
    assert runs[0][0] != runs[0][5], "different repeats must give different folds"
