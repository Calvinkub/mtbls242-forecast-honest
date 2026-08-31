"""Negative controls and uncertainty estimates shared by both forecast models.

Every headline number in this repo is reported next to the controls in this module.
A model that does not beat `trivial_signed_baseline_score`, or whose score sits inside
the `pairing_destroyed_null` distribution, has demonstrated nothing about forecasting -
regardless of how far its raw metric sits above the prevalence baseline.

All functions are deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------

def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC as average precision. Ties keep their input order (stable sort)."""
    y_true = np.asarray(y_true)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    ranked = y_true[np.argsort(-np.asarray(score), kind="mergesort")]
    precision = np.cumsum(ranked) / np.arange(1, len(y_true) + 1)
    return float(np.sum(precision * ranked) / positives)


# --------------------------------------------------------------------------
# The no-learning scorer
# --------------------------------------------------------------------------

def trivial_signed_baseline_score(
    x: np.ndarray,
    marker_idx: Sequence[int],
    signs: np.ndarray,
) -> np.ndarray:
    """Sum of signed, z-scored baseline markers. No fitting of any kind.

    This is the floor every trained model must clear. On MTBLS242 the original
    monotonic logistic model scored exactly the same AP as this function (0.653),
    which is what first exposed that it had not converged.
    """
    x = np.asarray(x, dtype=float)
    marker_idx = list(marker_idx)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = (x - x.mean(axis=0)) / std
    return z[:, marker_idx] @ np.asarray(signs, dtype=float)


# --------------------------------------------------------------------------
# Regression-to-the-mean removal
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidualChangeFit:
    """Per-marker linear fit of change-on-baseline, fitted on training rows only."""

    slope: np.ndarray
    intercept: np.ndarray

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Change, minus the part of it that the baseline mechanically predicts."""
        x = np.asarray(x, dtype=float)
        change = np.asarray(y, dtype=float) - x
        return change - (self.intercept + self.slope * x)


def residualize_change(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
) -> ResidualChangeFit:
    """Fit `change_j ~ a_j + b_j * baseline_j` on `train_idx`.

    A subject with an extreme baseline falls back toward the mean whether or not
    anything biological happened, so raw change is not a forecastable outcome when
    the baseline is also a feature. The residual is orthogonal to the baseline by
    construction, which removes that mechanical edge.

    The fit is part of the *label* definition, so it must never see held-out rows.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    train_idx = np.asarray(train_idx, dtype=int)

    xt = x[train_idx]
    change_t = y[train_idx] - xt
    x_mean = xt.mean(axis=0)
    change_mean = change_t.mean(axis=0)
    centered = xt - x_mean
    variance = np.sum(centered**2, axis=0)
    covariance = np.sum(centered * (change_t - change_mean), axis=0)
    slope = np.divide(
        covariance, variance, out=np.zeros_like(covariance), where=variance > 1e-12
    )
    intercept = change_mean - slope * x_mean
    return ResidualChangeFit(slope=slope, intercept=intercept)


# --------------------------------------------------------------------------
# Nulls
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PairingDestroyedNull:
    """Null in which no subject-level forecastable signal can exist."""

    average_precision: np.ndarray
    prevalence: np.ndarray
    lift: np.ndarray

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.lift, q))


def pairing_destroyed_null(
    x: np.ndarray,
    y: np.ndarray,
    outcome_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    scorer: Callable[[np.ndarray], np.ndarray],
    n_reps: int = 200,
    seed: int = 0,
) -> PairingDestroyedNull:
    """Shuffle which future profile belongs to which subject, then rebuild the outcome.

    This destroys any real subject-level signal while preserving both marginal
    distributions and every mechanical baseline-to-change relationship. A score that
    the null also achieves is regression to the mean, not forecasting.

    Destroying the pairing also shifts the outcome prevalence, so results are reported
    as **lift over each replicate's own prevalence**, which is comparable across
    replicates; raw AP alone is not.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    aps, prevalences = [], []

    for _ in range(int(n_reps)):
        labels = outcome_fn(x, y[rng.permutation(len(y))])
        prevalence = float(np.mean(labels))
        if labels.sum() < 2 or prevalence >= 1.0:
            continue
        aps.append(average_precision(labels, scorer(x)))
        prevalences.append(prevalence)

    if not aps:
        raise ValueError("Every null replicate was degenerate; check outcome_fn.")

    aps = np.asarray(aps, dtype=float)
    prevalences = np.asarray(prevalences, dtype=float)
    return PairingDestroyedNull(
        average_precision=aps, prevalence=prevalences, lift=aps / prevalences
    )


@dataclass(frozen=True)
class PermutationNull:
    """Standard significance test: labels permuted against a fixed score."""

    observed: float
    null_average_precision: np.ndarray
    p_value: float


def label_permutation_null(
    y_true: np.ndarray,
    score: np.ndarray,
    n_reps: int = 1000,
    seed: int = 0,
) -> PermutationNull:
    """Empirical p-value for an observed average precision.

    Uses the conventional (1 + #{null >= observed}) / (1 + n) estimator, which never
    reports p = 0.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    observed = average_precision(y_true, score)
    rng = np.random.default_rng(seed)
    null = np.array(
        [average_precision(rng.permutation(y_true), score) for _ in range(int(n_reps))]
    )
    p_value = float((1 + np.sum(null >= observed)) / (1 + len(null)))
    return PermutationNull(observed=observed, null_average_precision=null, p_value=p_value)


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaCI:
    """Paired bootstrap confidence interval for a difference in mean error."""

    point: float
    low: float
    high: float

    @property
    def spans_zero(self) -> bool:
        return self.low <= 0.0 <= self.high


def bootstrap_delta_ci(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_reps: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> DeltaCI:
    """CI for `mean(errors_a) - mean(errors_b)`, resampling subjects in pairs.

    Negative means model A has the lower error. Subjects are resampled together so the
    interval accounts for the two error series being measured on the same people.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if errors_a.shape != errors_b.shape:
        raise ValueError("Paired bootstrap needs one error per subject in both arrays.")

    rng = np.random.default_rng(seed)
    n = len(errors_a)
    draws = np.empty(int(n_reps), dtype=float)
    for i in range(int(n_reps)):
        pick = rng.integers(0, n, size=n)
        draws[i] = errors_a[pick].mean() - errors_b[pick].mean()

    return DeltaCI(
        point=float(errors_a.mean() - errors_b.mean()),
        low=float(np.percentile(draws, 100 * alpha / 2)),
        high=float(np.percentile(draws, 100 * (1 - alpha / 2))),
    )


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def repeated_stratified_cv(
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int = 0,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield `(repeat, fold, train_idx, test_idx)` for repeated stratified k-fold.

    A single 5-fold draw on 71 subjects puts 14 people in each test set; the spread
    across those folds says more about test-set size than about the model. Repeating
    the whole split under different seeds gives a mean whose variation reflects
    resampling instability, which is the number worth reporting.
    """
    y = np.asarray(y)
    all_idx = np.arange(len(y))

    for repeat in range(1, int(n_repeats) + 1):
        rng = np.random.default_rng(seed + 1000 * repeat)
        parts: list[list[np.ndarray]] = [[] for _ in range(n_splits)]
        for label in np.unique(y):
            idx = np.flatnonzero(y == label)
            rng.shuffle(idx)
            for fold_no, chunk in enumerate(np.array_split(idx, n_splits)):
                parts[fold_no].append(chunk)

        for fold_no, chunks in enumerate(parts, start=1):
            test_idx = np.sort(np.concatenate(chunks)).astype(int)
            train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
            yield repeat, fold_no, train_idx, test_idx
