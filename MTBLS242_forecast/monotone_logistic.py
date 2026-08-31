"""Sign-constrained logistic regression that actually converges.

The original `fit_monotone_logistic` ran fixed-step projected gradient descent for a
capped 6000 iterations against a penalty that was not scaled to sample size. On
MTBLS242 it stopped with a maximum standardized coefficient of 0.018 and selected
`l2=10` in 3 of 5 folds - a near-constant model whose ranking happened to coincide with
an unweighted sum of the markers. Every "monotonic constraint" it reported was
therefore meaningless: the coefficients had barely left zero.

This replaces it with L-BFGS-B under box constraints. A `+1` monotonic direction is the
bound `[0, inf)`, a `-1` direction is `(-inf, 0]`, and an unconstrained feature is
unbounded. Box constraints give exact signs and a convergence guarantee, so a
coefficient that lands on 0.0 means the constraint is genuinely *binding* - the data
pulled the other way - rather than that the optimizer gave up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class MonotoneLogisticModel:
    """A fitted sign-constrained logistic regression."""

    x_mean: np.ndarray
    x_std: np.ndarray
    weights: np.ndarray
    intercept: float
    l2: float
    converged: bool
    n_iterations: int

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=float) - self.x_mean) / self.x_std
        return _sigmoid(z @ self.weights + self.intercept)

    def binding_constraints(self, constraints: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """Mask of constrained features whose coefficient was pinned to zero.

        These are the features where the literature-assigned direction disagrees with
        this cohort. That disagreement is a result worth reporting, not a bug.
        """
        constraints = np.asarray(constraints)
        return (constraints != 0) & (np.abs(self.weights) <= tol)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def fit_monotone_logistic(
    x: np.ndarray,
    y: np.ndarray,
    constraints: np.ndarray,
    l2: float,
    max_iter: int = 20_000,
) -> MonotoneLogisticModel:
    """Minimise mean logistic loss + (l2/2)|w|^2 subject to per-feature sign bounds.

    The penalty is applied to the *mean* loss so that `l2` means the same thing at any
    sample size, and the intercept is never penalised.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    constraints = np.asarray(constraints)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    z = (x - x_mean) / x_std
    n, n_features = z.shape

    prevalence = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    start = np.zeros(n_features + 1)
    start[-1] = np.log(prevalence / (1 - prevalence))

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        weights, intercept = params[:-1], params[-1]
        logits = z @ weights + intercept
        # log(1 + exp(t)) evaluated stably.
        loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
        loss += 0.5 * l2 * float(weights @ weights)
        error = _sigmoid(logits) - y
        grad = np.empty_like(params)
        grad[:-1] = z.T @ error / n + l2 * weights
        grad[-1] = float(error.mean())
        return loss, grad

    bounds: list[tuple[float | None, float | None]] = []
    for direction in constraints:
        if direction > 0:
            bounds.append((0.0, None))
        elif direction < 0:
            bounds.append((None, 0.0))
        else:
            bounds.append((None, None))
    bounds.append((None, None))  # intercept

    result = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-10},
    )

    return MonotoneLogisticModel(
        x_mean=x_mean,
        x_std=x_std,
        weights=result.x[:-1],
        intercept=float(result.x[-1]),
        l2=float(l2),
        converged=bool(result.success),
        n_iterations=int(result.nit),
    )
