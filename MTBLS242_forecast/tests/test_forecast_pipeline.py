"""Pipeline-level guards, run against the real MTBLS242 tables shipped in this repo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controls import residualize_change, trivial_signed_baseline_score  # noqa: E402
from forecast_mtbls242 import load_longitudinal_pairs  # noqa: E402
from forecast_responder_pr_auc import (  # noqa: E402
    ARTIFACT_FEATURES,
    MIN_IMPROVED_MARKERS_RESIDUAL,
    MONOTONE_DIRECTIONS,
    legacy_outcome,
    marker_columns,
    residual_outcome,
)


@pytest.fixture(scope="module")
def cohort():
    return load_longitudinal_pairs()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def test_cohort_is_the_expected_71_paired_subjects(cohort):
    paired, x, y, metabolites = cohort
    assert len(paired) == 71
    assert x.shape == y.shape == (71, 21)
    assert not paired["subject_id"].duplicated().any()


def test_every_constrained_marker_exists_in_the_data(cohort):
    _, _, _, metabolites = cohort
    for name in MONOTONE_DIRECTIONS:
        assert name in metabolites


# --------------------------------------------------------------------------
# The outcome definition - the defect this rewrite exists to fix
# --------------------------------------------------------------------------

def test_legacy_outcome_is_strongly_determined_by_the_baseline(cohort):
    """Documents the defect: raw change is anti-correlated with its own baseline."""
    _, x, y, metabolites = cohort
    idx, _ = marker_columns(metabolites)
    correlations = [np.corrcoef(x[:, j], (y - x)[:, j])[0, 1] for j in idx]
    assert np.median(correlations) < -0.5


def test_residual_outcome_removes_the_baseline_dependence(cohort):
    _, x, y, metabolites = cohort
    idx, _ = marker_columns(metabolites)
    fit = residualize_change(x[:, idx], y[:, idx], np.arange(len(x)))
    residual = fit.transform(x[:, idx], y[:, idx])
    for position, j in enumerate(idx):
        assert abs(np.corrcoef(x[:, j], residual[:, position])[0, 1]) < 1e-8


def test_residual_outcome_reproduces_the_original_prevalence(cohort):
    """The threshold was pre-registered to hold the base rate at 31/71.

    This keeps the corrected PR-AUC directly comparable to the retired 0.671 instead
    of being confounded by a shift in prevalence.
    """
    _, x, y, metabolites = cohort
    labels, _ = residual_outcome(x, y, metabolites, np.arange(len(x)))
    assert labels.sum() == 31
    assert labels.mean() == pytest.approx(0.437, abs=0.001)


def test_legacy_and_residual_outcomes_disagree_about_who_responded(cohort):
    """If they agreed, the correction would change nothing."""
    _, x, y, metabolites = cohort
    legacy, _ = legacy_outcome(x, y, metabolites)
    residual, _ = residual_outcome(x, y, metabolites, np.arange(len(x)))
    assert legacy.sum() == residual.sum() == 31
    assert np.mean(legacy == residual) < 0.85


def test_pre_registered_threshold_is_not_silently_changed():
    assert MIN_IMPROVED_MARKERS_RESIDUAL == 6


# --------------------------------------------------------------------------
# Fold hygiene
# --------------------------------------------------------------------------

def test_residualization_never_sees_held_out_rows(cohort):
    """The residualization is part of the label, so it must be fitted in-fold only."""
    _, x, y, metabolites = cohort
    train_idx = np.arange(50)
    clean, _ = residual_outcome(x, y, metabolites, train_idx)

    corrupted = y.copy()
    corrupted[50:] += 5.0
    idx, _ = marker_columns(metabolites)
    fit_clean = residualize_change(x[:, idx], y[:, idx], train_idx)
    fit_dirty = residualize_change(x[:, idx], corrupted[:, idx], train_idx)
    assert np.allclose(fit_clean.slope, fit_dirty.slope)
    assert np.allclose(fit_clean.intercept, fit_dirty.intercept)
    assert clean.shape == (71,)


# --------------------------------------------------------------------------
# Artifact quarantine
# --------------------------------------------------------------------------

def test_isopropanol_really_is_elevated_before_surgery(cohort):
    """Bridge flags it as a surgical skin-prep contaminant. Confirm before excluding."""
    _, x, y, metabolites = cohort
    j = metabolites.index("isopropanol")
    assert x[:, j].mean() > y[:, j].mean() + 1.0


def test_artifact_features_are_all_present_and_excludable(cohort):
    _, _, _, metabolites = cohort
    for name in ARTIFACT_FEATURES:
        assert name in metabolites
    kept = [m for m in metabolites if m not in ARTIFACT_FEATURES]
    assert len(kept) == 18


def test_no_artifact_is_also_a_constrained_marker():
    """An exogenous compound must never carry a biological monotonic constraint."""
    assert not set(ARTIFACT_FEATURES) & set(MONOTONE_DIRECTIONS)


# --------------------------------------------------------------------------
# The claim itself
# --------------------------------------------------------------------------

def test_trivial_scorer_matches_the_retired_headline_on_the_legacy_outcome(cohort):
    """The retired 0.671 was reachable with no model at all.

    This is the finding that retired the original claim, pinned so it cannot be
    quietly lost: a sum of signed baseline z-scores scores what the 'trained'
    monotonic model scored.
    """
    from controls import average_precision

    _, x, y, metabolites = cohort
    idx, signs = marker_columns(metabolites)
    labels, _ = legacy_outcome(x, y, metabolites)
    trivial = trivial_signed_baseline_score(x, idx, signs)
    assert average_precision(labels, trivial) == pytest.approx(0.653, abs=0.02)
