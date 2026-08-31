"""Guards for the autoregressive panel, against the real MTBLS242 tables."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast_autoregressive import build_transitions, grouped_folds  # noqa: E402
from forecast_mtbls242 import TIME_ORDER, load_timepoint_panel  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return load_timepoint_panel()


@pytest.fixture(scope="module")
def transitions(panel):
    index, values, _ = panel
    return build_transitions(index, values)


# --------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------

def test_panel_uses_every_sample_not_just_the_pairs(panel):
    """The whole point: the pairwise loader discards 323 of 465 samples."""
    index, values, metabolites = panel
    assert len(index) == 465
    assert values.shape == (465, 21)


def test_panel_covers_all_five_time_points(panel):
    index, _, _ = panel
    assert set(index["timepoint"]) == set(TIME_ORDER)
    counts = index["timepoint"].value_counts().to_dict()
    assert counts["preop"] == 106
    assert counts["12 months after surgery"] == 71


def test_no_subject_has_two_samples_at_one_time_point(panel):
    index, _, _ = panel
    assert not index.duplicated(["subject_id", "timepoint"]).any()


def test_sixty_subjects_have_a_complete_trajectory(panel):
    index, _, _ = panel
    complete = index.groupby("subject_id")["time_order"].apply(
        lambda s: set(s) == set(range(5))
    )
    assert int(complete.sum()) == 60


# --------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------

def test_transition_count_matches_the_panel(transitions):
    meta, x, y = transitions
    assert len(meta) == 346
    assert x.shape == y.shape == (346, 21)


def test_transitions_are_strictly_consecutive(transitions):
    """A subject missing 6 months must NOT produce a manufactured 3→9 jump."""
    meta, _, _ = transitions
    order = {t: i for i, t in enumerate(TIME_ORDER)}
    steps = meta["to_timepoint"].map(order) - meta["from_timepoint"].map(order)
    assert (steps == 1).all()


def test_per_step_counts_are_as_expected(transitions):
    meta, _, _ = transitions
    counts = meta["transition"].value_counts().to_dict()
    assert counts["preop -> 3 months after surgery"] == 98
    assert counts["9 months after surgery -> 12 months after surgery"] == 66


def test_transitions_give_far_more_data_than_the_pairwise_setup(transitions):
    """346 transitions against 71 pairs — the reason for doing this at all."""
    meta, _, _ = transitions
    assert len(meta) > 4 * 71


def test_each_transition_row_belongs_to_one_subject(transitions, panel):
    meta, _, _ = transitions
    index, _, _ = panel
    assert set(meta["subject_id"]).issubset(set(index["subject_id"]))


# --------------------------------------------------------------------------
# Subject-grouped cross-validation — the leakage guard
# --------------------------------------------------------------------------

def test_no_subject_ever_spans_train_and_test(transitions):
    """One subject contributes up to 4 transitions; splitting rows would leak."""
    meta, _, _ = transitions
    groups = meta["subject_id"].to_numpy()
    for train_idx, test_idx in grouped_folds(groups, 5, seed=0):
        assert not set(groups[train_idx]) & set(groups[test_idx])


def test_grouped_folds_partition_every_row_exactly_once(transitions):
    meta, _, _ = transitions
    groups = meta["subject_id"].to_numpy()
    seen = []
    for _, test_idx in grouped_folds(groups, 5, seed=0):
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(len(groups)))


def test_grouped_folds_are_reasonably_balanced(transitions):
    meta, _, _ = transitions
    groups = meta["subject_id"].to_numpy()
    sizes = [len(test_idx) for _, test_idx in grouped_folds(groups, 5, seed=0)]
    assert max(sizes) - min(sizes) <= 0.35 * np.mean(sizes)


def test_grouped_folds_are_deterministic(transitions):
    meta, _, _ = transitions
    groups = meta["subject_id"].to_numpy()
    runs = [
        [t.tolist() for _, t in grouped_folds(groups, 5, seed=0)] for _ in range(2)
    ]
    assert runs[0] == runs[1]


# --------------------------------------------------------------------------
# The persistence baseline this experiment has to beat
# --------------------------------------------------------------------------

def test_persistence_is_a_genuinely_strong_baseline_late_in_the_trajectory(transitions):
    """By 9→12 months the profile barely moves, so carrying it forward is hard to beat.

    This is why persistence, not the cohort mean, is the comparator that matters for
    a short-horizon autoregressive forecast.
    """
    meta, x, y = transitions
    err = np.mean(np.abs(y - x), axis=1)
    early = err[(meta["transition"] == "preop -> 3 months after surgery").to_numpy()]
    late = err[
        (meta["transition"] == "9 months after surgery -> 12 months after surgery").to_numpy()
    ]
    assert late.mean() < early.mean() / 1.4
