"""Autoregressive forecast across the 3 / 6 / 9 / 12-month follow-ups.

The pairwise experiments use only preop and 12 months — 142 of 465 samples. This one
uses the whole panel: every consecutive transition a subject contributes, 346 of them
against the 71 pairs the responder model gets.

Two questions:

1. **One-step.** Given a subject's profile at time t, forecast it at t+1. Reported
   per transition, because the four are not comparable: surgery happens inside
   preop→3mo, and the metabolome moves far more there than across 9→12mo.
2. **Iterated vs direct.** Roll the one-step model forward four times from preop to
   reach 12 months, and compare against the direct preop→12mo ridge. Does walking the
   trajectory beat jumping straight to the end?

Baselines are the point. Over a 3-month gap a profile barely moves, so **persistence**
(forecast = carry the last observation forward) is a strong comparator, and any model
that cannot beat it has learned nothing. The training-fold mean is reported too,
per time point.

Evaluation is **subject-grouped**: one subject contributes up to four transitions, so
splitting transitions at random would put the same person in train and test. Groups are
whole subjects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from controls import bootstrap_delta_ci
from forecast_mtbls242 import TIME_ORDER, load_timepoint_panel
from forecast_responder_pr_auc import ARTIFACT_FEATURES

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_ar"
SEED = 42
N_SPLITS = 5
# The grid runs to 1e5 deliberately. An earlier grid capped at 100 selected 100 in every
# fold, and a hyperparameter pinned at the boundary means the grid, not the data, is
# choosing. Extending it confirmed 100 is a genuine interior optimum - every fold still
# picks it - so the selection is trustworthy and the result below is not a grid artifact.
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0)


def build_transitions(panel: pd.DataFrame, values: np.ndarray):
    """Every consecutive (t → t+1) pair a subject contributes.

    Consecutive means adjacent in `TIME_ORDER`. A subject missing 6 months contributes
    3→6 and 6→9 as nothing; the 3→9 jump is NOT manufactured, because a two-step gap is
    a different forecasting problem and pooling it would silently mix horizons.
    """
    rows, x_idx, y_idx = [], [], []
    for subject, group in panel.groupby("subject_id"):
        group = group.sort_values("time_order")
        positions = group.index.to_numpy()
        orders = group["time_order"].to_numpy()
        for a in range(len(orders) - 1):
            if orders[a + 1] - orders[a] != 1:
                continue
            x_idx.append(positions[a])
            y_idx.append(positions[a + 1])
            rows.append(
                {
                    "subject_id": subject,
                    "from_timepoint": TIME_ORDER[orders[a]],
                    "to_timepoint": TIME_ORDER[orders[a + 1]],
                    "transition": f"{TIME_ORDER[orders[a]]} -> {TIME_ORDER[orders[a + 1]]}",
                }
            )
    meta = pd.DataFrame(rows)
    return meta, values[np.array(x_idx)], values[np.array(y_idx)]


def grouped_folds(groups: np.ndarray, n_splits: int, seed: int):
    """Split whole subjects, greedily balancing fold sizes."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    counts = {g: int((groups == g).sum()) for g in unique}
    assignment, load = {}, np.zeros(n_splits, dtype=int)
    for g in sorted(unique, key=lambda g: -counts[g]):
        fold = int(np.argmin(load))
        assignment[g] = fold
        load[fold] += counts[g]

    all_idx = np.arange(len(groups))
    for fold in range(n_splits):
        test_mask = np.array([assignment[g] == fold for g in groups])
        yield all_idx[~test_mask], all_idx[test_mask]


def fit_ridge(x, y, alpha):
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)
    xz, yz = (x - x_mean) / x_std, (y - y_mean) / y_std
    coef = np.linalg.solve(xz.T @ xz + alpha * np.eye(xz.shape[1]), xz.T @ yz)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "coef": coef}


def predict_ridge(model, x):
    xz = (x - model["x_mean"]) / model["x_std"]
    return (xz @ model["coef"]) * model["y_std"] + model["y_mean"]


def select_alpha(x, y, groups, seed):
    scores = {}
    for alpha in RIDGE_ALPHAS:
        errs = []
        for train_idx, valid_idx in grouped_folds(groups, 4, seed):
            model = fit_ridge(x[train_idx], y[train_idx], alpha)
            errs.append(float(np.mean(np.abs(y[valid_idx] - predict_ridge(model, x[valid_idx])))))
        scores[alpha] = float(np.mean(errs))
    return min(scores, key=scores.get)


def one_step(x, y, meta, args):
    """Held-out one-step forecasts under subject-grouped CV."""
    groups = meta["subject_id"].to_numpy()
    oof = np.full_like(y, np.nan)
    oof_mean = np.full_like(y, np.nan)
    chosen = []

    for train_idx, test_idx in grouped_folds(groups, N_SPLITS, SEED):
        alpha = select_alpha(x[train_idx], y[train_idx], groups[train_idx], SEED)
        chosen.append(alpha)
        model = fit_ridge(x[train_idx], y[train_idx], alpha)
        oof[test_idx] = predict_ridge(model, x[test_idx])

        # Mean baseline is per transition: "the average 6-month profile" is not the
        # same target as "the average 12-month profile".
        for label in meta["transition"].unique():
            in_train = train_idx[meta["transition"].to_numpy()[train_idx] == label]
            in_test = test_idx[meta["transition"].to_numpy()[test_idx] == label]
            if len(in_train) == 0 or len(in_test) == 0:
                continue
            oof_mean[in_test] = y[in_train].mean(axis=0)

    return oof, oof_mean, chosen


def iterated_from_preop(panel, values, meta, x, y, metabolites, args):
    """Roll the one-step model forward preop → 3 → 6 → 9 → 12 and score at 12 months.

    Compared against the direct preop→12mo ridge on the SAME subjects, so the two
    numbers are commensurable. Subjects are held out as whole groups in both.
    """
    complete = (
        panel.groupby("subject_id")["time_order"].apply(lambda s: set(s) == set(range(5)))
    )
    subjects = complete[complete].index.to_numpy()
    if len(subjects) < 20:
        return None

    by_subject = {
        (row.subject_id, row.time_order): i for i, row in enumerate(panel.itertuples())
    }
    preop = np.array([values[by_subject[(s, 0)]] for s in subjects])
    final = np.array([values[by_subject[(s, 4)]] for s in subjects])

    groups = meta["subject_id"].to_numpy()
    iterated = np.full_like(final, np.nan)
    direct = np.full_like(final, np.nan)

    for fold, (_, test_subj_idx) in enumerate(grouped_folds(subjects, N_SPLITS, SEED)):
        held = set(subjects[test_subj_idx])
        # Train the one-step model on transitions from subjects NOT held out.
        train_rows = np.flatnonzero(~np.isin(groups, list(held)))
        alpha = select_alpha(x[train_rows], y[train_rows], groups[train_rows], SEED)
        step = fit_ridge(x[train_rows], y[train_rows], alpha)

        state = preop[test_subj_idx].copy()
        for _ in range(4):
            state = predict_ridge(step, state)
        iterated[test_subj_idx] = state

        # Direct preop -> 12mo, trained on the same held-in subjects.
        train_subj = np.array([i for i, s in enumerate(subjects) if s not in held])
        d_alpha = select_alpha(preop[train_subj], final[train_subj],
                               subjects[train_subj], SEED)
        d_model = fit_ridge(preop[train_subj], final[train_subj], d_alpha)
        direct[test_subj_idx] = predict_ridge(d_model, preop[test_subj_idx])

    per_subject = lambda pred: np.mean(np.abs(final - pred), axis=1)
    err_iter, err_direct = per_subject(iterated), per_subject(direct)
    err_persist = per_subject(preop)
    gap = bootstrap_delta_ci(err_iter, err_direct, 5000, SEED)

    return {
        "n_subjects_complete": int(len(subjects)),
        "iterated_mae": float(err_iter.mean()),
        "direct_mae": float(err_direct.mean()),
        "persistence_mae": float(err_persist.mean()),
        "iterated_minus_direct": gap.point,
        "iterated_minus_direct_ci95": [gap.low, gap.high],
        "iterated_beats_direct": bool(gap.high < 0.0),
        "VERDICT": (
            "Walking the trajectory beats jumping straight to 12 months."
            if gap.high < 0.0
            else "Walking the trajectory is not better than forecasting 12 months directly."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--include-artifacts", action="store_true",
                        help="Keep the exogenous compounds. They encode 'this is the "
                             "preop->3mo transition' rather than biology.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel, values, metabolites = load_timepoint_panel()
    dropped = [] if args.include_artifacts else [m for m in ARTIFACT_FEATURES if m in metabolites]
    keep = [i for i, m in enumerate(metabolites) if m not in dropped]
    values = values[:, keep]
    metabolites = [metabolites[i] for i in keep]

    meta, x, y = build_transitions(panel, values)
    oof, oof_mean, alphas = one_step(x, y, meta, args)

    err_model = np.mean(np.abs(y - oof), axis=1)
    err_persist = np.mean(np.abs(y - x), axis=1)
    err_mean = np.mean(np.abs(y - oof_mean), axis=1)

    rows = []
    for label in meta["transition"].unique():
        m = (meta["transition"] == label).to_numpy()
        vs_p = bootstrap_delta_ci(err_model[m], err_persist[m], 4000, SEED)
        vs_m = bootstrap_delta_ci(err_model[m], err_mean[m], 4000, SEED)
        rows.append({
            "transition": label,
            "n": int(m.sum()),
            "model_mae": float(err_model[m].mean()),
            "persistence_mae": float(err_persist[m].mean()),
            "mean_baseline_mae": float(err_mean[m].mean()),
            "vs_persistence": vs_p.point,
            "vs_persistence_ci_low": vs_p.low,
            "vs_persistence_ci_high": vs_p.high,
            "beats_persistence": bool(vs_p.high < 0.0),
            "vs_mean": vs_m.point,
            "beats_mean_baseline": bool(vs_m.high < 0.0),
        })
    per_transition = pd.DataFrame(rows)

    overall_p = bootstrap_delta_ci(err_model, err_persist, 5000, SEED)
    overall_m = bootstrap_delta_ci(err_model, err_mean, 5000, SEED)
    iterated = iterated_from_preop(panel, values, meta, x, y, metabolites, args)

    summary = {
        "task": "one-step autoregressive forecast across consecutive follow-ups",
        "n_samples_in_panel": int(len(panel)),
        "n_subjects": int(panel["subject_id"].nunique()),
        "n_transitions": int(len(meta)),
        "transitions_per_step": per_transition.set_index("transition")["n"].to_dict(),
        "features_used": len(metabolites),
        "artifact_features_dropped": dropped,
        "evaluation": f"subject-grouped {N_SPLITS}-fold CV; a subject never spans folds",
        "alphas_selected": alphas,
        "model_mae": float(err_model.mean()),
        "persistence_mae": float(err_persist.mean()),
        "mean_baseline_mae": float(err_mean.mean()),
        "CONTROL_vs_persistence": overall_p.point,
        "CONTROL_vs_persistence_ci95": [overall_p.low, overall_p.high],
        "CONTROL_beats_persistence": bool(overall_p.high < 0.0),
        "CONTROL_vs_mean_baseline": overall_m.point,
        "CONTROL_vs_mean_baseline_ci95": [overall_m.low, overall_m.high],
        "CONTROL_beats_mean_baseline": bool(overall_m.high < 0.0),
        "iterated_vs_direct": iterated,
    }
    summary["VERDICT"] = (
        "One-step AR beats both persistence and the per-timepoint mean."
        if summary["CONTROL_beats_persistence"] and summary["CONTROL_beats_mean_baseline"]
        else "One-step AR beats persistence but not the per-timepoint mean."
        if summary["CONTROL_beats_persistence"]
        else "One-step AR does not beat carrying the last observation forward."
    )

    predictions = meta.copy()
    predictions["mae_model"] = err_model
    predictions["mae_persistence"] = err_persist
    predictions["mae_mean_baseline"] = err_mean

    np.savez_compressed(
        args.output_dir / "ar_errors.npz",
        model=err_model, persistence=err_persist, mean_baseline=err_mean,
        transition=np.array(meta["transition"], dtype=object),
    )
    predictions.to_csv(args.output_dir / "transition_predictions.csv", index=False)
    per_transition.to_csv(args.output_dir / "per_transition_metrics.csv", index=False)
    (args.output_dir / "summary_ar.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
