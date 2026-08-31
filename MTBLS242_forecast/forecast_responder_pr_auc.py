"""Forecast a strong 12-month metabolic response from pre-operative serum NMR.

WHAT CHANGED AND WHY
--------------------
An earlier version of this script reported held-out PR-AUC 0.671 against a 0.437
prevalence baseline and called it a 1.54x lift. That claim has been retired; see
`CORRECTION.md`. Three defects produced it:

1. The outcome was `y_12mo - x_preop` on the same 11 metabolites used as features, so
   `corr(baseline, change)` was -0.62 by arithmetic. A subject with an extreme baseline
   falls back toward the mean whether or not anything biological happened.
2. When subject pairing was destroyed - so no forecastable signal could exist - the
   score went *up*, to 0.741. The headline sat below its own no-signal null.
3. The solver never converged, so the "monotonic constraints" described a model whose
   largest standardized coefficient was 0.018.

This version predicts **residualized** improvement: a marker counts as improved when it
improved more than its own baseline predicts. The residualization is fitted inside each
training fold, because it is part of the label. Everything is reported next to the
controls in `controls.py`, and the no-learning scorer is printed alongside the model so
the two can never again be confused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from controls import (
    average_precision,
    bootstrap_delta_ci,
    label_permutation_null,
    pairing_destroyed_null,
    repeated_stratified_cv,
    residualize_change,
    trivial_signed_baseline_score,
)
from forecast_mtbls242 import load_longitudinal_pairs
from monotone_logistic import fit_monotone_logistic

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_pr_auc"
SEED = 42
N_SPLITS = 5
N_REPEATS = 20
L2_VALUES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

# Direction each marker moves in obesity, from the obesity/insulin-resistance
# metabolomics literature (Newgard 2009; Wang 2011 Nat Med; Wuertz et al.), carried over
# from the Bridge_ai_summit_nmr_2026 Experiment A constraint set. "Improved" means the
# residual moves opposite to this direction.
MONOTONE_DIRECTIONS = {
    "L-valine": 1,
    "L-leucine": 1,
    "L-allo-Isoleucine": 1,
    "L-tyrosine": 1,
    "D-phenylalanine": 1,
    "L-alanine": 1,
    "lipoproteins": 1,
    "L-Lactic acid": 1,
    "glycine": -1,
    "L-glutamine": -1,
    "histidine": -1,
}

# Exogenous compounds, not obesity biology. Bridge_ai_summit_nmr_2026 flags isopropanol
# as a surgical skin-prep contaminant; confirmed here, preop mean 4.529 vs 12-month
# 3.205 (log1p). Their pre-operative level partly encodes "this sample was drawn in an
# operating theatre", which is a procedure artifact a forecast must not lean on.
ARTIFACT_FEATURES = ("isopropanol", "methanol", "Dimethyl sulfone")

# Pre-registered before any model was fitted. On the residualized scale a marker
# improves for about half of subjects by construction, so the legacy 9-of-11 rule would
# leave too few positives at n=71. 6 of 11 reproduces the originally reported
# prevalence exactly - 31 of 71, 0.437 - so the corrected PR-AUC is directly comparable
# to the superseded one rather than being confounded by a shift in base rate.
MIN_IMPROVED_MARKERS_RESIDUAL = 6
MIN_IMPROVED_MARKERS_LEGACY = 9


def marker_columns(metabolites: list[str]) -> tuple[list[int], np.ndarray]:
    idx = [metabolites.index(name) for name in MONOTONE_DIRECTIONS]
    signs = np.array([MONOTONE_DIRECTIONS[name] for name in MONOTONE_DIRECTIONS])
    return idx, signs


def legacy_outcome(x, y, metabolites, threshold=MIN_IMPROVED_MARKERS_LEGACY):
    """The superseded outcome. Retained only so the correction stays reproducible."""
    idx, signs = marker_columns(metabolites)
    improved = ((y[:, idx] - x[:, idx]) * signs < 0).sum(axis=1)
    return (improved >= threshold).astype(int), improved


def residual_outcome(x, y, metabolites, train_idx, threshold=MIN_IMPROVED_MARKERS_RESIDUAL):
    """Improvement beyond what the baseline mechanically predicts.

    `train_idx` selects the rows the change-on-baseline fit may see. Held-out rows are
    scored with the training fold's fit, never their own.
    """
    idx, signs = marker_columns(metabolites)
    fit = residualize_change(x[:, idx], y[:, idx], train_idx)
    residual = fit.transform(x[:, idx], y[:, idx])
    improved = (residual * signs < 0).sum(axis=1)
    return (improved >= threshold).astype(int), improved


def select_l2(x, y, constraints, seed):
    """Inner 4-fold selection of the ridge penalty, on training rows only."""
    scores = {}
    for l2 in L2_VALUES:
        aps = []
        for _, _, inner_train, inner_valid in repeated_stratified_cv(y, 4, 1, seed):
            if y[inner_train].sum() == 0 or y[inner_valid].sum() == 0:
                continue
            model = fit_monotone_logistic(x[inner_train], y[inner_train], constraints, l2)
            aps.append(average_precision(y[inner_valid], model.predict_proba(x[inner_valid])))
        scores[l2] = float(np.nanmean(aps)) if aps else float("nan")
    return max(scores, key=lambda k: (scores[k] if np.isfinite(scores[k]) else -np.inf))


def evaluate(x, y_future, metabolites, feature_idx, constraints, args):
    """Repeated stratified CV. Returns per-fold rows and pooled per-repeat predictions."""
    marker_idx, signs = marker_columns(metabolites)
    # Marker positions expressed in the (possibly reduced) feature matrix.
    feature_marker_idx = [feature_idx.index(j) for j in marker_idx if j in feature_idx]
    feature_marker_signs = np.array(
        [s for j, s in zip(marker_idx, signs) if j in feature_idx]
    )
    x_features = x[:, feature_idx]

    if args.legacy_outcome:
        stratify_labels, _ = legacy_outcome(x, y_future, metabolites)
    else:
        stratify_labels, _ = residual_outcome(x, y_future, metabolites, np.arange(len(x)))

    rows, pooled = [], {}
    n_converged, n_fits = 0, 0

    for repeat, fold, train_idx, test_idx in repeated_stratified_cv(
        stratify_labels, N_SPLITS, args.repeats, SEED
    ):
        if args.legacy_outcome:
            labels = stratify_labels
        else:
            labels, _ = residual_outcome(x, y_future, metabolites, train_idx)

        y_train, y_test = labels[train_idx], labels[test_idx]
        if y_train.sum() == 0 or y_test.sum() == 0:
            continue

        l2 = select_l2(x_features[train_idx], y_train, constraints, SEED + fold)
        model = fit_monotone_logistic(x_features[train_idx], y_train, constraints, l2)
        n_fits += 1
        n_converged += int(model.converged)

        prob = model.predict_proba(x_features[test_idx])
        trivial = trivial_signed_baseline_score(
            x_features, feature_marker_idx, feature_marker_signs
        )[test_idx]

        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_positive_test": int(y_test.sum()),
                "l2": l2,
                "converged": model.converged,
                "model_pr_auc": average_precision(y_test, prob),
                "trivial_pr_auc": average_precision(y_test, trivial),
                "baseline_prevalence": float(y_test.mean()),
            }
        )
        store = pooled.setdefault(
            repeat,
            {"prob": np.full(len(x), np.nan), "trivial": np.full(len(x), np.nan),
             "label": np.full(len(x), -1)},
        )
        store["prob"][test_idx] = prob
        store["trivial"][test_idx] = trivial
        store["label"][test_idx] = y_test

    return pd.DataFrame(rows), pooled, n_converged, n_fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--null-reps", type=int, default=500)
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Keep isopropanol/methanol/dimethyl sulfone as features (legacy behaviour).",
    )
    parser.add_argument(
        "--legacy-outcome",
        action="store_true",
        help="Use the superseded baseline-dependent outcome. Reproduces the retired 0.671.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired, x, y_future, metabolites = load_longitudinal_pairs()
    marker_idx, signs = marker_columns(metabolites)

    dropped = [] if args.include_artifacts else [m for m in ARTIFACT_FEATURES if m in metabolites]
    feature_idx = [j for j, m in enumerate(metabolites) if m not in dropped]
    feature_names = [metabolites[j] for j in feature_idx]
    constraints = np.array([MONOTONE_DIRECTIONS.get(m, 0) for m in feature_names])

    fold_metrics, pooled, n_converged, n_fits = evaluate(
        x, y_future, metabolites, feature_idx, constraints, args
    )

    per_repeat = fold_metrics.groupby("repeat")[["model_pr_auc", "trivial_pr_auc"]].mean()
    model_ap = float(per_repeat["model_pr_auc"].mean())
    trivial_ap = float(per_repeat["trivial_pr_auc"].mean())

    # --- Controls -------------------------------------------------------------
    # Does the OUTCOME DEFINITION reward baseline knowledge even with no real signal?
    def outcome_fn(xx, yy):
        labels, _ = (
            legacy_outcome(xx, yy, metabolites)
            if args.legacy_outcome
            else residual_outcome(xx, yy, metabolites, np.arange(len(xx)))
        )
        return labels

    def scorer(xx):
        return trivial_signed_baseline_score(xx[:, feature_idx],
                                             [feature_idx.index(j) for j in marker_idx
                                              if j in feature_idx],
                                             np.array([s for j, s in zip(marker_idx, signs)
                                                       if j in feature_idx]))

    null = pairing_destroyed_null(x, y_future, outcome_fn, scorer, args.null_reps, SEED)

    # Is the MODEL's ranking significant? Pooled held-out predictions from repeat 1.
    first = pooled[min(pooled)]
    mask = first["label"] >= 0
    permutation = label_permutation_null(
        first["label"][mask], first["prob"][mask], n_reps=2000, seed=SEED
    )

    # Does the model beat the no-learning scorer, per repeat?
    delta = bootstrap_delta_ci(
        per_repeat["model_pr_auc"].to_numpy(),
        per_repeat["trivial_pr_auc"].to_numpy(),
        n_reps=5000,
        seed=SEED,
    )

    final_l2 = select_l2(x[:, feature_idx], outcome_fn(x, y_future), constraints, SEED + 100)
    final_labels = outcome_fn(x, y_future)
    final_model = fit_monotone_logistic(x[:, feature_idx], final_labels, constraints, final_l2)
    binding = final_model.binding_constraints(constraints)
    prevalence = float(final_labels.mean())
    observed_lift = model_ap / prevalence

    summary = {
        "task": "forecast residualized 12-month metabolic response from preop NMR",
        "outcome_definition": (
            "legacy: >=9/11 markers move in the literature direction (SUPERSEDED)"
            if args.legacy_outcome
            else f">={MIN_IMPROVED_MARKERS_RESIDUAL}/11 markers improve MORE than their own "
                 "baseline predicts (change residualized on baseline, fitted in-fold)"
        ),
        "n_subjects": int(len(x)),
        "n_responders": int(final_labels.sum()),
        "prevalence_baseline": prevalence,
        "features_used": len(feature_idx),
        "artifact_features_dropped": dropped,
        "artifact_drop_reason": (
            "exogenous compounds; isopropanol is a surgical skin-prep contaminant "
            "(Bridge_ai_summit_nmr_2026, Experiment A caveats)"
        ),
        "metric": "PR-AUC (average precision)",
        "evaluation": f"{args.repeats} repeats x {N_SPLITS}-fold stratified CV, held-out only",
        "model_pr_auc_mean": model_ap,
        "model_pr_auc_std_across_repeats": float(per_repeat["model_pr_auc"].std(ddof=0)),
        "observed_lift_over_prevalence": observed_lift,
        "CONTROL_trivial_scorer_pr_auc": trivial_ap,
        "CONTROL_model_minus_trivial": delta.point,
        "CONTROL_model_minus_trivial_ci95": [delta.low, delta.high],
        "CONTROL_model_beats_trivial": not delta.spans_zero and delta.point > 0,
        "CONTROL_pairing_destroyed_null_lift_median": float(np.median(null.lift)),
        "CONTROL_pairing_destroyed_null_lift_p95": null.percentile(95),
        "CONTROL_observed_lift_clears_null": bool(observed_lift > null.percentile(95)),
        "CONTROL_permutation_p_value": permutation.p_value,
        "solver_convergence_rate": n_converged / max(n_fits, 1),
        "monotone_positive": int(np.sum(constraints == 1)),
        "monotone_negative": int(np.sum(constraints == -1)),
        "unconstrained": int(np.sum(constraints == 0)),
        "binding_constraints": [feature_names[i] for i in np.flatnonzero(binding)],
        "max_abs_standardized_coefficient": float(np.max(np.abs(final_model.weights))),
    }
    summary["VERDICT"] = (
        "Signal beyond regression to the mean."
        if summary["CONTROL_observed_lift_clears_null"]
        and summary["CONTROL_model_beats_trivial"]
        and permutation.p_value < 0.05
        else "No evidence of forecasting skill beyond regression to the mean."
    )

    # --- Outputs --------------------------------------------------------------
    predictions = paired[["subject_id", "Sample Name_preop", "Sample Name_12mo"]].copy()
    predictions["responder"] = final_labels
    predictions["heldout_probability_repeat1"] = first["prob"]
    predictions["trivial_score_repeat1"] = first["trivial"]

    coefficients = pd.DataFrame(
        {
            "metabolite": feature_names,
            "constraint": constraints,
            "standardized_coefficient": final_model.weights,
            "constraint_is_binding": binding,
        }
    ).sort_values("standardized_coefficient", ascending=False)

    np.savez_compressed(
        args.output_dir / "monotone_responder_model.npz",
        x_mean=final_model.x_mean,
        x_std=final_model.x_std,
        weights=final_model.weights,
        intercept=final_model.intercept,
        l2=final_model.l2,
        metabolites=np.array(feature_names, dtype=object),
        constraints=constraints,
    )
    np.savez_compressed(
        args.output_dir / "null_distributions.npz",
        pairing_destroyed_lift=null.lift,
        pairing_destroyed_prevalence=null.prevalence,
        permutation_null_ap=permutation.null_average_precision,
        repeat_model_pr_auc=per_repeat["model_pr_auc"].to_numpy(),
        repeat_trivial_pr_auc=per_repeat["trivial_pr_auc"].to_numpy(),
        pooled_label=first["label"],
        pooled_prob=first["prob"],
    )
    predictions.to_csv(args.output_dir / "heldout_responder_predictions.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "fold_pr_auc.csv", index=False)
    coefficients.to_csv(args.output_dir / "monotone_coefficients.csv", index=False)
    (args.output_dir / "summary_pr_auc.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
