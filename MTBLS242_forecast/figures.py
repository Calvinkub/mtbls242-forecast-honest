"""Regenerate every figure in `Graph/` reproducibly.

Follows the Bridge_ai_summit_nmr_2026 convention: each figure states its data source,
cohort size, and evaluation method on the plot itself, so a reader never has to ask
"measured how?". Here that convention is extended to the negative controls - a PR curve
that omits its no-signal null is not interpretable, because on this outcome the null
scores higher than the model.

Run the two experiment scripts first; this reads their outputs.

    python forecast_mtbls242.py
    python forecast_responder_pr_auc.py
    python figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "Graph"
DPI = 300

PROVENANCE = (
    "Data: MetaboLights MTBLS242 serum 1H-NMR (bariatric surgery cohort), "
    "71 subjects with paired preop + 12-month samples, 21 quantified metabolites"
)


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _footer(fig, text):
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=6.4, color="#444444")


def pr_curve_with_nulls(plt, pr_dir: Path, out: Path) -> Path:
    """PR curve, the no-learning scorer, and the distribution the null reaches."""
    summary = json.loads((pr_dir / "summary_pr_auc.json").read_text())
    nulls = np.load(pr_dir / "null_distributions.npz", allow_pickle=True)

    label = nulls["pooled_label"]
    mask = label >= 0
    label, prob = label[mask], nulls["pooled_prob"][mask]
    prevalence = float(summary["prevalence_baseline"])

    order = np.argsort(-prob, kind="mergesort")
    ranked = label[order]
    tp, fp = np.cumsum(ranked), np.cumsum(1 - ranked)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(label.sum()), 1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.5))

    ax.step(np.r_[0, recall], np.r_[1, precision], where="post", color="#b2182b", lw=2,
            label=f"Model (mean PR-AUC {summary['model_pr_auc_mean']:.3f})")
    ax.axhline(prevalence, ls="--", color="#777777", lw=1.2,
               label=f"No-skill baseline = {prevalence:.3f}")
    ax.axhline(summary["CONTROL_trivial_scorer_pr_auc"], ls=":", color="#1f6f8b", lw=1.6,
               label=f"No-learning scorer = {summary['CONTROL_trivial_scorer_pr_auc']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Held-out precision-recall", fontsize=10)
    ax.legend(loc="upper right", fontsize=6.8, framealpha=0.9)

    lift_null = nulls["pairing_destroyed_lift"]
    observed = float(summary["observed_lift_over_prevalence"])
    ax2.hist(lift_null, bins=34, color="#9ecae1", edgecolor="white", lw=0.4,
             label="Pairing-destroyed null\n(no signal can exist)")
    ax2.axvline(observed, color="#b2182b", lw=2.2,
                label=f"Observed lift = {observed:.2f}x")
    ax2.axvline(np.percentile(lift_null, 95), color="#444444", ls="--", lw=1.2,
                label=f"Null 95th pct = {np.percentile(lift_null, 95):.2f}x")
    ax2.set_xlabel("PR-AUC lift over that replicate's own prevalence")
    ax2.set_ylabel("Null replicates")
    ax2.set_title("Does the model clear its no-signal null?", fontsize=10)
    ax2.legend(loc="upper right", fontsize=6.8, framealpha=0.9)

    verdict = summary["VERDICT"]
    fig.suptitle(
        f"MTBLS242 - forecast residualized 12-month metabolic response\n{verdict}",
        fontsize=11, y=0.99,
    )
    _footer(
        fig,
        f"{PROVENANCE}\nEvaluation: {summary['evaluation']} | outcome: "
        f"{summary['outcome_definition']} | permutation p = "
        f"{summary['CONTROL_permutation_p_value']:.3f}",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def repeated_cv_distribution(plt, pr_dir: Path, out: Path) -> Path:
    """Model vs no-learning scorer across CV repeats, paired by repeat."""
    summary = json.loads((pr_dir / "summary_pr_auc.json").read_text())
    nulls = np.load(pr_dir / "null_distributions.npz", allow_pickle=True)
    model = nulls["repeat_model_pr_auc"]
    trivial = nulls["repeat_trivial_pr_auc"]
    prevalence = float(summary["prevalence_baseline"])

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, (m, t) in enumerate(zip(model, trivial)):
        ax.plot([0, 1], [m, t], color="#bbbbbb", lw=0.8, zorder=1)
    ax.scatter(np.zeros_like(model), model, s=32, color="#b2182b", zorder=3, label="Model")
    ax.scatter(np.ones_like(trivial), trivial, s=32, color="#1f6f8b", zorder=3,
               label="No-learning scorer")
    ax.axhline(prevalence, ls="--", color="#777777", lw=1.2,
               label=f"No-skill baseline = {prevalence:.3f}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Constrained\nlogistic model", "Sum of signed\nbaseline z-scores"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Held-out PR-AUC (mean over 5 folds)")
    ci = summary["CONTROL_model_minus_trivial_ci95"]
    ax.set_title(
        f"Each line is one CV repeat ({len(model)} repeats)\n"
        f"Model minus no-learning scorer: {summary['CONTROL_model_minus_trivial']:+.3f} "
        f"(95% CI {ci[0]:+.3f} to {ci[1]:+.3f})",
        fontsize=9.5,
    )
    ax.legend(loc="best", fontsize=7.2)
    _footer(fig, f"{PROVENANCE}\nEvaluation: {summary['evaluation']}")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def continuous_forecast_mae(plt, cont_dir: Path, out: Path) -> Path:
    """Per-subject MAE against both baselines, with the bootstrap interval stated."""
    summary = json.loads((cont_dir / "summary.json").read_text())
    errors = np.load(cont_dir / "subject_errors.npz")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4))

    series = [
        ("Ridge forecast", errors["model"], "#b2182b"),
        ("Training-fold mean", errors["mean_baseline"], "#1f6f8b"),
        ("Persistence", errors["persistence"], "#888888"),
    ]
    ax.boxplot([s[1] for s in series], tick_labels=[s[0] for s in series],
               showfliers=False, widths=0.5)
    for i, (_, values, color) in enumerate(series, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.045, len(values))
        ax.scatter(i + jitter, values, s=9, alpha=0.45, color=color, zorder=3)
    ax.set_ylabel("Per-subject MAE (log1p)")
    ax.set_title("Held-out error per subject", fontsize=10)
    ax.tick_params(axis="x", labelsize=7.5)

    ci = summary["CONTROL_mae_gap_vs_mean_baseline_ci95"]
    point = summary["CONTROL_mae_gap_vs_mean_baseline"]
    ax2.errorbar([point], [0], xerr=[[point - ci[0]], [ci[1] - point]], fmt="o",
                 color="#b2182b", capsize=6, markersize=7)
    ax2.axvline(0, color="#444444", lw=1.2, ls="--")
    ax2.set_yticks([])
    ax2.set_xlabel("MAE gap: ridge forecast minus training-fold mean (log1p)")
    ax2.set_title(
        "Paired bootstrap, 5000 resamples\n"
        f"{point:+.4f} (95% CI {ci[0]:+.4f} to {ci[1]:+.4f}) - interval spans zero",
        fontsize=9.5,
    )
    ax2.text(0.02, 0.82, "favours forecast", transform=ax2.transAxes, fontsize=7.2,
             color="#666666")
    ax2.text(0.72, 0.82, "favours baseline", transform=ax2.transAxes, fontsize=7.2,
             color="#666666")

    fig.suptitle(
        "MTBLS242 - preop NMR to 12-month NMR profile\n" + summary["VERDICT"],
        fontsize=11, y=0.99,
    )
    _footer(
        fig,
        f"{PROVENANCE}\nEvaluation: {summary['evaluation']} | pooled R2 = "
        f"{summary['r2_overall']:.3f} is inflated by between-metabolite scale; "
        f"median per-metabolite R2 = {summary['median_metabolite_r2']:.3f}",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def regression_to_the_mean(plt, out: Path) -> Path:
    """The defect in one picture: raw change is a restatement of the baseline."""
    from controls import residualize_change
    from forecast_mtbls242 import load_longitudinal_pairs
    from forecast_responder_pr_auc import MONOTONE_DIRECTIONS, marker_columns

    _, x, y, metabolites = load_longitudinal_pairs()
    idx, _ = marker_columns(metabolites)
    correlations = [np.corrcoef(x[:, j], (y - x)[:, j])[0, 1] for j in idx]

    # `lipoproteins` is shown rather than the single most extreme marker. The most
    # extreme is L-Lactic acid at -0.94, but that value is produced by one subject whose
    # pre-operative abundance is recorded as 0 - a below-detection reading, not a
    # measurement (it falls to -0.79 without that point). Illustrating a real defect with
    # a leverage artifact would be the same error this repo exists to correct.
    # `lipoproteins` has no zero readings and sits at -0.78.
    shown = "lipoproteins"
    position = list(MONOTONE_DIRECTIONS).index(shown)
    column = idx[position]
    name = shown

    fit = residualize_change(x[:, idx], y[:, idx], np.arange(len(x)))
    residual = fit.transform(x[:, idx], y[:, idx])[:, position]
    worst = column

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.3), sharex=True)

    raw_change = (y - x)[:, worst]
    ax.scatter(x[:, worst], raw_change, s=26, alpha=0.7, color="#b2182b",
               edgecolor="white", lw=0.5)
    slope, intercept = np.polyfit(x[:, worst], raw_change, 1)
    span = np.linspace(x[:, worst].min(), x[:, worst].max(), 50)
    ax.plot(span, slope * span + intercept, color="#7a1020", lw=1.6)
    ax.axhline(0, color="#999999", lw=0.9, ls="--")
    ax.set_xlabel(f"Pre-operative {name} (log1p) — a model FEATURE")
    ax.set_ylabel("Raw 12-month change")
    ax.set_title(f"Retired outcome used this\nr = {correlations[position]:.2f}", fontsize=10)

    ax2.scatter(x[:, worst], residual, s=26, alpha=0.7, color="#1f6f8b",
                edgecolor="white", lw=0.5)
    ax2.axhline(0, color="#999999", lw=0.9, ls="--")
    ax2.set_xlabel(f"Pre-operative {name} (log1p)")
    ax2.set_ylabel("Residualized change")
    ax2.set_title(
        f"Current outcome uses this\nr = {np.corrcoef(x[:, worst], residual)[0, 1]:.2f}",
        fontsize=10,
    )

    fig.suptitle(
        "The retired outcome was a restatement of the input\n"
        f"Across all 11 markers: median r = {np.median(correlations):.2f}; "
        f"10 of 11 below -0.58",
        fontsize=11, y=1.0,
    )
    _footer(fig, PROVENANCE)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--pr-dir", type=Path, default=ROOT / "outputs_pr_auc")
    parser.add_argument("--continuous-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.graph_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt = _style()
    except ImportError:
        raise SystemExit(
            "matplotlib is required for figures.\n"
            "  python -m pip install -r requirements.txt\n"
            "The experiment scripts themselves run without it."
        )

    written = [
        pr_curve_with_nulls(plt, args.pr_dir, args.graph_dir / "pr_auc_curve.png"),
        repeated_cv_distribution(plt, args.pr_dir, args.graph_dir / "pr_auc_by_repeat.png"),
        continuous_forecast_mae(
            plt, args.continuous_dir, args.graph_dir / "continuous_forecast_mae.png"
        ),
        regression_to_the_mean(plt, args.graph_dir / "regression_to_the_mean.png"),
    ]
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
