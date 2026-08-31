"""Forecast the 12-month MTBLS242 serum NMR profile from pre-operative NMR.

This is a longitudinal forecast: each row is one subject with a pre-operative
measurement (features) and that same subject's 12-month measurement (targets).
It is not a disease classifier because MTBLS242 contains no future disease
incidence outcome.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from controls import bootstrap_delta_ci


ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = ROOT / "s_MTBLS242.txt"
MAF_FILE = ROOT / "m_MTBLS242_v2_maf.tsv"
OUTPUT_DIR = ROOT / "outputs"
SEED = 42
N_SPLITS = 5
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def subject_id(sample_name: str) -> str:
    """Extract the stable four-digit subject code from an MTBLS242 sample ID."""
    match = re.search(r"(?:^|[_-])(\d{4})(?:[_-]|$)", str(sample_name))
    if not match:
        raise ValueError(f"Cannot recover subject ID from sample name: {sample_name}")
    return match.group(1)


TIME_ORDER = (
    "preop",
    "3 months after surgery",
    "6 months after surgery",
    "9 months after surgery",
    "12 months after surgery",
)


def _read_panel() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Shared parsing: sample sheet joined to the metabolite abundance table."""
    samples = pd.read_csv(SAMPLE_FILE, sep="\t", dtype=str)
    required = {"Sample Name", "Factor Value[time point]"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Sample sheet is missing columns: {sorted(missing)}")

    samples = samples[["Sample Name", "Factor Value[time point]"]].copy()
    samples["subject_id"] = samples["Sample Name"].map(subject_id)

    maf = pd.read_csv(MAF_FILE, sep="\t", low_memory=False)
    if "metabolite_identification" not in maf.columns:
        raise ValueError("MAF is missing metabolite_identification")

    sample_columns = [c for c in maf.columns if c in set(samples["Sample Name"])]
    if not sample_columns:
        raise ValueError("No sample IDs overlap between the sample sheet and MAF")

    metabolites = maf["metabolite_identification"].astype(str).tolist()
    abundance = maf.set_index("metabolite_identification")[sample_columns].T
    abundance = abundance.apply(pd.to_numeric, errors="coerce")
    abundance.index.name = "Sample Name"
    return samples, abundance, metabolites


def load_timepoint_panel() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Every sample at every time point, not just the preop/12-month pair.

    Returns `(index, values, metabolites)` where `index` has one row per sample with
    `subject_id`, `timepoint`, and `time_order`, aligned row-for-row with `values`
    (log1p abundances). This is what the autoregressive model uses: the pairwise loader
    discards 3/6/9 months entirely, which is 323 of the 465 samples.
    """
    samples, abundance, metabolites = _read_panel()

    panel = samples.loc[
        samples["Sample Name"].isin(abundance.index)
        & samples["subject_id"].notna()
        & samples["Factor Value[time point]"].isin(TIME_ORDER)
    ].copy()
    panel = panel.rename(columns={"Factor Value[time point]": "timepoint"})
    panel["time_order"] = panel["timepoint"].map({t: i for i, t in enumerate(TIME_ORDER)})
    panel = panel.sort_values(["subject_id", "time_order"]).reset_index(drop=True)

    if panel.duplicated(["subject_id", "timepoint"]).any():
        raise ValueError("A subject has more than one sample at the same time point")

    values = abundance.loc[panel["Sample Name"]].to_numpy(dtype=float)
    values = np.log1p(np.clip(values, 0, None))

    valid = np.isfinite(values).all(axis=0)
    values = values[:, valid]
    metabolites = [m for m, keep in zip(metabolites, valid) if keep]
    if not metabolites:
        raise ValueError("No complete metabolite features remain after numeric validation")

    return panel[["subject_id", "Sample Name", "timepoint", "time_order"]], values, metabolites


def load_longitudinal_pairs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    samples = pd.read_csv(SAMPLE_FILE, sep="\t", dtype=str)
    required = {"Sample Name", "Factor Value[time point]"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Sample sheet is missing columns: {sorted(missing)}")

    samples = samples[["Sample Name", "Factor Value[time point]"]].copy()
    samples["subject_id"] = samples["Sample Name"].map(subject_id)

    maf = pd.read_csv(MAF_FILE, sep="\t", low_memory=False)
    if "metabolite_identification" not in maf.columns:
        raise ValueError("MAF is missing metabolite_identification")

    sample_columns = [c for c in maf.columns if c in set(samples["Sample Name"])]
    if not sample_columns:
        raise ValueError("No sample IDs overlap between the sample sheet and MAF")

    metabolites = maf["metabolite_identification"].astype(str).tolist()
    abundance = maf.set_index("metabolite_identification")[sample_columns].T
    abundance = abundance.apply(pd.to_numeric, errors="coerce")
    abundance.index.name = "Sample Name"

    preop = samples.loc[samples["Factor Value[time point]"].eq("preop")]
    month12 = samples.loc[
        samples["Factor Value[time point]"].eq("12 months after surgery")
    ]
    paired = preop.merge(month12, on="subject_id", suffixes=("_preop", "_12mo"))
    paired = paired.loc[
        paired["Sample Name_preop"].isin(abundance.index)
        & paired["Sample Name_12mo"].isin(abundance.index)
    ].sort_values("subject_id").reset_index(drop=True)

    if paired["subject_id"].duplicated().any():
        raise ValueError("More than one preop/12-month pair was found for a subject")
    if len(paired) < 25:
        raise ValueError(f"Only {len(paired)} paired subjects; forecast is not viable")

    x = abundance.loc[paired["Sample Name_preop"]].to_numpy(dtype=float)
    y = abundance.loc[paired["Sample Name_12mo"]].to_numpy(dtype=float)
    x = np.log1p(np.clip(x, 0, None))
    y = np.log1p(np.clip(y, 0, None))

    valid = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)
    x, y = x[:, valid], y[:, valid]
    metabolites = [m for m, keep in zip(metabolites, valid) if keep]
    if not metabolites:
        raise ValueError("No complete metabolite features remain after numeric validation")

    return paired, x, y, metabolites


def folds(n_rows: int, n_splits: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [part.astype(int) for part in np.array_split(rng.permutation(n_rows), n_splits)]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)
    xz, yz = (x - x_mean) / x_std, (y - y_mean) / y_std
    penalty = alpha * np.eye(xz.shape[1])
    coef = np.linalg.solve(xz.T @ xz + penalty, xz.T @ yz)
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "coef": coef,
    }


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xz = (x - model["x_mean"]) / model["x_std"]
    return (xz @ model["coef"]) * model["y_std"] + model["y_mean"]


def select_alpha(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    inner_folds = folds(len(x), 4, seed)
    scores: dict[float, float] = {}
    all_idx = np.arange(len(x))
    for alpha in RIDGE_ALPHAS:
        fold_mae = []
        for valid_idx in inner_folds:
            train_idx = np.setdiff1d(all_idx, valid_idx, assume_unique=True)
            model = fit_ridge(x[train_idx], y[train_idx], alpha)
            pred = predict_ridge(model, x[valid_idx])
            fold_mae.append(float(np.mean(np.abs(y[valid_idx] - pred))))
        scores[alpha] = float(np.mean(fold_mae))
    return min(scores, key=scores.get)


def r2_score(y: np.ndarray, pred: np.ndarray, axis=None) -> np.ndarray:
    ss_res = np.sum((y - pred) ** 2, axis=axis)
    mean = np.mean(y, axis=axis, keepdims=True) if axis is not None else np.mean(y)
    ss_tot = np.sum((y - mean) ** 2, axis=axis)
    return 1.0 - np.divide(ss_res, ss_tot, out=np.full_like(ss_res, np.nan), where=ss_tot > 0)


def cross_validate(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return held-out ridge predictions, held-out mean-baseline predictions, and folds.

    The mean baseline is returned per subject, not just as a fold average, so the gap
    between the two can be given a paired bootstrap interval. A 2.4% MAE improvement on
    71 subjects is well inside the range that noise produces, and reporting it without
    an interval invites it to be read as a result.
    """
    outer_folds = folds(len(x), N_SPLITS, SEED)
    oof = np.full_like(y, np.nan)
    oof_mean = np.full_like(y, np.nan)
    rows = []
    all_idx = np.arange(len(x))

    for fold_no, test_idx in enumerate(outer_folds, start=1):
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        alpha = select_alpha(x[train_idx], y[train_idx], SEED + fold_no)
        model = fit_ridge(x[train_idx], y[train_idx], alpha)
        pred = predict_ridge(model, x[test_idx])
        oof[test_idx] = pred

        mean_pred = np.repeat(y[train_idx].mean(axis=0, keepdims=True), len(test_idx), axis=0)
        oof_mean[test_idx] = mean_pred
        rows.append(
            {
                "fold": fold_no,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "alpha": alpha,
                "mae_log1p": float(np.mean(np.abs(y[test_idx] - pred))),
                "rmse_log1p": float(np.sqrt(np.mean((y[test_idx] - pred) ** 2))),
                "mean_baseline_mae_log1p": float(np.mean(np.abs(y[test_idx] - mean_pred))),
                "persistence_mae_log1p": float(np.mean(np.abs(y[test_idx] - x[test_idx]))),
            }
        )

    return oof, oof_mean, pd.DataFrame(rows)


def save_plot(y: np.ndarray, pred: np.ndarray, path: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Dependency-free SVG fallback so the result remains viewable anywhere.
        svg_path = path.with_suffix(".svg")
        width, height, margin = 760, 680, 85
        low = float(min(y.min(), pred.min()))
        high = float(max(y.max(), pred.max()))
        span = max(high - low, 1e-9)

        def sx(value: float) -> float:
            return margin + (value - low) / span * (width - 2 * margin)

        def sy(value: float) -> float:
            return height - margin - (value - low) / span * (height - 2 * margin)

        circles = "\n".join(
            f'<circle cx="{sx(float(obs)):.2f}" cy="{sy(float(est)):.2f}" '
            'r="2.2" fill="#087f8c" fill-opacity="0.30" />'
            for obs, est in zip(y.ravel(), pred.ravel())
        )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">MTBLS242: preop NMR → 12-month NMR forecast</text>
<text x="{width/2}" y="58" text-anchor="middle" font-family="Arial" font-size="14">Held-out 5-fold predictions</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{margin}" stroke="#555" stroke-dasharray="6 5"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>
{circles}
<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="15">Observed 12-month abundance (log1p)</text>
<text x="23" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="15" transform="rotate(-90 23 {height/2})">Forecast 12-month abundance (log1p)</text>
</svg>'''
        svg_path.write_text(svg, encoding="utf-8")
        return svg_path

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(y.ravel(), pred.ravel(), s=13, alpha=0.35, color="#087f8c")
    low = float(min(y.min(), pred.min()))
    high = float(max(y.max(), pred.max()))
    ax.plot([low, high], [low, high], "--", color="#555555", linewidth=1)
    ax.set_xlabel("Observed 12-month abundance (log1p)")
    ax.set_ylabel("Forecast 12-month abundance (log1p)")
    ax.set_title("MTBLS242: preop NMR → 12-month NMR forecast\nHeld-out 5-fold predictions")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired, x, y, metabolites = load_longitudinal_pairs()
    oof, oof_mean, fold_metrics = cross_validate(x, y)

    # Per-subject error, so the comparison against the mean baseline can be given an
    # interval that accounts for both being measured on the same 71 people.
    subject_mae = np.mean(np.abs(y - oof), axis=1)
    subject_mae_mean_baseline = np.mean(np.abs(y - oof_mean), axis=1)
    subject_mae_persistence = np.mean(np.abs(y - x), axis=1)
    gap_vs_mean = bootstrap_delta_ci(subject_mae, subject_mae_mean_baseline, 5000, SEED)
    gap_vs_persistence = bootstrap_delta_ci(subject_mae, subject_mae_persistence, 5000, SEED)

    per_metabolite = pd.DataFrame(
        {
            "metabolite": metabolites,
            "mae_log1p": np.mean(np.abs(y - oof), axis=0),
            "rmse_log1p": np.sqrt(np.mean((y - oof) ** 2, axis=0)),
            "r2": r2_score(y, oof, axis=0),
        }
    ).sort_values("r2", ascending=False)

    overall = {
        "task": "preop serum NMR to same-subject 12-month serum NMR profile",
        "n_subjects": int(len(paired)),
        "n_metabolites": int(len(metabolites)),
        "evaluation": "nested 5-fold cross-validation; each subject occurs once",
        "mae_log1p": float(np.mean(np.abs(y - oof))),
        "rmse_log1p": float(np.sqrt(np.mean((y - oof) ** 2))),
        "r2_overall": float(r2_score(y, oof)),
        "median_metabolite_r2": float(np.nanmedian(per_metabolite["r2"])),
        "mean_baseline_mae_log1p": float(fold_metrics["mean_baseline_mae_log1p"].mean()),
        "persistence_mae_log1p": float(fold_metrics["persistence_mae_log1p"].mean()),
        "CONTROL_mae_gap_vs_mean_baseline": gap_vs_mean.point,
        "CONTROL_mae_gap_vs_mean_baseline_ci95": [gap_vs_mean.low, gap_vs_mean.high],
        "CONTROL_beats_mean_baseline": bool(gap_vs_mean.high < 0.0),
        "CONTROL_mae_gap_vs_persistence": gap_vs_persistence.point,
        "CONTROL_mae_gap_vs_persistence_ci95": [
            gap_vs_persistence.low,
            gap_vs_persistence.high,
        ],
        "CONTROL_beats_persistence": bool(gap_vs_persistence.high < 0.0),
        "VERDICT": (
            "Forecast beats predicting the training-fold mean."
            if gap_vs_mean.high < 0.0
            else "Forecast is not distinguishable from predicting the training-fold mean."
        ),
    }

    prediction_rows = paired[["subject_id", "Sample Name_preop", "Sample Name_12mo"]].copy()
    for idx, metabolite in enumerate(metabolites):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", metabolite).strip("_")
        prediction_rows[f"observed_{safe}"] = y[:, idx]
        prediction_rows[f"forecast_{safe}"] = oof[:, idx]

    paired_export = paired[["subject_id", "Sample Name_preop", "Sample Name_12mo"]].copy()
    for idx, metabolite in enumerate(metabolites):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", metabolite).strip("_")
        paired_export[f"preop_{safe}"] = x[:, idx]
        paired_export[f"month12_{safe}"] = y[:, idx]

    best_alpha = select_alpha(x, y, SEED + 100)
    final_model = fit_ridge(x, y, best_alpha)
    np.savez_compressed(
        args.output_dir / "forecast_model.npz",
        **final_model,
        alpha=np.array(best_alpha),
        metabolites=np.array(metabolites, dtype=object),
    )

    np.savez_compressed(
        args.output_dir / "subject_errors.npz",
        model=subject_mae,
        mean_baseline=subject_mae_mean_baseline,
        persistence=subject_mae_persistence,
        observed=y,
        predicted=oof,
    )
    paired_export.to_csv(args.output_dir / "paired_longitudinal_dataset.csv", index=False)
    prediction_rows.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    per_metabolite.to_csv(args.output_dir / "metabolite_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_path = save_plot(y, oof, args.output_dir / "forecast_observed_vs_predicted.png")

    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"Selected final alpha: {best_alpha}")
    print(f"Outputs: {args.output_dir.resolve()}")
    print(f"Figure: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
