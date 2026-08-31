# MTBLS242 12-month forecast

## Forecast question

Use the **pre-operative serum ¹H-NMR profile** to forecast the **same subject's serum
NMR profile at 12 months after bariatric surgery**.

This is a genuine longitudinal forecast because sample IDs encode a stable four-digit
subject identifier. The public metadata contains 106 pre-operative samples and 71
twelve-month samples; **71 subjects have both time points**.

This is **not** a forecast of obesity, diabetes, weight loss, or clinical events.
MTBLS242 does not provide a future disease-incidence outcome or BMI/weight-loss target in
the supplied metadata. A clinical risk forecast requires those future outcomes to be
linked at subject level.

## Two experiments

**1. `forecast_mtbls242.py`** — continuous forecast of the 12-month NMR profile.
Multi-output ridge regression, penalty selected inside each training fold, nested 5-fold
CV across 71 subjects. Compared against the training-fold mean and a persistence
forecast, each with a paired bootstrap interval.

→ Not distinguishable from predicting the mean; does beat persistence.
[`RESULTS.md`](RESULTS.md)

**2. `forecast_responder_pr_auc.py`** — binary forecast of a strong 12-month metabolic
response, evaluated by PR-AUC. Sign-constrained logistic regression fitted with L-BFGS-B
under box constraints, 20 repeats × 5-fold stratified CV.

→ No evidence of skill beyond regression to the mean.
[`RESULTS_PR_AUC.md`](RESULTS_PR_AUC.md)

## The outcome, and why it is residualized

A responder is a subject for whom at least **6 of 11** literature-directed obesity
markers improve **more than their own baseline predicts**: change is regressed on
baseline per marker and the residual is used.

Raw change cannot be the outcome here, because the baseline is also a feature. Across
the 11 markers `corr(baseline, change)` has a median of −0.62 — a subject with an
extreme baseline falls back toward the mean whether or not anything biological happened.
Grading a model on raw change grades it on something it was handed.

The residualization is part of the label, so it is fitted **inside each training fold**
and applied to held-out subjects with the training fold's coefficients.

The threshold of 6 was fixed before any model was fitted, as the value reproducing the
originally reported prevalence exactly (31 of 71, 43.7%), so the corrected PR-AUC stays
comparable to the retired one.

> An earlier version of this analysis used raw change with a 9-of-11 threshold and
> reported PR-AUC 0.671. That claim is withdrawn — see [`CORRECTION.md`](CORRECTION.md).

## Monotonic constraints

Monotonic signs come from obesity/insulin-resistance metabolomics literature, carried
over from the `Bridge_ai_summit_nmr_2026` Experiment A constraint set. They are assigned
from published findings, never from this cohort's own correlation signs.

They encode an explicit "opportunity to improve" assumption: a more obesity-associated
baseline profile cannot *reduce* the modelled probability of a strong response. This is a
modelling guardrail, not a causal claim.

Constraints are **not** imposed on the continuous forecast. A metabolite's direction
relative to an obesity label does not establish that a higher baseline concentration must
produce a higher or lower concentration after surgery, and applying those signs to a
different input-output relationship would not be justified.

In the current fit **8 of 11 constraints are binding** — pinned at exactly 0.0 because
this cohort pulls the opposite way. That is only interpretable because the solver
converges; the retired implementation could not distinguish a binding constraint from an
optimizer that had stopped early.

## Excluded features

`isopropanol`, `methanol`, and `Dimethyl sulfone` are excluded by default. They are
exogenous compounds, not obesity biology, and `Bridge_ai_summit_nmr_2026` flags
isopropanol as a **surgical skin-prep contaminant** — confirmed here, pre-operative mean
4.529 vs 12-month 3.205 (log1p). Their pre-operative level partly encodes "this sample
was drawn in an operating theatre."

`--include-artifacts` restores them for comparison against the legacy configuration.

## Run

```bash
python -m pip install -r requirements.txt

python forecast_mtbls242.py
python forecast_responder_pr_auc.py
python figures.py
python make_deck.py --lang en        # English deck
python make_deck.py --lang th        # Thai deck
python -m pytest tests/ -q
```

The analysis scripts need only NumPy, pandas, and SciPy. matplotlib is needed for
`figures.py`, and python-pptx for `make_deck.py`.

Two decks are built from the same layout and the same numbers — only the wording differs:
`mtbls242_forecast_correction.pptx` (English) and `mtbls242_forecast_correction_th.pptx`
(Thai). The Thai deck defaults to **Sukhumvit Set**, which ships with macOS; pass
`--font Tahoma` for a face present on both macOS and Windows. Figures keep English axis
labels in both, since relabelling them per language would fork the figure pipeline for no
scientific gain.

### Flags

| Flag | Effect |
|---|---|
| `--repeats N` | CV repeats for the responder model (default 20) |
| `--null-reps N` | Pairing-destroyed null replicates (default 500) |
| `--include-artifacts` | Keep the three exogenous compounds as features |
| `--legacy-outcome` | Use the superseded baseline-dependent outcome |
| `--output-dir PATH` | Where to write results |

## Files

| File | Purpose |
|---|---|
| [`controls.py`](controls.py) | Negative controls and bootstrap intervals, shared |
| [`monotone_logistic.py`](monotone_logistic.py) | Sign-constrained logistic regression (L-BFGS-B) |
| [`forecast_mtbls242.py`](forecast_mtbls242.py) | Continuous profile forecast |
| [`forecast_responder_pr_auc.py`](forecast_responder_pr_auc.py) | Responder forecast |
| [`figures.py`](figures.py) | Regenerates `Graph/` at 300 DPI |
| [`make_deck.py`](make_deck.py) | Builds the presentation decks from the summary JSONs |
| [`deck_content.py`](deck_content.py) | Slide copy, English + Thai (layout stays shared) |
| [`METHODS.md`](METHODS.md) | Why each control exists |
| [`CORRECTION.md`](CORRECTION.md) | The retired PR-AUC 0.671 claim |
| `tests/` | 42 tests, each against a case with a known answer |

## Outputs

`outputs/` (continuous):
`summary.json`, `fold_metrics.csv`, `metabolite_metrics.csv`, `heldout_predictions.csv`,
`paired_longitudinal_dataset.csv`, `subject_errors.npz`, `forecast_model.npz`,
`forecast_observed_vs_predicted.png`.

`outputs_pr_auc/` (responder):
`summary_pr_auc.json`, `fold_pr_auc.csv`, `heldout_responder_predictions.csv`,
`monotone_coefficients.csv`, `monotone_responder_model.npz`, `null_distributions.npz`.

Keys prefixed `CONTROL_` in the summary JSONs carry the negative controls; `VERDICT`
states the conclusion those controls support.

## Limitations

- The cohort is bariatric surgery patients only, so it estimates nothing about outcomes
  without surgery or about the general population.
- With 71 complete subjects, estimates carry substantial uncertainty and need external
  validation. A small true effect would not be detectable here.
- Forecasting a metabolic profile is not equivalent to forecasting a clinical outcome.
- Treatment type and clinical covariates are absent from the public sample sheet.
- The three mid time points (3, 6, 9 months) are unused; only preop and 12 months are
  paired.
- **`(R)-3-Hydroxybutyric acid` is 45% zeros** (32/71 pre-operative, 27/71 twelve-month).
  These are almost certainly below-detection readings rather than true absence, and
  `log1p` maps them to exactly 0, so the metabolite is treated as if it were measured at
  zero. It is left unconstrained and is not among the outcome markers, so it does not
  affect the responder verdict, but its per-metabolite continuous result should not be
  read as a measurement. Six other metabolites carry one or two zeros each.

## Data provenance

The included ISA-Tab sample sheet and metabolite abundance table were downloaded from the
official EMBL-EBI MetaboLights public MTBLS242 directory.
