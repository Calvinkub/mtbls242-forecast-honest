# Methods — and why each control exists

Every control in [`controls.py`](controls.py) exists because a specific way of being
wrong was found in this repo, not because it is conventional. This document records
which failure each one catches.

## The three controls

### 1. The no-learning scorer

```python
trivial_signed_baseline_score(x, marker_idx, signs)   # sum of signed, z-scored baseline
```

**Catches:** a model that has not converged, or that has learned nothing its inputs did
not already say.

The retired responder model scored PR-AUC 0.653. So did this function, which fits
nothing. They matched to three decimals because the solver had stopped with a maximum
coefficient of 0.018, leaving a near-uniform positive sum of the markers — exactly what
this scorer computes.

Any trained model is reported next to this number. A model that does not beat it has
demonstrated nothing, however far its raw metric sits above prevalence.

### 2. The pairing-destroyed null

```python
pairing_destroyed_null(x, y, outcome_fn, scorer, n_reps, seed)
```

**Catches:** an outcome that is manufactured from the features.

Shuffle which 12-month profile belongs to which subject, then **rebuild the outcome from
scratch**. This destroys every real subject-level relationship while preserving both
marginal distributions and every mechanical baseline-to-change relationship. Whatever
score survives is the part that was never about forecasting.

Destroying the pairing also shifts the outcome's base rate, so results are reported as
**lift over each replicate's own prevalence**. Raw AP across replicates with different
prevalences is not comparable, and comparing it that way would reintroduce the error
this control exists to catch.

On the legacy outcome the null reaches a median lift of 2.04× against an observed 1.57×.

### 3. The label-permutation test

```python
label_permutation_null(y_true, score, n_reps, seed)
```

**Catches:** a ranking that is no better than chance against a *fixed* outcome.

Necessary but **not sufficient**, and this repo is the counterexample. On the legacy
outcome it returns p = 0.002 — apparently strong evidence. It is not malfunctioning: the
baseline genuinely does rank those labels better than chance. But it holds the outcome
fixed, so it cannot see that the outcome was built out of the baseline.

**A permutation test cannot validate a change-score outcome when the baseline is a
feature.** Only control 2 can.

## Residualized outcome

A subject with an extreme baseline moves back toward the mean whether or not anything
biological happened. Since the baseline is also a feature, raw change is not a
forecastable outcome — the model is graded on something it was handed.

For each marker, `change_j` is regressed on `baseline_j` and the residual is kept. A
marker counts as improved when it improved **more than its own baseline predicts**. The
residual is orthogonal to the baseline by construction (verified to machine precision in
`tests/test_forecast_pipeline.py`), which removes the mechanical edge while keeping all
features available.

The fit is part of the **label**, so it is fitted inside each training fold and applied
to held-out subjects using the training fold's coefficients. Fitting it on the full
cohort would leak the held-out subjects' futures into their own labels.

### The threshold

On the residualized scale a marker improves for roughly half of subjects by
construction, so the legacy 9-of-11 rule would leave too few positives at n=71.

The threshold is **6 of 11**, chosen before any model was fitted as the value
reproducing the originally reported prevalence exactly — 31 of 71, 0.437. Holding the
base rate constant keeps the corrected PR-AUC directly comparable to the retired one
instead of confounding the comparison with a shift in prevalence. It is pinned by a test
so it cannot drift.

## Sign-constrained logistic regression

Monotonic directions come from obesity and insulin-resistance metabolomics literature
(Newgard 2009; Wang 2011 *Nat Med*; Würtz et al.), carried over from
`Bridge_ai_summit_nmr_2026` Experiment A. They are assigned from published findings, never
from this cohort's own correlation signs — mining the direction from the data being
tested is how a constraint stops being a prior and starts being a leak.

Fitting is L-BFGS-B under box constraints: `[0, ∞)` for a `+1` direction, `(-∞, 0]` for
`-1`, unbounded otherwise. Box constraints give exact signs *and* a convergence
guarantee, which matters for interpretation: a coefficient that lands on exactly 0.0
means the constraint is **binding** — the data pulled the other way — rather than that
the optimizer gave up. The retired implementation could not distinguish those two cases.

**8 of 11 constraints are binding** in the current fit, including `lipoproteins`. The
literature directions for obesity *status* largely do not transfer to forecasting
12-month *change* in this cohort. That is a result, and it is consistent with the
verdict: there is little for the constrained model to fit.

## Artifact quarantine

`isopropanol`, `methanol`, and `Dimethyl sulfone` are exogenous compounds, not obesity
biology. `Bridge_ai_summit_nmr_2026` flags isopropanol as a **surgical skin-prep
contaminant**; confirmed here, pre-operative mean 4.529 vs 12-month 3.205 (log1p), and
pinned by a test.

Their pre-operative level partly encodes "this sample was drawn in an operating
theatre." A forecast that leans on that is reading the procedure, not the patient. They
are excluded by default; `--include-artifacts` restores them for the legacy comparison.

Note they were *unconstrained* in the retired model, and carried positive fitted
coefficients. Leaving a confound unconstrained does not stop a model from using it.

## Evaluation

**20 repeats × 5-fold stratified CV.** A single 5-fold draw on 71 subjects puts 14
people in each test set; the retired report's ±0.184 was the spread across those folds,
which measures test-set size more than model instability. Repeating the whole split
under different seeds gives a mean whose variation reflects resampling instability — the
number worth reporting. It is ±0.027 here.

**PR-AUC (average precision)** is the headline rather than ROC-AUC, following the Bridge
convention: it is the more honest read under class imbalance, and it degrades visibly
when a model ranks the minority class poorly.

**Uncertainty** on the continuous model is a paired bootstrap over subjects, resampling
both error series together because they are measured on the same 71 people. Its 2.4% MAE
edge over the training-fold mean has a 95% CI of [−0.0158, +0.0001] — it spans zero, so
the forecast is not distinguishable from predicting the mean.

## What is deliberately not here

- **Gradient boosting and SHAP.** A stronger learner would fit the same tautology more
  tightly. A linear model's coefficients already are the explanation.
- **A single-target lipoprotein forecast.** Worth building, but on top of these
  controls rather than before them. `lipoproteins` has a per-metabolite R² of 0.057 in
  the continuous model and a binding constraint in the responder model, so it should be
  expected to show little.
- **Any claim about weight loss, remission, or clinical events.** MTBLS242's public
  metadata contains no such outcome.
