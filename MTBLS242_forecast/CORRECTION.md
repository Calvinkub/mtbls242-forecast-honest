# Correction — the retired PR-AUC 0.671 responder claim

**Date:** 2026-08-31
**Status:** the superseded claim is withdrawn. The code that produced it is still
runnable via `--legacy-outcome --include-artifacts`, so the correction is reproducible.

## What was claimed

> Held-out PR-AUC **0.671** against a **0.437** prevalence baseline — a **1.54×** lift,
> read as evidence that pre-operative serum NMR carries signal about who will be a
> strong 12-month metabolic responder.

## Why it was wrong

### 1. The outcome was a function of the features

`make_outcome` labelled a subject a responder when at least 9 of 11 metabolites moved in
the literature-expected direction, measured as `y_12mo[j] - x_preop[j]`. Those same 11
metabolites were model inputs.

Across the 11 markers, `corr(baseline, change)` has median **−0.62** and reaches
**−0.94**. That is arithmetic, not biology: a subject with an extreme baseline falls
back toward the mean whether or not anything happened to them. The model was rewarded
for reading the baseline, which it was given.

### 2. The score sat below its own no-signal null

Shuffling which 12-month profile belongs to which subject destroys any real
subject-level signal while preserving both marginal distributions and every mechanical
baseline-to-change relationship. Rebuilding the outcome under that shuffle and scoring
it with a no-learning scorer gives a **median lift of 2.04×** and a 95th percentile of
**2.51×**.

The observed lift was **1.57×**. The real data scored *below the middle of a
distribution in which no forecastable signal can exist.*

### 3. A standard permutation test could not have caught this

On the legacy outcome the label-permutation test returns **p = 0.002**. It looks
strongly significant. It is not wrong — it correctly reports that the baseline ranks
these labels better than chance — but it holds the outcome fixed and therefore cannot
see that the outcome was manufactured from the baseline in the first place.

**This is the methodological point worth keeping:** when the outcome is derived from a
change score and the baseline is a feature, permutation testing is not a sufficient
control. The pairing must be destroyed and the outcome rebuilt.

### 4. The model had not converged

`fit_monotone_logistic` ran fixed-step projected gradient descent for a capped 6000
iterations against a penalty that was not scaled to sample size. It stopped with a
maximum standardized coefficient of **0.018** and selected `l2=10` in 3 of 5 folds — a
near-constant model. Its ranking coincided exactly with an unweighted sum of the
markers, which is why a no-learning scorer reproduced its PR-AUC to three decimals
(both **0.653**).

Every "monotonic constraint" it reported was therefore vacuous: the coefficients had
barely left zero, so nothing was being constrained.

## What replaced it

| | Retired | Current |
|---|---|---|
| Outcome | ≥9/11 markers move in the literature direction | ≥6/11 markers improve **more than their own baseline predicts** |
| Residualization | none | change regressed on baseline, **fitted inside each training fold** |
| Prevalence | 0.437 (31/71) | 0.437 (31/71) — held constant by design |
| Solver | projected gradient descent, unconverged | L-BFGS-B under box constraints, 100% convergence |
| Features | 21, including 3 exogenous artifacts | 18, artifacts quarantined |
| Evaluation | one 5-fold draw, seed 42 | 20 repeats × 5-fold |
| Controls | none | no-learning scorer, pairing-destroyed null, permutation test |

The threshold moved from 9 to 6 because residualization halves the per-marker
improvement rate by construction. 6 was chosen before any model was fitted, as the value
that reproduces the original prevalence exactly, so the two numbers stay comparable.

## The corrected result

| Metric | Value |
|---|---|
| Model PR-AUC (20 × 5-fold) | **0.483 ± 0.027** |
| No-skill baseline (prevalence) | 0.437 |
| Observed lift | 1.11× |
| **No-learning scorer** | **0.522** — *higher than the model* |
| Pairing-destroyed null, median lift | 1.04× |
| Pairing-destroyed null, 95th percentile | 1.24× |
| Permutation p-value | 0.894 |

**Verdict: no evidence of forecasting skill beyond regression to the mean.**

The observed 1.11× lift sits inside the null distribution. The trained model is *worse*
than adding up signed baseline z-scores, with a 95% CI of [−0.058, −0.023] on that gap
— 18 features fitted on 57 training subjects overfits relative to a fixed-weight sum.

## What this does not say

It does not say pre-operative NMR cannot forecast metabolic response. It says **this
cohort, at n=71, with this outcome, does not demonstrate that it does.** A cohort with
linked clinical outcomes — weight loss, remission, treatment type — and enough subjects
to estimate a small effect could still find one. MTBLS242's public metadata has none of
those.

## Reproducing both numbers

```bash
# The corrected result
python forecast_responder_pr_auc.py

# The retired claim, with the controls that retired it
python forecast_responder_pr_auc.py --legacy-outcome --include-artifacts \
    --output-dir outputs_legacy
```

The legacy path reports PR-AUC 0.683 ± 0.043 and lift 1.57× — reproducing the retired
claim closely; it differs slightly from the original 0.671 only because the solver now
converges and the CV is repeated. Its controls report a null median of 2.04×, which is
what retires it.
