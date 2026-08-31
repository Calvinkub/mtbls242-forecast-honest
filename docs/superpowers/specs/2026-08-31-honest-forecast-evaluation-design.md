# Honest evaluation for the MTBLS242 12-month forecast

Date: 2026-08-31
Repo: `Chaiporn2005/Forecast`, branch `honest-evaluation`
Related work: `Bridge_ai_summit_nmr_2026` (PhenoInsure / BDI Hackathon 2026)

## Problem

`forecast_responder_pr_auc.py` reports a held-out **PR-AUC of 0.671 against a 0.437
prevalence baseline** and describes this as a 1.54x lift, i.e. as evidence that the
pre-operative serum NMR profile carries signal about who will be a strong 12-month
metabolic responder.

A read-only probe against the repo's own data shows that claim does not hold.

| Check | Result |
|---|---|
| Trivial scorer: sum of signed baseline z-scores, no learning at all | AP **0.653** |
| The repo's trained monotonic logistic model, mean over folds | AP **0.653** |
| `corr(baseline, delta)` across the 11 outcome markers | median **-0.62**, min **-0.94** |
| Pairing-destroyed null (shuffle which 12-month profile belongs to which subject), trivial scorer | AP **0.741** |

Three separate failures:

1. **The outcome is a function of the features.** `make_outcome` labels a subject a
   responder when at least 9 of 11 markers satisfy `y_12mo[j] - x_preop[j]` in the
   healthy direction. Those same 11 markers are model inputs. A subject with a high
   baseline is mechanically more likely to fall. `corr(baseline, delta) = -0.62` is
   arithmetic, not biology.
2. **The headline number is below its own no-signal null.** When subject pairing is
   destroyed, so that no forecastable subject-level signal can exist by construction,
   the trivial scorer still reaches 0.741 - higher than the reported 0.671.
3. **The model is not fitted.** Maximum standardized coefficient is 0.018 and `l2=10`
   is selected in 3 of 5 folds, because `fit_monotone_logistic` runs fixed-step
   projected gradient descent for 6000 iterations against a penalty that is not scaled
   to sample size. The model is near-constant; its ranking is a near-uniform positive
   sum of BCAAs, which is why it exactly matches the trivial scorer.

Two findings from the Bridge work were not carried across and change the analysis:

- **Isopropanol is a surgical skin-prep contaminant** (Bridge README, Experiment A
  caveats). Confirmed here: preop mean 4.529 vs 12-month mean 3.205 (log1p). It is an
  unconstrained feature with a positive fitted coefficient, so the responder model is
  partly reading whether a sample was drawn in an operating theatre. The same applies
  to methanol and dimethyl sulfone.
- **Provenance stated on the figure.** Every Bridge PR curve writes its data source and
  evaluation method onto the plot. No figure in this repo does.

The continuous model in `forecast_mtbls242.py` is *not* subject to failures 1-3, and
`RESULTS.md` already correctly disowns its pooled R^2 of 0.931. Its remaining gap is
that the reported 2.4% MAE improvement over the training-fold mean carries no
uncertainty estimate, so it cannot be distinguished from noise.

## Goals

- Replace the responder outcome with one that is not mechanically determined by the
  features.
- Ship the negative controls as first-class, always-computed outputs rather than as a
  one-off probe, so no future revision can quietly regress to the current claim.
- Fit the constrained model with a solver that converges, so the reported coefficients
  mean something.
- Estimate uncertainty honestly on both models.
- Carry across the two Bridge findings above.
- State plainly, in the repo, what was previously reported and why it was wrong.

## Non-goals

- Gradient-boosted models or SHAP. A linear model's coefficients are already the
  explanation, and a stronger learner would only fit the same tautology more tightly.
- A dedicated single-target lipoprotein forecast. Considered and deferred; it should be
  built on top of the controls this spec delivers, not before them.
- Any claim about weight loss, obesity remission, or clinical events. MTBLS242's public
  metadata contains no such outcome, as the existing README already states.

## Design

### 1. `controls.py` (new) - shared negative controls

One module, no model-specific knowledge, consumed by both experiment scripts.

- `trivial_signed_baseline_score(x, marker_idx, signs)` - the no-learning scorer. Any
  trained model that does not beat this has demonstrated nothing.
- `pairing_destroyed_null(x, y, outcome_fn, scorer, n_reps)` - permutes the subject to
  12-month assignment, rebuilds the outcome, rescores. Returns the null AP
  distribution **and each replicate's prevalence**, because destroying the pairing also
  shifts prevalence (0.372 vs 0.437 observed); comparisons are reported as lift over
  each replicate's own prevalence so they are commensurable.
- `label_permutation_null(y_true, score, n_reps)` - standard significance test; permutes
  labels against fixed scores. Yields an empirical p-value.
- `bootstrap_delta_ci(metric_a, metric_b, n_reps)` - paired bootstrap over subjects for
  a difference in metrics, used for the continuous model's MAE gap.
- `repeated_stratified_cv(y, n_splits, n_repeats, seed)` - fold generator.

Every function is deterministic given a seed.

### 2. Outcome redefinition

The primary outcome becomes **residualized improvement**. For each of the 11 markers,
regress `delta_j = y_j - x_j` on `x_j` and take the residual `r_j`. A marker counts as
improved when `r_j` moves in the healthy direction - that is, when it improved *more
than its own baseline predicts*. The residual is orthogonal to `x_j` by construction,
so the mechanical edge is removed while all 21 features are retained.

The per-marker regression is **fitted inside each training fold and applied to the held
-out fold**, never on the full cohort, since it is part of the label definition.

The threshold needs an explicit decision, because residualization changes the base
rate: a residual is negative for roughly half of subjects by construction, so requiring
9 of 11 would make responders rare and the positive class unusably small at n=71.
**The threshold is therefore set to the count that reproduces the originally observed
prevalence as closely as possible (31 of 71, 43.7%), so the corrected number is
comparable to the superseded one rather than being confounded by a prevalence shift.**
The chosen count is computed once, recorded in the summary JSON, and fixed before any
model is fitted.

A **sensitivity analysis** runs a disjoint split instead: the 11 literature markers
define the outcome and only the remaining metabolites are features. This is
conceptually cleaner but leaves a weak feature set, so it is reported alongside rather
than as the headline.

### 3. Solver replacement

`fit_monotone_logistic` is replaced with `scipy.optimize.minimize(method="L-BFGS-B")`
on the L2-penalized logistic loss, with per-coefficient bounds: `(0, inf)` for a `+1`
constraint, `(-inf, 0)` for `-1`, unbounded for `0`. Bounds give exact sign constraints
and the optimizer actually converges. Convergence status is asserted, and the L2 grid is
rescaled so the penalty is comparable per sample.

scipy is already available in the bare environment (1.13.1) and in `superaiss6`
(1.17.1), so this adds a dependency the repo can satisfy either way. `requirements.txt`
gains `scipy>=1.11`.

### 4. Artifact quarantine

`isopropanol`, `methanol`, and `Dimethyl sulfone` are excluded from the feature set by
default, with a `--include-artifacts` flag to reproduce the old behaviour. The exclusion
and its Bridge-sourced reason are recorded in the summary JSON, so the figure and the
docs can state it.

### 5. Uncertainty

- Responder model: **20 repeats x 5-fold** CV. Headline is mean +/- std **across
  repeats**, not across folds - the current +/- 0.184 is the spread of 14-subject test
  sets and overstates precision of the mean while understating instability.
- Continuous model: paired bootstrap CI on the MAE gap versus the training-fold mean
  baseline, plus a label-permutation null. If the CI spans zero, the docs say so.

### 6. Figures

`figures.py` (new) regenerates everything in `Graph/` reproducibly. Bridge convention:
data source, cohort size, and evaluation method written onto the plot itself. Three
figures - PR curve with the trivial scorer and both nulls overlaid; repeated-CV
distribution; continuous-model MAE with bootstrap CI. matplotlib at 300 DPI under
`superaiss6`; the existing dependency-free SVG fallback is retained for the bare
environment.

### 7. Documentation

- Root `README.md` is currently **empty**. It gets the orientation: what the repo
  forecasts, what it does not, headline numbers, how to run.
- `RESULTS_PR_AUC.md` is rewritten around the corrected numbers. Currently Thai while
  `RESULTS.md` is English; both become English with a Thai summary section, matching the
  Bridge repo's bilingual habit.
- `METHODS.md` (new) documents the controls and why each exists.
- `CORRECTION.md` (new) records the superseded 0.671 claim, the evidence that retired
  it, and the date. The prior numbers are not deleted from history.

## Expected outcome

The headline PR-AUC will fall, plausibly to near the prevalence baseline. **That is the
correct result, not a failure of the work.** The deliverable is a forecast pipeline
whose reported number survives scrutiny, plus reusable controls. If residualized
response turns out to be genuinely unpredictable from baseline NMR at n=71, the repo
will say exactly that.

## Testing

Each control gets a test against a case with a known answer:

- Trivial scorer on separable synthetic data returns AP 1.0; on random data, prevalence.
- Pairing-destroyed null on data with genuine injected signal returns a null
  distribution the real score clears; on pure-RTM synthetic data it does not.
- Residualization: `corr(r_j, x_j) ~= 0` on held-out folds, asserted.
- Sign constraints: solver output has no coefficient violating its bound.
- Fold hygiene: no subject appears in both train and test in any repeat.
- Regression guard: reproduce the old pipeline under `--include-artifacts` with the
  original outcome and confirm it still yields ~0.67, so the correction is demonstrably
  about methodology and not about an unrelated code change.

## Delivery

The upstream repo `Chaiporn2005/Forecast` belongs to another account. Work is developed
on the local `honest-evaluation` branch at `/Users/admin/Project/Forecast` and delivered
by **copying into the user's own GitHub account (`Calvinkub`)** as a separate repository,
with `Chaiporn2005/Forecast` retained in the git history and credited in the README as
the origin of the pipeline. Nothing is pushed to the upstream repo.
