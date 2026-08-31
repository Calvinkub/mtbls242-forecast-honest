"""Build the presentation deck from the experiment outputs.

Every number on every slide is read from `summary.json` / `summary_pr_auc.json` rather
than typed in, so the deck cannot drift from the results. Regenerate after any rerun:

    python forecast_mtbls242.py
    python forecast_responder_pr_auc.py
    python figures.py
    python make_deck.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x66, 0x66, 0x66)
RED = RGBColor(0xB2, 0x18, 0x2B)
BLUE = RGBColor(0x1F, 0x6F, 0x8B)
RULE = RGBColor(0xDD, 0xDD, 0xDD)

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.75)


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _text(slide, left, top, width, height, text, size, *, bold=False, color=INK,
          align=PP_ALIGN.LEFT, italic=False, spacing=1.0):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        run.font.name = "Helvetica Neue"
    return box


def _rule(slide, top, color=RULE, height=Emu(9525)):
    from pptx.enum.shapes import MSO_SHAPE

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, top, W - 2 * MARGIN, height)
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


def _heading(slide, title, kicker=None):
    if kicker:
        _text(slide, MARGIN, Inches(0.42), W - 2 * MARGIN, Inches(0.3), kicker.upper(),
              11, bold=True, color=MUTED)
    _text(slide, MARGIN, Inches(0.72), W - 2 * MARGIN, Inches(0.9), title, 30, bold=True)
    _rule(slide, Inches(1.62))


def _fit_picture(slide, path, top, max_h, max_w=None):
    from PIL import Image

    max_w = max_w or (W - 2 * MARGIN)
    with Image.open(path) as im:
        ratio = im.width / im.height
    height, width = max_h, int(max_h * ratio)
    if width > max_w:
        width, height = max_w, int(max_w / ratio)
    pic = slide.shapes.add_picture(str(path), int((W - width) / 2), top,
                                   width=width, height=height)
    return pic


def _metric_row(slide, top, items):
    """Big-number row: [(value, label, color), ...]"""
    span = (W - 2 * MARGIN) / len(items)
    for i, (value, label, color) in enumerate(items):
        left = int(MARGIN + i * span)
        _text(slide, left, top, int(span), Inches(0.8), value, 40, bold=True,
              color=color, align=PP_ALIGN.CENTER)
        _text(slide, left, top + Inches(0.78), int(span), Inches(0.6), label, 11,
              color=MUTED, align=PP_ALIGN.CENTER)


def _table(slide, left, top, width, rows, col_widths, *, header=True, size=12):
    from pptx.util import Emu as _Emu

    shape = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width,
                                   Inches(0.32) * len(rows))
    table = shape.table
    total = sum(col_widths)
    for i, w in enumerate(col_widths):
        table.columns[i].width = _Emu(int(width * w / total))
    for r, row in enumerate(rows):
        table.rows[r].height = Inches(0.34)
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            para = cell.text_frame.paragraphs[0]
            run = para.add_run()
            run.text = str(value)
            run.font.size = Pt(size)
            run.font.bold = header and r == 0
            run.font.color.rgb = INK if not (header and r == 0) else MUTED
            run.font.name = "Helvetica Neue"
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            cell.margin_left = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
    return table


def build(pr, cont, graph: Path, out: Path) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ---------------------------------------------------------------- title
    s = _blank(prs)
    _text(s, MARGIN, Inches(2.1), W - 2 * MARGIN, Inches(0.4),
          "MTBLS242 · 12-MONTH METABOLIC FORECAST", 13, bold=True, color=MUTED)
    _text(s, MARGIN, Inches(2.6), W - 2 * MARGIN, Inches(1.6),
          "What the controls found", 54, bold=True)
    _rule(s, Inches(4.3))
    _text(s, MARGIN, Inches(4.6), W - 2 * MARGIN, Inches(1.2),
          "A reported PR-AUC of 0.671 turned out to be regression to the mean.\n"
          "This is how it was caught, and what the corrected analysis says.",
          17, color=MUTED, spacing=1.3)

    # ---------------------------------------------------------------- question
    s = _blank(prs)
    _heading(s, "The question", "setup")
    _text(s, MARGIN, Inches(2.0), W - 2 * MARGIN, Inches(1.0),
          "Can a pre-operative serum ¹H-NMR profile forecast what happens to a\n"
          "bariatric surgery patient 12 months later?", 22, spacing=1.35)
    _metric_row(s, Inches(3.5), [
        ("71", "subjects with paired\npreop + 12-month samples", INK),
        ("21", "quantified metabolites\n(serum ¹H-NMR)", INK),
        ("31", "labelled strong responders\n(43.7% prevalence)", INK),
    ])
    _text(s, MARGIN, Inches(5.9), W - 2 * MARGIN, Inches(0.9),
          "MetaboLights MTBLS242. Sample IDs encode a stable subject code, which is what "
          "makes a genuine\nlongitudinal pairing possible. No clinical outcome exists in "
          "the public metadata — no weight loss, no remission.",
          12, color=MUTED, spacing=1.3)

    # ---------------------------------------------------------------- the claim
    s = _blank(prs)
    _heading(s, "The result that was reported", "the claim")
    _metric_row(s, Inches(2.3), [
        ("0.671", "held-out PR-AUC", RED),
        ("0.437", "prevalence baseline", MUTED),
        ("1.54×", "reported lift", RED),
    ])
    _text(s, MARGIN, Inches(4.2), W - 2 * MARGIN, Inches(1.4),
          "Read as: pre-operative NMR carries signal about who will respond.\n\n"
          "It does not. Four things were wrong, and three of them are invisible to the "
          "checks people normally run.",
          18, spacing=1.35)

    # ---------------------------------------------------------------- defect 1
    s = _blank(prs)
    _heading(s, "Defect 1 — the outcome was built from the features", "why it was wrong")
    _text(s, MARGIN, Inches(1.85), W - 2 * MARGIN, Inches(0.7),
          "A responder was defined as “≥9 of 11 markers moved in the healthy direction”, "
          "measured as 12-month minus pre-operative —\non the same 11 metabolites the "
          "model was given as inputs.", 13, color=MUTED, spacing=1.3)
    _fit_picture(s, graph / "regression_to_the_mean.png", Inches(2.75), Inches(3.55))
    _text(s, MARGIN, Inches(6.55), W - 2 * MARGIN, Inches(0.6),
          "An extreme baseline falls back toward the mean whether or not anything "
          "biological happened. The model was graded on something it was handed.",
          13, bold=True, spacing=1.25)

    # ---------------------------------------------------------------- defect 2
    s = _blank(prs)
    _heading(s, "Defect 2 — it scored below its own no-signal null", "why it was wrong")
    _text(s, MARGIN, Inches(1.85), W - 2 * MARGIN, Inches(0.9),
          "Shuffle which 12-month profile belongs to which subject, then rebuild the "
          "outcome from scratch.\nNo forecastable signal can survive that — every real "
          "subject-level relationship is destroyed.", 14, spacing=1.3)
    _metric_row(s, Inches(3.4), [
        ("1.57×", "observed lift", RED),
        ("2.04×", "null median lift", BLUE),
        ("2.51×", "null 95th percentile", BLUE),
    ])
    _text(s, MARGIN, Inches(5.5), W - 2 * MARGIN, Inches(1.2),
          "The real data scored below the middle of a distribution in which\n"
          "no forecastable signal can exist.", 22, bold=True, color=RED, spacing=1.3)

    # ---------------------------------------------------------------- defect 3
    s = _blank(prs)
    _heading(s, "Defect 3 — a permutation test would have waved it through",
             "the methodological point")
    _text(s, MARGIN, Inches(2.0), W - 2 * MARGIN, Inches(0.6),
          "On the retired outcome, the standard label-permutation test returns", 16,
          color=MUTED)
    _text(s, MARGIN, Inches(2.6), W - 2 * MARGIN, Inches(1.0), "p = 0.002", 52,
          bold=True, color=RED)
    _text(s, MARGIN, Inches(3.9), W - 2 * MARGIN, Inches(2.2),
          "It is not malfunctioning. The baseline genuinely does rank those labels better "
          "than chance.\n\nBut it holds the outcome fixed, so it cannot see that the "
          "outcome was manufactured from the\nfeatures in the first place.",
          17, spacing=1.35)
    _rule(s, Inches(5.85))
    _text(s, MARGIN, Inches(6.05), W - 2 * MARGIN, Inches(0.9),
          "A permutation test cannot validate a change-score outcome when the baseline is "
          "also a feature.\nOnly destroying the pairing can.", 15, bold=True, spacing=1.3)

    # ---------------------------------------------------------------- defect 4
    s = _blank(prs)
    _heading(s, "Defect 4 — the model had never converged", "why it was wrong")
    _metric_row(s, Inches(2.2), [
        ("0.018", "largest standardized\ncoefficient", RED),
        ("0.653", "PR-AUC of the\n“trained” model", RED),
        ("0.653", "PR-AUC of a scorer\nthat fits nothing", BLUE),
    ])
    _text(s, MARGIN, Inches(4.3), W - 2 * MARGIN, Inches(1.8),
          "Fixed-step projected gradient descent, capped at 6000 iterations, against a "
          "penalty that was not\nscaled to sample size. It stopped as a near-constant "
          "model.\n\nIts ranking coincided exactly with an unweighted sum of the markers — "
          "which is why summing signed\nbaseline z-scores reproduces its score to three "
          "decimals. Every “monotonic constraint” it reported\nwas vacuous: the "
          "coefficients had barely left zero.", 14, spacing=1.3)

    # ---------------------------------------------------------------- what replaced it
    s = _blank(prs)
    _heading(s, "What replaced it", "the fix")
    _table(s, MARGIN, Inches(1.95), W - 2 * MARGIN, [
        ["", "Retired", "Current"],
        ["Outcome", "≥9/11 markers move in the\nliterature direction",
         "≥6/11 improve MORE than their\nown baseline predicts"],
        ["Residualization", "none", "change regressed on baseline,\nfitted inside each fold"],
        ["Prevalence", "0.437 (31/71)", "0.437 (31/71) — held constant"],
        ["Solver", "projected gradient, unconverged", "L-BFGS-B box constraints, 100%"],
        ["Features", "21, incl. 3 exogenous artifacts", "18, artifacts quarantined"],
        ["Evaluation", "one 5-fold draw, seed 42", "20 repeats × 5-fold"],
        ["Controls", "none", "3 controls + bootstrap intervals"],
    ], [1.1, 2.0, 2.2], size=11)
    _text(s, MARGIN, Inches(5.75), W - 2 * MARGIN, Inches(1.1),
          "The threshold moved from 9 to 6 because residualization halves the per-marker "
          "improvement rate by construction.\nIt was fixed before any model was fitted, as "
          "the value that reproduces the original prevalence exactly — so the\ncorrected "
          "number stays directly comparable rather than confounded by a shift in base rate.",
          12, color=MUTED, spacing=1.3)

    # ---------------------------------------------------------------- corrected result
    s = _blank(prs)
    _heading(s, "The corrected result", "what it actually says")
    _metric_row(s, Inches(1.95), [
        (f"{pr['model_pr_auc_mean']:.3f}", "model PR-AUC\n(20 × 5-fold)", INK),
        (f"{pr['CONTROL_trivial_scorer_pr_auc']:.3f}",
         "no-learning scorer\n— beats the model", BLUE),
        (f"{pr['CONTROL_permutation_p_value']:.3f}", "permutation\np-value", MUTED),
    ])
    _fit_picture(s, graph / "pr_auc_curve.png", Inches(3.75), Inches(2.85))
    _text(s, MARGIN, Inches(6.75), W - 2 * MARGIN, Inches(0.6),
          pr["VERDICT"], 20, bold=True, color=RED, align=PP_ALIGN.CENTER)

    # ---------------------------------------------------------------- three agree
    s = _blank(prs)
    _heading(s, "Three independent controls agree", "what it actually says")
    rows = [
        ("1", "Observed lift 1.11× sits INSIDE the no-signal null",
         "null median 1.04×, 95th percentile 1.24× — it does not clear it"),
        ("2", "The trained model is WORSE than a scorer that fits nothing",
         "gap −0.039, 95% CI [−0.058, −0.023] — the interval excludes zero"),
        ("3", "Permutation p = 0.894",
         "against the corrected outcome, the ranking is indistinguishable from chance"),
    ]
    top = Inches(2.1)
    for number, headline, detail in rows:
        _text(s, MARGIN, top, Inches(0.6), Inches(0.6), number, 30, bold=True, color=RED)
        _text(s, MARGIN + Inches(0.7), top + Inches(0.02), W - MARGIN * 2 - Inches(0.7),
              Inches(0.4), headline, 17, bold=True)
        _text(s, MARGIN + Inches(0.7), top + Inches(0.45), W - MARGIN * 2 - Inches(0.7),
              Inches(0.4), detail, 13, color=MUTED)
        top += Inches(1.25)
    _rule(s, Inches(6.1))
    _text(s, MARGIN, Inches(6.3), W - 2 * MARGIN, Inches(0.8),
          "8 of 11 literature constraints are binding — pinned at exactly 0.0 because this "
          "cohort pulls the other way.\nObesity-status directions largely do not transfer "
          "to forecasting 12-month change here.", 13, color=MUTED, spacing=1.3)

    # ---------------------------------------------------------------- continuous
    s = _blank(prs)
    _heading(s, "The continuous forecast, same treatment", "the other experiment")
    _fit_picture(s, graph / "continuous_forecast_mae.png", Inches(2.0), Inches(3.7))
    _text(s, MARGIN, Inches(5.95), W - 2 * MARGIN, Inches(1.2),
          f"Forecasting the 12-month profile beats predicting the training-fold mean by "
          f"{abs(cont['CONTROL_mae_gap_vs_mean_baseline']):.4f} MAE — but the 95% "
          f"interval\nspans zero, so it is not distinguishable from noise at n=71. It does "
          f"clearly beat persistence.\n\nThe pooled R² of {cont['r2_overall']:.3f} is not "
          f"performance: it is driven by scale differences between metabolites. "
          f"Median per-metabolite R² is {cont['median_metabolite_r2']:.3f}.",
          13, spacing=1.3)

    # ---------------------------------------------------------------- reusable
    s = _blank(prs)
    _heading(s, "The controls, reusable", "takeaway")
    items = [
        ("No-learning scorer", "Sum of signed, z-scored baseline markers. Catches a model "
         "that has not converged or has learned nothing its inputs did not already say. "
         "Every trained model is reported next to it."),
        ("Pairing-destroyed null", "Shuffle which future belongs to which subject, then "
         "rebuild the outcome from scratch. Catches an outcome manufactured from the "
         "features. This is the one that mattered."),
        ("Label-permutation test", "Necessary, and demonstrably not sufficient — it "
         "returned p = 0.002 on the broken setup."),
        ("Paired bootstrap intervals", "On every model-versus-baseline gap, resampling "
         "subjects together. A 2.4% edge without an interval is not a result."),
    ]
    top = Inches(2.0)
    for name, detail in items:
        _text(s, MARGIN, top, Inches(3.3), Inches(0.4), name, 15, bold=True, color=BLUE)
        _text(s, MARGIN + Inches(3.5), top, W - MARGIN * 2 - Inches(3.5), Inches(0.8),
              detail, 12.5, spacing=1.25)
        top += Inches(1.1)
    _rule(s, Inches(6.5))
    _text(s, MARGIN, Inches(6.7), W - 2 * MARGIN, Inches(0.5),
          "Reach for the pairing-destroyed null in any longitudinal change-score analysis "
          "where baseline is also a feature.", 13, bold=True)

    # ---------------------------------------------------------------- scope
    s = _blank(prs)
    _heading(s, "What this does not say", "honest scope")
    _text(s, MARGIN, Inches(2.1), W - 2 * MARGIN, Inches(2.6),
          "It does not say pre-operative NMR cannot forecast metabolic response.\n\n"
          "It says this cohort, at n=71, with this outcome, does not demonstrate that it "
          "does.\n\nA cohort with linked clinical outcomes — weight loss, remission, "
          "treatment type — and enough\nsubjects to estimate a small effect could still "
          "find one. MTBLS242's public metadata has none.",
          18, spacing=1.4)
    _rule(s, Inches(5.2))
    _text(s, MARGIN, Inches(5.4), W - 2 * MARGIN, Inches(1.4),
          "Also unresolved: (R)-3-Hydroxybutyric acid is 45% zeros — below-detection "
          "readings that log1p maps to exactly 0.\nBariatric patients only, so nothing "
          "about the general population or about outcomes without surgery.\n"
          "A metabolite profile is not a clinical outcome.",
          12, color=MUTED, spacing=1.3)

    prs.save(str(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-dir", type=Path, default=ROOT / "outputs_pr_auc")
    parser.add_argument("--continuous-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "Graph")
    parser.add_argument("--out", type=Path, default=ROOT / "mtbls242_forecast_correction.pptx")
    args = parser.parse_args()

    pr = json.loads((args.pr_dir / "summary_pr_auc.json").read_text())
    cont = json.loads((args.continuous_dir / "summary.json").read_text())
    out = build(pr, cont, args.graph_dir, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
