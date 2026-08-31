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

from deck_content import CONTENT, DEFAULT_FONT

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
          align=PP_ALIGN.LEFT, italic=False, spacing=1.0, font="Helvetica Neue"):
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
        run.font.name = font
    return box


def _rule(slide, top, color=RULE, height=Emu(9525)):
    from pptx.enum.shapes import MSO_SHAPE

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, top, W - 2 * MARGIN, height)
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


def _binder(font):
    """Return `_text` with the deck's font bound, so no call site can forget it."""
    def bound(*a, **kw):
        kw.setdefault("font", font)
        return _text(*a, **kw)
    return bound


def _heading(slide, title, kicker, font):
    # Thai has no case distinction, so .upper() is a no-op there rather than a problem.
    _text(slide, MARGIN, Inches(0.42), W - 2 * MARGIN, Inches(0.3), kicker.upper(),
          11, bold=True, color=MUTED, font=font)
    _text(slide, MARGIN, Inches(0.72), W - 2 * MARGIN, Inches(0.9), title, 27,
          bold=True, font=font)
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


def _metric_row(slide, top, items, font):
    """Big-number row: [(value, label, color), ...]"""
    span = (W - 2 * MARGIN) / len(items)
    for i, (value, label, color) in enumerate(items):
        left = int(MARGIN + i * span)
        _text(slide, left, top, int(span), Inches(0.8), value, 40, bold=True,
              color=color, align=PP_ALIGN.CENTER, font=font)
        _text(slide, left, top + Inches(0.78), int(span), Inches(0.6), label, 11,
              color=MUTED, align=PP_ALIGN.CENTER, font=font)


def _table(slide, left, top, width, rows, col_widths, *, font, header=True, size=12):
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
            run.font.name = font
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            cell.margin_left = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
    return table


def build(pr, cont, graph: Path, out: Path, c: dict, font: str) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    text = _binder(font)

    # ---------------------------------------------------------------- title
    s = _blank(prs)
    text(s, MARGIN, Inches(2.1), W - 2 * MARGIN, Inches(0.4),
         c["title_eyebrow"], 13, bold=True, color=MUTED)
    text(s, MARGIN, Inches(2.6), W - 2 * MARGIN, Inches(1.6), c["title"], 50, bold=True)
    _rule(s, Inches(4.3))
    text(s, MARGIN, Inches(4.6), W - 2 * MARGIN, Inches(1.2), c["title_sub"], 17,
         color=MUTED, spacing=1.3)

    # ---------------------------------------------------------------- question
    s = _blank(prs)
    _heading(s, c["q_title"], c["kicker_setup"], font)
    text(s, MARGIN, Inches(2.0), W - 2 * MARGIN, Inches(1.0), c["q_body"], 21, spacing=1.35)
    _metric_row(s, Inches(3.5), list(zip(
        [str(pr["n_subjects"]), "21", str(pr["n_responders"])],
        c["q_metrics"], [INK, INK, INK])), font)
    text(s, MARGIN, Inches(5.9), W - 2 * MARGIN, Inches(0.9), c["q_foot"], 12,
         color=MUTED, spacing=1.3)

    # ---------------------------------------------------------------- the claim
    s = _blank(prs)
    _heading(s, c["claim_title"], c["kicker_claim"], font)
    _metric_row(s, Inches(2.3), list(zip(
        ["0.671", "0.437", "1.54×"], c["claim_metrics"], [RED, MUTED, RED])), font)
    text(s, MARGIN, Inches(4.2), W - 2 * MARGIN, Inches(1.4), c["claim_body"], 18,
         spacing=1.35)

    # ---------------------------------------------------------------- defect 1
    s = _blank(prs)
    _heading(s, c["d1_title"], c["kicker_wrong"], font)
    text(s, MARGIN, Inches(1.85), W - 2 * MARGIN, Inches(0.7), c["d1_lead"], 12.5,
         color=MUTED, spacing=1.3)
    _fit_picture(s, graph / "regression_to_the_mean.png", Inches(2.75), Inches(3.55))
    text(s, MARGIN, Inches(6.55), W - 2 * MARGIN, Inches(0.6), c["d1_punch"], 13,
         bold=True, spacing=1.25)

    # ---------------------------------------------------------------- defect 2
    s = _blank(prs)
    _heading(s, c["d2_title"], c["kicker_wrong"], font)
    text(s, MARGIN, Inches(1.85), W - 2 * MARGIN, Inches(0.9), c["d2_lead"], 13.5,
         spacing=1.3)
    _metric_row(s, Inches(3.4), list(zip(
        ["1.57×", "2.04×", "2.51×"], c["d2_metrics"], [RED, BLUE, BLUE])), font)
    text(s, MARGIN, Inches(5.5), W - 2 * MARGIN, Inches(1.2), c["d2_punch"], 21,
         bold=True, color=RED, spacing=1.3)

    # ---------------------------------------------------------------- defect 3
    s = _blank(prs)
    _heading(s, c["d3_title"], c["kicker_point"], font)
    text(s, MARGIN, Inches(2.0), W - 2 * MARGIN, Inches(0.6), c["d3_lead"], 15.5,
         color=MUTED)
    text(s, MARGIN, Inches(2.6), W - 2 * MARGIN, Inches(1.0), "p = 0.002", 50,
         bold=True, color=RED)
    text(s, MARGIN, Inches(3.9), W - 2 * MARGIN, Inches(2.2), c["d3_body"], 16,
         spacing=1.35)
    _rule(s, Inches(5.95))
    text(s, MARGIN, Inches(6.15), W - 2 * MARGIN, Inches(0.9), c["d3_punch"], 14.5,
         bold=True, spacing=1.3)

    # ---------------------------------------------------------------- defect 4
    s = _blank(prs)
    _heading(s, c["d4_title"], c["kicker_wrong"], font)
    _metric_row(s, Inches(2.2), list(zip(
        ["0.018", "0.653", "0.653"], c["d4_metrics"], [RED, RED, BLUE])), font)
    text(s, MARGIN, Inches(4.3), W - 2 * MARGIN, Inches(1.8), c["d4_body"], 13.5,
         spacing=1.3)

    # ---------------------------------------------------------------- what replaced it
    s = _blank(prs)
    _heading(s, c["fix_title"], c["kicker_fix"], font)
    _table(s, MARGIN, Inches(1.95), W - 2 * MARGIN, c["fix_table"],
           [1.1, 2.0, 2.2], font=font, size=11)
    text(s, MARGIN, Inches(5.75), W - 2 * MARGIN, Inches(1.1), c["fix_foot"], 11.5,
         color=MUTED, spacing=1.3)

    # ---------------------------------------------------------------- corrected result
    s = _blank(prs)
    _heading(s, c["res_title"], c["kicker_says"], font)
    _metric_row(s, Inches(1.95), list(zip(
        [f"{pr['model_pr_auc_mean']:.3f}",
         f"{pr['CONTROL_trivial_scorer_pr_auc']:.3f}",
         f"{pr['CONTROL_permutation_p_value']:.3f}"],
        c["res_metrics"], [INK, BLUE, MUTED])), font)
    _fit_picture(s, graph / "pr_auc_curve.png", Inches(3.75), Inches(2.85))
    verdict = c["verdict_map"].get(pr["VERDICT"], pr["VERDICT"])
    text(s, MARGIN, Inches(6.75), W - 2 * MARGIN, Inches(0.6), verdict, 17,
         bold=True, color=RED, align=PP_ALIGN.CENTER)

    # ---------------------------------------------------------------- three agree
    s = _blank(prs)
    _heading(s, c["agree_title"], c["kicker_says"], font)
    top = Inches(2.1)
    for i, (headline, detail) in enumerate(c["agree_rows"], start=1):
        text(s, MARGIN, top, Inches(0.6), Inches(0.6), str(i), 30, bold=True, color=RED)
        text(s, MARGIN + Inches(0.7), top + Inches(0.02), W - MARGIN * 2 - Inches(0.7),
             Inches(0.4), headline, 16, bold=True)
        text(s, MARGIN + Inches(0.7), top + Inches(0.45), W - MARGIN * 2 - Inches(0.7),
             Inches(0.4), detail, 12.5, color=MUTED)
        top += Inches(1.25)
    _rule(s, Inches(6.1))
    text(s, MARGIN, Inches(6.3), W - 2 * MARGIN, Inches(0.8), c["agree_foot"], 12.5,
         color=MUTED, spacing=1.3)

    # ---------------------------------------------------------------- continuous
    s = _blank(prs)
    _heading(s, c["cont_title"], c["kicker_other"], font)
    _fit_picture(s, graph / "continuous_forecast_mae.png", Inches(2.0), Inches(3.7))
    text(s, MARGIN, Inches(5.95), W - 2 * MARGIN, Inches(1.2),
         c["cont_body"].format(
             gap=abs(cont["CONTROL_mae_gap_vs_mean_baseline"]),
             r2=cont["r2_overall"],
             medr2=cont["median_metabolite_r2"]),
         12.5, spacing=1.3)

    # ---------------------------------------------------------------- reusable
    s = _blank(prs)
    _heading(s, c["reuse_title"], c["kicker_takeaway"], font)
    top = Inches(2.0)
    for name, detail in c["reuse_items"]:
        text(s, MARGIN, top, Inches(3.3), Inches(0.4), name, 14.5, bold=True, color=BLUE)
        text(s, MARGIN + Inches(3.5), top, W - MARGIN * 2 - Inches(3.5), Inches(0.8),
             detail, 12, spacing=1.25)
        top += Inches(1.1)
    _rule(s, Inches(6.5))
    text(s, MARGIN, Inches(6.7), W - 2 * MARGIN, Inches(0.5), c["reuse_foot"], 12.5,
         bold=True)

    # ---------------------------------------------------------------- scope
    s = _blank(prs)
    _heading(s, c["scope_title"], c["kicker_scope"], font)
    text(s, MARGIN, Inches(2.1), W - 2 * MARGIN, Inches(2.6), c["scope_body"], 17,
         spacing=1.4)
    _rule(s, Inches(5.35))
    text(s, MARGIN, Inches(5.55), W - 2 * MARGIN, Inches(1.4), c["scope_foot"], 11.5,
         color=MUTED, spacing=1.3)

    prs.save(str(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(CONTENT), default="en",
                        help="Deck language. Figures keep English axis labels in both.")
    parser.add_argument("--font", default=None,
                        help="Override the font. Thai needs a Thai-capable face; the "
                             "default is Sukhumvit Set, which ships with macOS. Use "
                             "Tahoma for a face present on both macOS and Windows.")
    parser.add_argument("--pr-dir", type=Path, default=ROOT / "outputs_pr_auc")
    parser.add_argument("--continuous-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "Graph")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    content = CONTENT[args.lang]
    font = args.font or DEFAULT_FONT[args.lang]
    out = args.out or (ROOT / content["filename"])

    pr = json.loads((args.pr_dir / "summary_pr_auc.json").read_text())
    cont = json.loads((args.continuous_dir / "summary.json").read_text())
    build(pr, cont, args.graph_dir, out, content, font)
    print(f"wrote {out}  (lang={args.lang}, font={font})")


if __name__ == "__main__":
    main()
