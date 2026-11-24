"""10-slide PPTX for the viva/presentation, bullets only, every font >= 24 pt, numbers read from the same generated
files as the report (report/generated/numbers.json + results/iters.json), so slides and report cannot disagree.

    python scripts/make_report_assets.py --results results --out report/generated
    python scripts/make_slides.py --numbers report/generated/numbers.json --iters results/iters.json --out results/slides.pptx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

TITLE_PT, BODY_PT, MIN_PT = 36, 24, 24


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def cell(nums, scene, arm, n=8):
    for c in nums["cells"]:
        if c["scene"] == scene and c["arm"] == arm and int(c["n"]) == n:
            return c
    return None


def dense(nums, scene):
    cs = [c for c in nums["cells"] if c["scene"] == scene and c["arm"] == "A" and int(c["n"]) > 32]
    return cs[0] if cs else None


def gap(nums, scene, arm="D"):
    for g in nums["gap"]:
        if g["scene"] == scene and g["arm"] == arm:
            return g
    return None


def add_slide(prs, title, bullets):
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    for p in s.shapes.title.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(TITLE_PT)
    body = s.placeholders[1].text_frame
    body.clear()
    for i, b in enumerate(bullets):
        level = 1 if b.startswith("  ") else 0
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text, p.level = b.strip(), level
        for r in p.runs:
            r.font.size = Pt(BODY_PT if level == 0 else MIN_PT)
    return s


def build(nums: dict, iters: dict, extra: dict) -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for layout in prs.slide_layouts:          # widen placeholders to the 16:9 page
        for ph in layout.placeholders:      # set all four: setting only left/width zeroes top/height
            ph.left, ph.width = Inches(0.6), Inches(12.1)
            if ph.placeholder_format.idx == 0:
                ph.top, ph.height = Inches(0.35), Inches(1.1)
            else:
                ph.top, ph.height = Inches(1.6), Inches(5.5)
    I = iters.get("iters", "?")
    scenes = [s for s in ("lego", "chair", "real") if cell(nums, s, "A")]

    add_slide(prs, "Sparse-View NeRF with FreeNeRF", [
        "UCS532 Computer Vision, Topic 16",
        "Waris (1024170155), Tanishq (1024170156)",
        "Supervisor: Dr. Shailendra Tiwari",
        "Code: github.com/warismittal/freenerf-nerfstudio"])
    add_slide(prs, "Problem", [
        "NeRF needs many posed views",
        "  100 views: sharp novel views; 8 views: floaters and haze",
        "Task: implement one published regulariser",
        "Question: how much of the 100-to-8 gap does it recover?",
        "  rho = (PSNR_reg - PSNR_van) / (PSNR_dense - PSNR_van)"])
    add_slide(prs, "Method: FreeNeRF in nerfstudio", [
        "Base: nerfstudio vanilla-nerf (reused)",
        "Frequency mask: open encoding bands linearly over 70% of training",
        "Occlusion loss: penalise density in the first 20 samples, weight 0.01",
        "Comparator: RegNeRF depth smoothness on unseen-pose patches",
        "Separate evaluator: PSNR, SSIM, LPIPS, rho, paired bootstrap"])
    add_slide(prs, "Setup", [
        "NeRF-synthetic Lego and Chair, 400x400, DietNeRF 8-view split",
        "25 held-out test views (every 8th test image)",
        f"{I} iterations for every arm, 4096 rays per batch"
        + (", mixed precision" if iters.get("amp") else ""),
        f"Budget rule: {fmt(iters.get('s_per_iter'), 3)} s/iter measured on a Kaggle T4",
        "Seeds fixed and logged; 3 seeds for the main comparison"])
    rows = []
    for s in scenes:
        a, d, dn, g = cell(nums, s, "A"), cell(nums, s, "D"), dense(nums, s), gap(nums, s)
        rows.append(f"{s.capitalize()}: vanilla {fmt(a['psnr'])} dB -> FreeNeRF {fmt(d['psnr']) if d else '--'} dB")
        rows.append(f"  LPIPS {fmt(a.get('lpips_alex'), 3)} -> {fmt(d.get('lpips_alex'), 3) if d else '--'}"
                    + (f"; 100 views {fmt(dn['psnr'])} dB" if dn else ""))
        if g:
            rows.append(f"  rho = {fmt(g['rho_psnr'])} [{fmt(g['rho_psnr_lo'])}, {fmt(g['rho_psnr_hi'])}]")
    add_slide(prs, "Main result: 8 input views", rows[:9])
    ab = []
    for s in ("lego", "chair"):
        parts = []
        for arm, name in (("B", "mask only"), ("C", "occlusion only"), ("D", "both")):
            g = gap(nums, s, arm)
            if g:
                parts.append(f"{name} {g['dpsnr_vs_A']:+.2f}")
        if parts:
            ab.append(f"{s.capitalize()} dPSNR vs vanilla: " + ", ".join(parts))
    add_slide(prs, "Ablation: which half does the work?", ab + extra.get("ablation", []))
    cross = Path(__file__).resolve().parents[1] / "results/crosscheck_pytorch/lego_s42/comparison.png"
    if cross.exists():   # our independent PyTorch implementation, matched Lego orbit
        s = add_slide(prs, "Independent check: a second code base", [])
        body = s.placeholders[1]
        body._element.getparent().remove(body._element)
        s.shapes.add_picture(str(cross), Inches(0.6), Inches(1.6), height=Inches(5.6))
        tf = s.shapes.add_textbox(Inches(7.6), Inches(1.8), Inches(5.2), Inches(5)).text_frame
        tf.word_wrap = True
        for i, line in enumerate(["Our separate plain PyTorch NeRF", "Frequency mask only, 75k steps, Lego",
                                  "Rows: 100 views, 8 views, 8 views + mask",
                                  "8-view floaters above the bucket disappear with the mask"]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.space_after = Pt(14)
            for r in p.runs:
                r.font.size = Pt(BODY_PT)
    else:
        add_slide(prs, "More views or a regulariser?", extra.get("views", ["See report Table 3"]))
    add_slide(prs, "Failure analysis", extra.get("failure", ["See report Section 5"]))
    add_slide(prs, "Changes from the proposal", extra.get("changes", ["See report Section 6"]))
    add_slide(prs, "Conclusion and declarations", extra.get("conclusion", []) + [
        "Design, mapping, flowcharts, PyTorch cross-check: Waris and Tanishq",
        "AI use (declared): Claude wrote the code, ran the Kaggle jobs, drafted report and slide text",
        "All numbers come from logged runs in the public repository"])
    return prs


def check_fonts(prs: Presentation) -> int:
    small = 0
    for s in prs.slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        if r.font.size is not None and r.font.size < Pt(MIN_PT):
                            small += 1
    return small


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", default="report/generated/numbers.json")
    ap.add_argument("--iters", default="results/iters.json")
    ap.add_argument("--extra", default="report/slides_text.json", help="hand-written bullets (no numbers)")
    ap.add_argument("--out", default="results/slides.pptx")
    a = ap.parse_args(argv)
    nums = json.loads(Path(a.numbers).read_text())
    iters = json.loads(Path(a.iters).read_text()) if Path(a.iters).exists() else {}
    extra = json.loads(Path(a.extra).read_text()) if Path(a.extra).exists() else {}
    prs = build(nums, iters, extra)
    assert len(prs.slides) <= 10, "brief: at most 10 slides"
    assert check_fonts(prs) == 0, "brief: every font must be >= 24 pt"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prs.save(a.out)
    print(f"wrote {a.out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
