"""Build the written report (.docx) from the audit outputs.

    python src/build_report.py --out-dir outputs --notes report/student_notes.md \
                               --report report/Traffic_Sign_Audit_Report.docx

All numbers and the data-driven sentences come from ``results.json`` and the
CSV tables written by ``run_audit.py``, so the text always matches the results.
The student's own reasoning is read from ``student_notes.md`` (see that file).
If the outputs were produced from the synthetic demo data, a red banner says so.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from traffic_audit import CLASS_NAMES  # noqa: E402

CITATION = ("Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. German Traffic Sign Recognition Benchmark "
            "(GTSRB). Real-Time Computer Vision Group, Institut für Neuroinformatik, Ruhr-Universität Bochum. "
            "https://benchmark.ini.rub.de/gtsrb_dataset.html (archival copy: https://zenodo.org/records/13741936).")


# ---------------------------------------------------------------- helpers
def read_notes(path):
    """Parse ``## Heading`` sections of the student notes; empty/TODO sections -> None."""
    notes = {}
    p = Path(path)
    if p.exists():
        for m in re.finditer(r"^##\s+(.+?)\s*\n(.*?)(?=^##\s|\Z)", p.read_text(encoding="utf-8"),
                             flags=re.S | re.M):
            body = re.sub(r"<!--.*?-->", "", m.group(2), flags=re.S).strip()
            notes[m.group(1).strip()] = None if (not body or body.startswith("TODO")) else body
    return notes


def shade(run, color="FFFF00"):
    rpr = run._r.get_or_add_rPr()
    h = OxmlElement("w:shd")
    h.set(qn("w:val"), "clear"); h.set(qn("w:fill"), color)
    rpr.append(h)


def para(doc, text, bold=False, italic=False, size=10, color=None, space=4, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    p.paragraph_format.space_after = Pt(space)
    if align:
        p.alignment = align
    return p


def student_block(doc, notes, key, prompt):
    """Insert the student's own words, or a highlighted prompt if still missing."""
    text = notes.get(key)
    p = doc.add_paragraph()
    r = p.add_run(f"{key}: ")
    r.bold = True; r.font.size = Pt(10)
    r2 = p.add_run(text if text else f"[WRITE THIS IN YOUR OWN WORDS in report/student_notes.md - {prompt}]")
    r2.font.size = Pt(10)
    if not text:
        shade(r2)
    p.paragraph_format.space_after = Pt(4)


def table(doc, header, rows, widths=None, size=8.5):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(str(h)); run.bold = True; run.font.size = Pt(size)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(doc, path, width, caption=None):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(4)


def strength(rho):
    a = abs(rho)
    return "weak" if a < .3 else "moderate" if a < .6 else "strong"


def nm(c):
    return CLASS_NAMES.get(int(c), str(c))


# ---------------------------------------------------------------- build
def build(out_dir, notes_path, report_path, student="[Your name]", course="[Course / instructor]"):
    out = Path(out_dir)
    res = json.loads((out / "processed" / "results.json").read_text())
    cs = pd.read_csv(out / "processed" / "class_summary.csv", index_col=0)
    rates = pd.read_csv(out / "processed" / "sensitivity_flag_rates.csv", index_col=0)
    diff = pd.read_csv(out / "processed" / "feature_differences.csv", index_col=0)
    assoc = pd.read_csv(out / "processed" / "spearman_association.csv", index_col=0)
    ft = pd.read_csv(out / "processed" / "feature_table.csv")
    fig = out / "figures"
    notes = read_notes(notes_path)
    ins = res["inspection"]
    base = "baseline (P10, >=2)"
    cut = res["cutoffs"]

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)

    if res["demo_data"]:
        para(doc, "DEMO OUTPUT - computed from SYNTHETIC drawn shapes, not from GTSRB. Every number, figure and "
                  "sentence below is a pipeline test. Re-run on the real GTSRB archive and rebuild this report "
                  "before submitting.", bold=True, color="B00020", size=10)

    doc.add_heading("Traffic Sign Visibility Audit: which images should be reviewed first?", level=1)
    para(doc, f"{student} | {course} | Dataset: GTSRB training images | Tools: Python, NumPy, pandas, Matplotlib, "
              f"OpenCV (no deep learning)", italic=True, size=9)

    # 1 case question
    doc.add_heading("1. Case question and scope", level=2)
    para(doc, "A road-safety team cannot inspect every traffic-sign photograph by hand. The audit question is: "
              "which images should be reviewed first, and what measured evidence supports that priority? "
              "The workflow measures brightness, contrast, sharpness and size with OpenCV, converts them into a "
              "transparent percentile-based triage rule with NumPy, and compares classes with pandas. A flag is a "
              "triage aid for human review; it is not proof that a sign is unsafe, unreadable or misclassified.")
    sel = ", ".join(f"{c} ({nm(c)})" for c in res["classes"])
    para(doc, f"Selected classes ({len(res['classes'])}): {sel}. Full-dataset counts use all valid annotation rows; "
              f"image measurements use a seeded (seed {res['seed']}) stratified sample of at most "
              f"{res['per_class_cap']} images per class ({res['n_sample']} images in total).")
    nl = ins.get("n_rows_without_local_image", 0)
    if nl:
        para(doc, f"Data availability note: the annotation file (Train.csv) covers all {ins['n_rows_raw']:,} GTSRB training "
                  f"rows, so class counts are for the full dataset. Image files were available locally for only "
                  f"{ins['n_rows_raw'] - nl} of them, so the OpenCV measurements, rule and plots rest on "
                  f"{res['n_sample']} images from {res['n_tracks_sample']} tracks. Results at this size illustrate the "
                  f"workflow but are NOT statistically reliable; per-class rates below are based on 3-7 images.",
             bold=True, color="B00020", size=9.5)
    student_block(doc, notes, "Class selection rationale", "why these classes?")

    # 2 data prep
    doc.add_heading("2. Data inspection and preparation", level=2)
    miss = {k: v for k, v in ins["missing_values"].items() if v}
    para(doc, f"Full annotations: {ins['n_rows_raw']:,} rows in {ins['n_classes_raw']} classes. Duplicate rows: "
              f"{ins['n_duplicate_rows']}; missing values: {miss if miss else 'none'}; non-numeric cells: "
              f"{ins['n_non_numeric_cells']}; invalid image paths: {ins['n_invalid_image_paths']}; invalid class ids: "
              f"{ins['n_invalid_class_id']}; class-folder mismatches: {ins['n_class_folder_mismatch']}; invalid or "
              f"absent ROI: {ins['n_invalid_or_absent_roi']}. After exclusions {ins['n_rows_valid']:,} rows remain "
              f"({ins['n_excluded_total']} excluded, each logged with a reason in exclusion_log.csv).")
    para(doc, "Handling rules. (a) Exact duplicate annotation rows and rows whose image file is missing, whose class id is "
              "not 0-42 or does not match its folder, or whose width/height is not positive are excluded and logged. "
              "(b) Rows with an absent or invalid ROI (missing, outside the image, or shorter than 4 px) are kept and "
              "measured on the full image; this is a documented transformation, not a deletion. (c) An image that "
              "exists but cannot be decoded by cv2.imread stays in the table with status 'unreadable' and is always "
              "flagged for review. (d) Labels are the supplied ClassId values; nothing is inferred from colour. "
              "(e) The raw dataset is only read; all outputs go to outputs/.")
    sk = res["skipped_counts"]
    para(doc, f"Skipped/excluded record counts by reason: {sk if sk else 'none'}. "
              f"In the sample, {res['n_roi_fallback_sample']} images used the full-image fallback and "
              f"{res['n_dim_mismatch']} had annotated dimensions that differ from the decoded image.")

    # 3 features
    doc.add_heading("3. Feature definitions", level=2)
    table(doc, ["Feature", "Definition (OpenCV / NumPy)", "Unit / range"], [
        ["brightness", "mean of the grayscale ROI crop (cv2.cvtColor BGR2GRAY)", "grey level, 0-255"],
        ["contrast", "standard deviation of the same grayscale crop", "grey level, ~0-128"],
        ["sharpness", "variance of cv2.Laplacian (CV_64F) on the crop resized to 48x48 px", "unitless, >=0"],
        ["min_side", "shorter side of the ROI crop (full image if ROI invalid)", "pixels"],
        ["brightness_hsv_v", "mean of the HSV Value channel (cross-check)", "0-255"],
        ["*_full", "brightness, contrast, sharpness on the whole image", "as above"]],
        widths=[1.2, 4.2, 1.4])
    para(doc, "Sharpness is a proxy: the Laplacian variance responds to edges, noise and image size, and is not a human "
              "readability score. It is computed on a fixed 48x48 patch so images of different resolution are "
              "comparable; consequently very small images, which must be enlarged, tend to score as soft, which is "
              "intended for triage but means sharpness and size are partly entangled. OpenCV loads BGR; images "
              "are converted to RGB only for display.")
    rf = res["roi_vs_full"]
    para(doc, "ROI crop vs full image (seeded sample of " + str(rf["brightness"]["n"]) + "): " + "; ".join(
        f"{k} median {v['median_crop']:.1f} vs {v['median_full']:.1f} (Spearman {v['spearman']:.2f})"
        for k, v in rf.items()) + ".")

    # 4 rule
    doc.add_heading("4. Review-priority rule and sensitivity", level=2)
    para(doc, f"Rule (project-specific triage setting, not a safety standard). Compute the 10th percentile of "
              f"brightness, contrast, sharpness and min_side over the analysis sample (cutoffs: "
              f"{cut['brightness']:.1f}, {cut['contrast']:.1f}, {cut['sharpness']:.1f}, {cut['min_side']:.0f} px). "
              f"An image triggers one condition per feature below its cutoff. Tier 'review first' = at least 2 "
              f"triggers (or unreadable); 'watch' = 1 trigger; otherwise 'routine'. Flags are boolean NumPy masks "
              f"combined with np.select. Percentiles are pooled over the selected classes, so rates are relative to "
              f"this sample.")
    student_block(doc, notes, "Why I chose this rule", "justify percentile cutoffs and the 2-trigger requirement")
    tc = res["tier_counts"]
    para(doc, f"Result: {res['n_flag_baseline']} of {res['n_sample']} sampled images "
              f"({100*res['n_flag_baseline']/res['n_sample']:.1f}%) are flagged; tiers {tc}. Triggers: "
              f"{res['trigger_counts']}.")
    rows = []
    for lab in rates.columns:
        c = res["sensitivity_cutoffs"][lab]
        rows.append([lab, f"{c['brightness']:.0f} / {c['contrast']:.1f} / {c['sharpness']:.0f} / {c['min_side']:.0f}",
                     res["overall_flagged"][lab],
                     f"{100*res['overall_flagged'][lab]/res['n_sample']:.1f}%",
                     ", ".join(str(x) for x in res["top3"][lab]), f"{res['rank_agreement'][lab]:.2f}"])
    table(doc, ["Setting", "Cutoffs (bright / contrast / sharp / px)", "Flagged", "Share", "Top-3 classes",
                "Rank agreement"], rows, widths=[1.5, 2.0, .6, .6, 1.0, .8])

    # 5 findings
    doc.add_heading("5. Findings", level=2)
    big, small = cs["n_full"].idxmax(), cs["n_full"].idxmin()
    para(doc, f"Q1 Class counts and imbalance. Among the selected classes, class {big} ({nm(big)}) has the most "
              f"images ({int(cs.loc[big,'n_full']):,}) and class {small} ({nm(small)}) the fewest "
              f"({int(cs.loc[small,'n_full']):,}); the ratio is {cs.loc[big,'n_full']/cs.loc[small,'n_full']:.1f}x. "
              f"Capping at {res['per_class_cap']} images per class makes sampled counts more even "
              f"({int(cs['n_sample'].min())}-{int(cs['n_sample'].max())}), but small classes have wide uncertainty "
              f"(Figure 4 whiskers), and images come in tracks of 30 near-duplicates, so the effective sample is "
              f"closer to the track counts ({int(cs['n_tracks_sample'].min())}-{int(cs['n_tracks_sample'].max())} "
              f"tracks per class) than to the image counts.", size=10)
    mat = diff[diff["median_gap_in_IQR"] >= 0.5].index.tolist()
    para(doc, "Q2 Differences across classes. Between-class share of rank variance (eta-squared) / gap between the "
              "highest and lowest class median in overall IQRs: " + "; ".join(
                  f"{f} {r.rank_eta2:.2f} / {r.median_gap_in_IQR:.2f}" for f, r in diff.iterrows()) +
         f". Using a project rule of thumb that a median gap of at least 0.5 IQR matters for review priority, the "
         f"features that matter are: {', '.join(mat) if mat else 'none'}.")
    top, bot = rates[base].idxmax(), rates[base].idxmin()
    stable = res["rank_agreement"]
    para(doc, f"Q3 Flag rates. The baseline flag rate is highest for class {top} ({nm(top)}, {rates.loc[top, base]:.1f}%) "
              f"and lowest for class {bot} ({nm(bot)}, {rates.loc[bot, base]:.1f}%). Spearman rank agreement with the "
              f"alternatives: " + ", ".join(f"{k.split(' (')[0]} {v:.2f}" for k, v in stable.items() if k != base) +
         ". " + ("The top-3 set is identical in all settings." if len({tuple(sorted(v)) for v in res['top3'].values()}) == 1
                else "The top-3 set changes between settings (see table in section 4)."))
    rs = assoc.loc["sharpness"]
    para(doc, f"Q4 Association. Spearman correlation of sharpness with contrast is {rs['contrast']:.2f} "
              f"({strength(rs['contrast'])}), with brightness {rs['brightness']:.2f} ({strength(rs['brightness'])}) and "
              f"with ROI size {rs['min_side']:.2f} ({strength(rs['min_side'])}). These are descriptive associations on "
              f"images from a limited number of tracks; they do not show that darkness or low contrast causes blur.")
    ex = res["examples"]
    para(doc, "Q5 Illustrative images (Figure 7): " + " ".join(
        f"{e['label']}: class {e['class_id']} ({nm(e['class_id'])}), brightness {e['brightness']:.0f}, contrast "
        f"{e['contrast']:.1f}, sharpness {e['sharpness']:.0f}, size {e['min_side']:.0f} px, {e['n_triggers']} triggers."
        for e in ex))
    student_block(doc, notes, "What I see in the example images",
                  "after looking at Figures 1 and 7: where is the rule right, where is it a false alarm or a miss?")

    for f, w in [("fig01_contact_sheet.png", 5.6), ("fig02_distributions_cutoffs.png", 5.8),
                 ("fig03_class_boxplots.png", 6.4), ("fig04_flag_rate_by_class.png", 5.8),
                 ("fig05_size_contrast_by_class.png", 5.6), ("fig07_example_images.png", 5.4)]:
        figure(doc, fig / f, w)
    para(doc, "Additional figures saved in outputs/figures: fig06_sharpness_association.png (sharpness vs contrast and "
              "brightness) and fig08_class_counts_full.png (full-dataset class counts). Each figure carries its own "
              "caption.", italic=True, size=9)
    student_block(doc, notes, "My interpretation of the figures", "what surprised you? what would you check next?")

    # 6 memo (own page)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.add_heading("6. Decision memo for the road-safety team", level=2)
    para(doc, "To: Road-safety review team | Re: Prioritising manual review of traffic-sign images | "
              "Status: triage settings are project choices, not standards", italic=True, size=9)
    tot = rates[base]
    n_cls = cs["n_sample"]
    k = (tot / 100 * n_cls).round()
    from math import sqrt
    overall = res["n_flag_baseline"] / res["n_sample"]

    def lower(kk, nn, z=1.96):
        p = kk / nn
        d = 1 + z * z / nn
        return ((p + z * z / (2 * nn)) - z * sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn))) / d
    attention = [int(c) for c in tot.index if lower(k[c], n_cls[c]) > overall]
    drivers = sorted(res["trigger_counts"].items(), key=lambda kv: -kv[1])
    para(doc, "Recommendation", bold=True, space=1)
    for b in [
        f"1. Review unreadable files first ({res['n_unreadable']} in the sample): they cannot be assessed by the pipeline at all.",
        f"2. Next review the 'review first' tier ({tc.get('review first', 0)} images), ordered by number of triggers and "
        f"lowest sharpness. This is {100*tc.get('review first', 0)/res['n_sample']:.1f}% of the sample, a much smaller "
        f"workload than checking every file.",
        f"3. Classes that deserve attention first (95% interval for the flag rate above the overall "
        f"{100*overall:.1f}%): " + (", ".join(f"{c} ({nm(c)})" for c in attention) if attention else
                                     "none stands out beyond sampling noise") + ".",
        f"4. The most frequent triggers are " + ", ".join(f"{k.replace('flag_', '')} ({v})" for k, v in drivers[:2]) +
        "; reviewers should check these conditions first.",
        "5. Use the 'watch' tier only as a sampling frame for spot checks."]:
        para(doc, b, space=1)
    para(doc, "Evidence that would change this recommendation", bold=True, space=1)
    for b in ["- If a reviewer-checked random sample of flagged images shows mostly acceptable signs (high false-alarm "
              "rate), raise the trigger requirement or lower the percentile.",
              "- If unflagged images are found to be unreadable at a noticeable rate, add conditions (e.g. over-exposure, "
              "colour proportion) or switch to the looser setting.",
              f"- If class ranks change materially under other cutoffs (rank agreement here: "
              f"{min(v for k2, v in stable.items() if k2 != base):.2f} at the lowest), class-level conclusions should "
              f"be treated as tentative.",
              "- Analysis of the full dataset (not a capped sample) or data from local roads that contradicts these class patterns."]:
        para(doc, b, space=1)
    para(doc, "Limitations", bold=True, space=1)
    for b in ["- Sample: at most 500 images per class from a limited number of 30-image tracks of the same physical sign; "
              "percentile cutoffs are relative to the selected classes and do not describe the other classes.",
              "- Sharpness (Laplacian variance) is a proxy; it responds to noise and image size and is not a measure of human readability.",
              "- GTSRB photographs are German, from specific cameras and weather conditions, and do not represent every local road environment.",
              "- A numerical flag is a triage aid; it is not proof that a sign is unsafe or misclassified."]:
        para(doc, b, space=1)
    student_block(doc, notes, "Limitation I will discuss in class", "pick one limitation and explain it in your own words")

    doc.add_heading("7. Reproducibility and citation", level=2)
    para(doc, f"Run python run_audit.py then python src/build_report.py (see README.md). Seed {res['seed']}, per-class cap "
              f"{res['per_class_cap']}. Dataset citation: {CITATION}", size=9)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(report_path)
    return report_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--notes", default="report/student_notes.md")
    ap.add_argument("--report", default="report/Traffic_Sign_Audit_Report.docx")
    ap.add_argument("--student", default="[Your name]")
    ap.add_argument("--course", default="[Course / instructor]")
    a = ap.parse_args()
    print("Report written to", build(a.out_dir, a.notes, a.report, a.student, a.course))
