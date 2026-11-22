"""Matplotlib figures for the Traffic Sign Visibility Audit (no Seaborn).

``make_all_figures(audit)`` saves eight PNG files to ``audit.fig_dir`` and
returns a dict ``{filename: caption}``. Captions are built from the measured
numbers so they stay correct whenever the data or the seed changes.
"""
from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt
import numpy as np

from traffic_audit import CLASS_NAMES, FEATURE_COLUMNS, load_rgb, wilson_interval

BLUE, ORANGE, GREY, RED = "#3b6ea5", "#d1691f", "#8a8f98", "#b03a2e"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 11,
                     "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5})


def _save(fig, audit, name, caption):
    """Add the caption under the figure, save, close; return the caption."""
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(caption, 125)), fontsize=7.8,
             va="bottom", ha="left", color="#333333")
    fig.savefig(audit.fig_dir / name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return caption


def _label(cid):
    return f"{cid}\n{CLASS_NAMES[cid]}"


def fig_contact_sheet(audit):
    t = audit.flagged[audit.flagged["status"] == "ok"]
    rng = np.random.default_rng(audit.seed)
    flagged = t[t["review_flag"]].sort_values(["n_triggers", "sharpness"], ascending=[False, True])
    # spread the flagged tiles over classes: best-ranked image of each class first
    flagged = flagged.groupby("class_id", sort=False).head(3).head(12)
    pool = t[~t["review_flag"]]
    unflagged = pool.iloc[rng.choice(len(pool), size=min(12, len(pool)), replace=False)]
    fig, axes = plt.subplots(4, 6, figsize=(10.5, 7.6))
    slots = list(flagged.iterrows())[:12]
    slots += [None] * (12 - len(slots)) + list(unflagged.iterrows())   # unflagged always in rows 3-4
    for ax, item in zip(axes.ravel(), slots):
        if item is None:
            ax.axis("off")
            continue
        _, r = item
        roi = (r["roi_x1"], r["roi_y1"], r["roi_x2"], r["roi_y2"]) if r["roi_ok_annotation"] else None
        img = load_rgb(audit.data_dir / r["image_path"], roi)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        col = ORANGE if r["review_flag"] else BLUE
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(2.2)
        ax.set_title(f"cls {int(r['class_id'])} | {'FLAG' if r['review_flag'] else 'ok'}", fontsize=8, color=col)
        ax.set_xlabel(f"B={r['brightness']:.0f} C={r['contrast']:.0f}\nS={r['sharpness']:.0f} px={r['min_side']:.0f}",
                      fontsize=7)
    fig.suptitle("Contact sheet: flagged (orange, top rows) vs unflagged (blue, bottom rows) sign crops",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    cap = ("Figure 1. Sign crops (annotated ROI, resized to 96 px for display). B = mean grey level, C = grey-level "
           "std, S = Laplacian variance (sharpness proxy), px = shorter ROI side. Flagged crops tend to be small, "
           "dark or soft, but the unflagged row also contains imperfect images, so the flag is a triage aid only.")
    return _save(fig, audit, "fig01_contact_sheet.png", cap)


def fig_distributions(audit):
    t = audit.flagged[audit.flagged["status"] == "ok"]
    c = audit.cutoffs
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(t["brightness"], bins=40, color=BLUE, alpha=.85)
    axes[0].axvline(c["brightness"], color=RED, ls="--", lw=1.6, label=f"P10 cutoff = {c['brightness']:.1f}")
    axes[0].set_xlabel("Mean grey level (0 = black, 255 = white)")
    axes[0].set_ylabel("Number of sampled images")
    axes[0].set_title("Brightness of sign crops")
    axes[0].legend(frameon=False)
    sharp = np.log10(t["sharpness"].clip(lower=1e-3))
    axes[1].hist(sharp, bins=40, color=BLUE, alpha=.85)
    axes[1].axvline(np.log10(c["sharpness"]), color=RED, ls="--", lw=1.6,
                    label=f"P10 cutoff = {c['sharpness']:.0f}")
    axes[1].set_xlabel("log10 of Laplacian variance (higher = more edge detail)")
    axes[1].set_ylabel("Number of sampled images")
    axes[1].set_title("Sharpness proxy of sign crops")
    axes[1].legend(frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    cap = (f"Figure 2. Distributions over the {len(t)} measured sample images with the project-specific 10th-percentile "
           "cutoffs (dashed). By construction about 10% of images fall below each line; the cutoffs are triage "
           "settings, not safety standards.")
    return _save(fig, audit, "fig02_distributions_cutoffs.png", cap)


def fig_boxplots(audit):
    t = audit.flagged[audit.flagged["status"] == "ok"]
    order = [int(i) for i in audit.class_summary().index]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, f, yl in zip(axes, ["brightness", "contrast", "sharpness"],
                         ["Mean grey level", "Grey-level std", "Laplacian variance (log scale)"]):
        data = [t.loc[t["class_id"] == k, f].to_numpy() for k in order]
        ax.boxplot(data, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#cfdcec", color=BLUE), medianprops=dict(color=RED))
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([str(k) for k in order])
        ax.set_xlabel("GTSRB class id (ordered by full-dataset image count)")
        ax.set_ylabel(yl)
        ax.set_title(f"{f.capitalize()} by class")
        if f == "sharpness":
            ax.set_yscale("log")
        ax.axhline(audit.cutoffs[f], color=ORANGE, ls="--", lw=1.2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    d = audit.feature_differences()
    cap = ("Figure 3. Per-class distributions (boxes = IQR, line = median, outliers hidden, orange dashed = P10 cutoff). "
           f"Between-class share of rank variance (eta-squared): brightness {d.loc['brightness','rank_eta2']:.2f}, "
           f"contrast {d.loc['contrast','rank_eta2']:.2f}, sharpness {d.loc['sharpness','rank_eta2']:.2f}.")
    return _save(fig, audit, "fig03_class_boxplots.png", cap)


def fig_flag_rate(audit):
    rates, agree, top3, overall = audit.sensitivity_summary()
    cs = audit.class_summary()
    order = list(rates["baseline (P10, >=2)"].sort_values(ascending=False).index)
    base = audit.flagged.groupby("class_id")["review_flag"].agg(["sum", "count"]).loc[order]
    lo, hi = wilson_interval(base["sum"], base["count"])
    p = 100 * base["sum"] / base["count"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(order))
    ax.bar(x - .2, p, width=.4, color=BLUE, label="baseline (P10, >=2 triggers)",
           yerr=[p - 100 * lo, 100 * hi - p], capsize=3, error_kw=dict(ecolor="#333", lw=1))
    ax.bar(x + .2, rates.loc[order, "looser (P20, >=2)"], width=.4, color="#9db7d5", label="looser (P20, >=2)")
    ax.scatter(x + .2, rates.loc[order, "stricter (P5, >=2)"], color=ORANGE, marker="D", s=28, zorder=3,
               label="stricter (P5, >=2)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_label(k)}\n(n={int(cs.loc[k, 'n_sample'])})" for k in order], fontsize=7)
    ax.set_xlabel("GTSRB class (sample size n)")
    ax.set_ylabel("Images flagged for review (%)")
    ax.set_title("Review-flag rate by class under three cutoff settings")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    cap = (f"Figure 4. Share of sampled images flagged per class (bars; whiskers = 95% Wilson interval for the baseline). "
           f"Overall baseline flag rate {100*audit.flagged['review_flag'].mean():.1f}%. Rank agreement with the baseline "
           f"(Spearman): P5 {agree['stricter (P5, >=2)']:.2f}, P20 {agree['looser (P20, >=2)']:.2f}, "
           f"any-trigger {agree['any trigger (P10, >=1)']:.2f}.")
    return _save(fig, audit, "fig04_flag_rate_by_class.png", cap)


def fig_size_contrast(audit):
    t = audit.flagged[audit.flagged["status"] == "ok"]
    order = [int(i) for i in audit.class_summary().index]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, f, yl, ttl in zip(axes, ["min_side", "contrast"],
                              ["Shorter ROI side (pixels)", "Grey-level standard deviation"],
                              ["Image size by class", "Contrast by class"]):
        data = [t.loc[t["class_id"] == k, f].to_numpy() for k in order]
        ax.boxplot(data, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#cfdcec", color=BLUE), medianprops=dict(color=RED))
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([str(k) for k in order])
        ax.set_xlabel("GTSRB class id (ordered by full-dataset image count)")
        ax.set_ylabel(yl)
        ax.set_title(ttl)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    d = audit.feature_differences()
    cap = ("Figure 5. Size and contrast by class. Class medians of the shorter ROI side differ by "
           f"{d.loc['min_side','median_gap_in_IQR']:.2f} overall IQRs, those of contrast by "
           f"{d.loc['contrast','median_gap_in_IQR']:.2f}; small images are expected to look soft after resizing.")
    return _save(fig, audit, "fig05_size_contrast_by_class.png", cap)


def fig_association(audit):
    t = audit.flagged[audit.flagged["status"] == "ok"]
    a = audit.association()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f, xl in zip(axes, ["contrast", "brightness"], ["Contrast (grey-level std)", "Brightness (mean grey level)"]):
        ax.scatter(t[f], t["sharpness"], s=5, alpha=.25, color=BLUE, rasterized=True)
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("Sharpness proxy (Laplacian variance, log)")
        ax.set_title(f"Sharpness vs {f}  (Spearman rho = {a.loc['sharpness', f]:.2f})")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    cap = ("Figure 6. Pairwise association of the sharpness proxy with contrast and brightness (Spearman rank "
           "correlation on the sample). The association describes how the measurements co-vary; it does not show "
           "that low contrast or darkness causes low sharpness. Images from one track are near-duplicates, so the "
           "effective sample is smaller than the point count.")
    return _save(fig, audit, "fig06_sharpness_association.png", cap)


def fig_examples(audit):
    ex = audit.pick_examples()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6))
    for ax, (label, r) in zip(axes, ex):
        row = audit.flagged[audit.flagged["image_path"] == r["image_path"]].iloc[0]
        roi = (row["roi_x1"], row["roi_y1"], row["roi_x2"], row["roi_y2"]) if row["roi_ok_annotation"] else None
        ax.imshow(load_rgb(audit.data_dir / r["image_path"], roi, upscale=160))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label}\nclass {int(r['class_id'])}: {CLASS_NAMES[int(r['class_id'])]}", fontsize=9.5)
        ax.set_xlabel(f"brightness {r['brightness']:.0f} | contrast {r['contrast']:.0f}\n"
                      f"sharpness {r['sharpness']:.0f} | size {r['min_side']:.0f} px | triggers {int(r['n_triggers'])}",
                      fontsize=8.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    cap = ("Figure 7. Three images chosen by a fixed procedure: the flagged image with most triggers and lowest "
           "sharpness; the flagged image closest to its cutoffs (possible false alarm); and the unflagged image closest "
           "to the sample medians (reference).")
    return _save(fig, audit, "fig07_example_images.png", cap)


def fig_class_counts(audit):
    full = audit.report["full_class_counts"]
    ids = sorted(full)
    fig, ax = plt.subplots(figsize=(11, 4))
    cols = [ORANGE if i in audit.classes else GREY for i in ids]
    ax.bar(range(len(ids)), [full[i] for i in ids], color=cols)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([str(i) for i in ids], fontsize=7)
    ax.set_xlabel("GTSRB class id")
    ax.set_ylabel("Annotation rows (full dataset)")
    ax.set_title("Full-dataset class counts (selected classes in orange)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    sel = [full[i] for i in audit.classes]
    cap = (f"Figure 8. Counts use ALL valid annotation rows ({sum(full.values())}). Selected classes range from "
           f"{min(sel)} to {max(sel)} images ({max(sel)/min(sel):.1f}x); the image analysis uses at most "
           f"{audit.per_class_cap} images per class, so sampled counts are more balanced than the full counts.")
    return _save(fig, audit, "fig08_class_counts_full.png", cap)


def make_all_figures(audit):
    """Create all eight figures; return ``{file name: caption}``."""
    caps = {}
    for fn in (fig_contact_sheet, fig_distributions, fig_boxplots, fig_flag_rate,
               fig_size_contrast, fig_association, fig_examples, fig_class_counts):
        caps[fn.__name__] = fn(audit)
    return {f"fig{i+1:02d}": c for i, c in enumerate(caps.values())}
