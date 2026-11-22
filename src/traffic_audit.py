"""Traffic Sign Visibility Audit - core workflow (GTSRB training images).

This module holds the reusable functions and the ``TrafficSignAudit`` class.
Nothing here trains a model: every number comes from OpenCV image
measurements, pandas summaries and NumPy percentile rules.

Reusable functions
------------------
load_annotations          read every GT-*.csv into one pandas table
inspect_annotations       validate columns / ROI / paths and report problems
crop_roi                  validate ROI coordinates and crop
extract_image_features    OpenCV brightness / contrast / sharpness (+ size)
assign_review_priority    NumPy percentile cutoffs + vectorised review flags
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1",
                    "Roi.X2", "Roi.Y2", "ClassId"]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
STD_SIZE = 48          # side (px) of the standard patch used for sharpness
MIN_ROI_SIDE = 4       # smallest ROI side (px) accepted as a usable crop
N_CLASSES_GTSRB = 43

CLASS_NAMES = {
    0: "Speed limit 20", 1: "Speed limit 30", 2: "Speed limit 50",
    3: "Speed limit 60", 4: "Speed limit 70", 5: "Speed limit 80",
    6: "End speed limit 80", 7: "Speed limit 100", 8: "Speed limit 120",
    9: "No passing", 10: "No passing >3.5t", 11: "Right-of-way at junction",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Vehicles >3.5t prohibited", 17: "No entry", 18: "General caution",
    19: "Curve left", 20: "Curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians",
    28: "Children crossing", 29: "Bicycles crossing", 30: "Ice/snow",
    31: "Wild animals", 32: "End all limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Straight or right",
    37: "Straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout", 41: "End no passing", 42: "End no passing >3.5t",
}

#: Feature columns that the rule and the summaries rely on.
FEATURE_COLUMNS = ["brightness", "contrast", "sharpness", "min_side"]


class ImageReadError(Exception):
    """Raised when OpenCV cannot decode an image file."""


# --------------------------------------------------------------------------
# 1. Annotations
# --------------------------------------------------------------------------
def load_annotations(data_dir):
    """Read all ``GT-*.csv`` files below ``data_dir`` into one DataFrame.

    Parameters
    ----------
    data_dir : str | Path
        Folder that contains one sub-folder per class (``00000`` ... ``00042``),
        each holding ``GT-<ClassID>.csv`` (semicolon separated) and the images.

    Returns
    -------
    (pandas.DataFrame, list[dict])
        The concatenated annotations (extra columns ``annotation_file`` and
        ``class_dir``) and a list of problems met while reading (unreadable or
        malformed CSV files). Raises FileNotFoundError with a hint if the folder
        or the CSV files do not exist.
    """
    data_dir = Path(data_dir)
    if (data_dir / "Train.csv").is_file():          # Kaggle-style copy: Train.csv + Train/<class>/*.png
        try:
            raw = pd.read_csv(data_dir / "Train.csv")
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ValueError(f"Train.csv is unreadable ({type(exc).__name__})") from exc
        need = [c for c in REQUIRED_COLUMNS[1:] if c not in raw.columns] + \
               [c for c in ("Path",) if c not in raw.columns]
        if need:
            raise ValueError(f"Train.csv is missing columns {need}")
        # keep only the part of Path from 'Train/' on, so no personal/absolute prefix is used
        rel = raw["Path"].astype(str).str.replace("\\", "/", regex=False) \
            .str.replace(r"^.*?(?=Train/)", "", regex=True)
        raw["Filename"] = rel.str.split("/").str[-1]
        raw["class_dir"] = rel.str.rsplit("/", n=1).str[0]       # e.g. 'Train/33'
        raw["annotation_file"] = "Train.csv"
        return raw.drop(columns=["Path"]), []
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder '{data_dir}' not found. Download GTSRB (see README, "
            "section 'Getting the data') and unzip it so that class folders such "
            "as '00000' sit directly inside this folder.")
    csv_files = sorted(data_dir.glob("*/GT-*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No 'GT-*.csv' annotation files found under '{data_dir}'. "
            "Check that you used the TRAINING archive (GTSRB_Final_Training_Images) "
            "or a folder containing Train.csv and Train/<class>/ images.")

    frames, problems = [], []
    for csv_path in csv_files:
        rel = csv_path.relative_to(data_dir).as_posix()
        try:
            part = pd.read_csv(csv_path, sep=";")
        except (pd.errors.EmptyDataError, pd.errors.ParserError,
                UnicodeDecodeError) as exc:
            problems.append({"file": rel, "problem": f"unreadable CSV ({type(exc).__name__})"})
            continue
        missing = [c for c in REQUIRED_COLUMNS if c not in part.columns]
        if missing:
            problems.append({"file": rel, "problem": f"missing columns {missing}"})
            continue
        part["annotation_file"] = rel
        part["class_dir"] = csv_path.parent.name
        frames.append(part)
    if not frames:
        raise ValueError("Every annotation file was unreadable - see 'problems'.")
    return pd.concat(frames, ignore_index=True), problems


def inspect_annotations(df, data_dir):
    """Validate the annotation table and return ``(clean_df, report, exclusions)``.

    Checks: duplicates, missing values, non-numeric values, usable class id,
    image dimensions, ROI coordinates (vectorised), class-folder consistency and
    existence of the image file. Rows are never dropped silently: every
    exclusion is returned as a dict with a reason.

    * duplicate rows and rows whose image file is missing, whose class id or
      dimensions are unusable are EXCLUDED (logged);
    * rows with an absent/invalid ROI are KEPT and later measured on the full
      image (``roi_ok`` is False) - this is a transformation, also reported.
    """
    data_dir = Path(data_dir)
    report = {
        "n_rows_raw": int(len(df)),
        "n_classes_raw": int(df["ClassId"].nunique()),
        "dtypes_raw": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS},
    }
    exclusions = []

    # --- duplicates -------------------------------------------------------
    dup = df.duplicated(subset=REQUIRED_COLUMNS + ["class_dir"], keep="first")
    report["n_duplicate_rows"] = int(dup.sum())
    for _, r in df[dup].iterrows():
        exclusions.append({"filename": r["Filename"], "class_dir": r["class_dir"],
                           "stage": "annotation", "reason": "duplicate annotation row"})
    work = df[~dup].copy()

    # --- numeric coercion ---------------------------------------------------
    bad_numeric = pd.Series(False, index=work.index)
    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(work[col], errors="coerce")
        bad_numeric |= coerced.isna() & work[col].notna()
        work[col] = coerced
    report["n_non_numeric_cells"] = int(bad_numeric.sum())

    # --- vectorised validity masks -----------------------------------------
    work["class_ok"] = work["ClassId"].between(0, N_CLASSES_GTSRB - 1) & \
        (work["ClassId"] == work["ClassId"].round())
    work["dims_ok"] = (work["Width"] > 0) & (work["Height"] > 0)
    x1, y1, x2, y2 = (work[c] for c in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"))
    work["roi_ok"] = (x1.notna() & y1.notna() & x2.notna() & y2.notna()
                      & (x1 >= 0) & (y1 >= 0)
                      & (x2 - x1 + 1 >= MIN_ROI_SIDE) & (y2 - y1 + 1 >= MIN_ROI_SIDE)
                      & (x2 < work["Width"]) & (y2 < work["Height"]))
    folder = work["class_dir"].str.split("/").str[-1]
    work["class_dir_ok"] = folder.str.isdigit() & (pd.to_numeric(folder, errors="coerce")
                                                    == work["ClassId"])
    work["file_ok"] = [(data_dir / d / f).is_file()
                       for d, f in zip(work["class_dir"], work["Filename"])]

    report["n_invalid_class_id"] = int((~work["class_ok"]).sum())
    report["n_invalid_dimensions"] = int((~work["dims_ok"]).sum())
    report["n_invalid_or_absent_roi"] = int((~work["roi_ok"]).sum())
    report["n_class_folder_mismatch"] = int((~work["class_dir_ok"]).sum())
    report["n_invalid_image_paths"] = int((~work["file_ok"]).sum())
    report["n_malformed_rows"] = int(bad_numeric.sum())

    # --- exclusions (reason chosen by priority) ------------------------------
    reason = pd.Series("", index=work.index)
    reason[~work["class_ok"]] = "invalid class id"
    reason[(reason == "") & ~work["class_dir_ok"]] = "class id does not match class folder"
    reason[(reason == "") & ~work["dims_ok"]] = "invalid width/height"
    excluded = reason != ""
    for idx in work.index[excluded]:
        exclusions.append({"filename": work.at[idx, "Filename"],
                           "class_dir": work.at[idx, "class_dir"],
                           "stage": "annotation", "reason": reason[idx]})
    clean = work[~excluded].copy()
    clean["ClassId"] = clean["ClassId"].astype(int)
    parts = clean["Filename"].str.replace(r"\.\w+$", "", regex=True).str.split("_")
    clean["track_id"] = parts.apply(lambda p: "_".join(p[:2]) if len(p) >= 3 else p[0])
    clean["image_path"] = clean["class_dir"] + "/" + clean["Filename"]
    # annotation rows whose image file is absent are KEPT for full-dataset counts but cannot be measured
    not_local = clean[~clean["file_ok"]]
    report["n_rows_without_local_image"] = int(len(not_local))
    for cid, n in not_local.groupby("ClassId").size().items():
        exclusions.append({"filename": f"({n} rows of class {cid})", "class_dir": "",
                           "stage": "sampling",
                           "reason": "annotation row kept for counts; image file not present locally, so not measured"})
    report["n_rows_valid"] = int(len(clean))
    report["n_classes_valid"] = int(clean["ClassId"].nunique())
    report["n_excluded_total"] = int(len(df) - len(clean))
    report["full_class_counts"] = {int(k): int(v) for k, v in
                                   clean["ClassId"].value_counts().sort_index().items()}
    return clean.reset_index(drop=True), report, exclusions


# --------------------------------------------------------------------------
# 2. OpenCV measurements
# --------------------------------------------------------------------------
def crop_roi(img, roi):
    """Crop ``img`` (H x W x 3 array) to ``roi`` = (x1, y1, x2, y2), inclusive.

    Returns ``(crop, used)``. ``used`` is False (and the full image is returned
    unchanged) when ROI values are missing, non-finite, outside the image or
    smaller than ``MIN_ROI_SIDE`` pixels.
    """
    h, w = img.shape[:2]
    try:
        x1, y1, x2, y2 = (int(v) for v in roi)
    except (TypeError, ValueError, OverflowError):
        return img, False
    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
        return img, False
    if x2 - x1 + 1 < MIN_ROI_SIDE or y2 - y1 + 1 < MIN_ROI_SIDE:
        return img, False
    return img[y1:y2 + 1, x1:x2 + 1], True


def _standard_gray(gray, size=STD_SIZE):
    """Resize a grayscale patch to ``size`` x ``size`` (area / linear interpolation)."""
    interp = cv2.INTER_AREA if max(gray.shape) > size else cv2.INTER_LINEAR
    return cv2.resize(gray, (size, size), interpolation=interp)


def _measure(bgr):
    """Return brightness, contrast, sharpness and HSV-V brightness of a BGR patch."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lap = cv2.Laplacian(_standard_gray(gray), cv2.CV_64F)
    return {
        "brightness": float(gray.mean()),          # 0-255 grey level
        "contrast": float(gray.std()),             # grey-level std, 0-~128
        "sharpness": float(lap.var()),             # Laplacian variance (proxy)
        "brightness_hsv_v": float(hsv[..., 2].mean()),
    }


def extract_image_features(path, roi=None):
    """Measure one image with OpenCV.

    Parameters
    ----------
    path : str | Path   image file.
    roi  : tuple | None (x1, y1, x2, y2) annotated sign region, inclusive.

    Returns
    -------
    dict with: width_read, height_read, roi_used, brightness, contrast,
    sharpness, brightness_hsv_v, min_side, aspect_ratio (all on the ROI crop
    when valid, else the full image) and ``*_full`` versions on the whole image.

    Raises
    ------
    ImageReadError if ``cv2.imread`` returns None (missing/corrupt file).
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)   # BGR, uint8
    if img is None or img.size == 0:
        raise ImageReadError(f"cv2.imread could not decode '{path}'")
    h, w = img.shape[:2]
    crop, used = crop_roi(img, roi) if roi is not None else (img, False)
    main = _measure(crop)
    full = _measure(img) if used else main
    ch, cw = crop.shape[:2]
    out = {"width_read": w, "height_read": h, "roi_used": used, **main,
           "min_side": min(ch, cw), "aspect_ratio": cw / ch}
    out.update({f"{k}_full": v for k, v in full.items()
                if k in ("brightness", "contrast", "sharpness")})
    return out


def load_rgb(path, roi=None, upscale=96):
    """Load an image for Matplotlib: crop to ROI (if valid), BGR->RGB, resize for display."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if roi is not None:
        img, _ = crop_roi(img, roi)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (upscale, upscale), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------
# 3. Review-priority rule (NumPy)
# --------------------------------------------------------------------------
def assign_review_priority(df, pct=10, min_triggers=2, cutoffs=None):
    """Add vectorised review flags to a feature table.

    Rule (project-specific triage setting, NOT a safety standard): an image
    triggers a condition when it falls in the lowest ``pct`` percent of the
    analysis sample for brightness, contrast, sharpness or ROI size. Images with
    >= ``min_triggers`` triggers get ``review_flag = True``; unreadable images
    are always flagged.

    Parameters
    ----------
    df : DataFrame with columns brightness, contrast, sharpness, min_side, status.
    pct : percentile (0-50) used for all four low-end cutoffs.
    min_triggers : number of triggered conditions needed to flag an image.
    cutoffs : optional dict to reuse fixed cutoffs instead of recomputing them.

    Returns
    -------
    (DataFrame, dict) - copy of ``df`` with added columns ``flag_dark``,
    ``flag_low_contrast``, ``flag_blurry``, ``flag_small``, ``n_triggers``,
    ``priority_tier`` and boolean ``review_flag``; and the cutoffs used.
    """
    if not 0 < pct <= 50:
        raise ValueError("pct must be in (0, 50]")
    out = df.copy()
    ok = (out["status"] == "ok").to_numpy()
    if cutoffs is None:
        cutoffs = {name: float(np.nanpercentile(out.loc[ok, name].to_numpy(), pct))
                   for name in FEATURE_COLUMNS}
    # vectorised masks; NaN compares False, so unreadable rows trigger nothing
    masks = {
        "flag_dark": out["brightness"].to_numpy() < cutoffs["brightness"],
        "flag_low_contrast": out["contrast"].to_numpy() < cutoffs["contrast"],
        "flag_blurry": out["sharpness"].to_numpy() < cutoffs["sharpness"],
        "flag_small": out["min_side"].to_numpy() < cutoffs["min_side"],
    }
    for name, m in masks.items():
        out[name] = m
    n_trig = np.sum(np.column_stack(list(masks.values())), axis=1)
    out["n_triggers"] = n_trig
    out["priority_tier"] = np.select(
        [~ok, n_trig >= min_triggers, n_trig >= 1],
        ["unreadable", "review first", "watch"], default="routine")
    out["review_flag"] = (~ok) | (n_trig >= min_triggers)
    out["review_flag"] = out["review_flag"].astype(bool)
    return out, cutoffs


def wilson_interval(k, n, z=1.96):
    """95 % Wilson score interval for a proportion (vectorised over arrays)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    denom = 1 + z**2 / np.maximum(n, 1)
    centre = (p + z**2 / (2 * np.maximum(n, 1))) / denom
    half = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z**2 / (4 * np.maximum(n, 1)**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


# --------------------------------------------------------------------------
# 4. The workflow class
# --------------------------------------------------------------------------
class TrafficSignAudit:
    """Organises the whole audit: load -> validate -> sample -> measure ->
    flag -> summarise -> plot. The raw dataset folder is only ever read."""

    DEFAULT_CLASSES = (4, 11, 33, 36, 40)
    SENSITIVITY = {"baseline (P10, >=2)": (10, 2),
                   "stricter (P5, >=2)": (5, 2),
                   "looser (P20, >=2)": (20, 2),
                   "any trigger (P10, >=1)": (10, 1)}

    def __init__(self, data_dir, out_dir="outputs", classes=DEFAULT_CLASSES,
                 per_class_cap=500, seed=42):
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.fig_dir = self.out_dir / "figures"
        self.proc_dir = self.out_dir / "processed"
        for d in (self.fig_dir, self.proc_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.classes = list(classes)
        self.per_class_cap = int(per_class_cap)
        self.seed = int(seed)
        # a DEMO_DATA.txt marker (written by tools/make_demo_data.py) next to or above the data
        self.demo = any((d / "DEMO_DATA.txt").exists()
                        for d in [self.data_dir, *list(self.data_dir.parents)[:3]])
        self.annotations = None      # all valid rows (full dataset)
        self.report = {}             # inspection report (full dataset)
        self.skipped = []            # every exclusion / skipped file
        self.table = None            # tidy per-image table (sample)
        self.flagged = None          # table with baseline flags
        self.cutoffs = {}
        self.sensitivity_tables = {}
        self.sensitivity_cutoffs = {}

    # -- Part 1/2 -----------------------------------------------------------
    def load_annotations(self):
        """Load + inspect the full annotation set; fills ``annotations``/``report``."""
        raw, problems = load_annotations(self.data_dir)
        for p in problems:
            self.skipped.append({"filename": p["file"], "class_dir": "", "stage": "csv",
                                 "reason": p["problem"]})
        self.annotations, self.report, excl = inspect_annotations(raw, self.data_dir)
        self.skipped.extend(excl)
        missing = [c for c in self.classes if c not in set(self.annotations["ClassId"])]
        if missing:
            raise ValueError(f"Selected classes {missing} not present in the annotations.")
        return self.annotations

    def select_sample(self):
        """Stratified sample: up to ``per_class_cap`` rows per selected class (seeded)."""
        if self.annotations is None:
            self.load_annotations()
        sub = self.annotations[self.annotations["ClassId"].isin(self.classes) &
                               self.annotations["file_ok"]]
        if sub.empty:
            raise ValueError("None of the selected classes has image files on disk - check the data folder.")
        parts = [g.sample(n=min(self.per_class_cap, len(g)), random_state=self.seed)
                 for _, g in sub.groupby("ClassId")]
        return pd.concat(parts).sort_values(["ClassId", "Filename"]).reset_index(drop=True)

    # -- Part 3 -------------------------------------------------------------
    def extract_features(self):
        """Measure every sampled image; unreadable images are kept (status) and counted."""
        sample = self.select_sample()
        rows = []
        for r in sample.to_dict("records"):
            base = {
                "image_path": r["image_path"], "filename": r["Filename"],
                "class_id": int(r["ClassId"]), "class_name": CLASS_NAMES[int(r["ClassId"])],
                "track_id": r["track_id"], "width_ann": int(r["Width"]),
                "height_ann": int(r["Height"]),
                "roi_x1": r["Roi.X1"], "roi_y1": r["Roi.Y1"],
                "roi_x2": r["Roi.X2"], "roi_y2": r["Roi.Y2"],
                "roi_ok_annotation": bool(r["roi_ok"]),
            }
            roi = (r["Roi.X1"], r["Roi.Y1"], r["Roi.X2"], r["Roi.Y2"]) if r["roi_ok"] else None
            try:
                feats = extract_image_features(self.data_dir / r["image_path"], roi)
                base.update(feats)
                base["status"] = "ok"
            except ImageReadError:
                base["status"] = "unreadable"
                # keep paths relative: message must not leak a personal absolute path
                self.skipped.append({"filename": r["image_path"], "class_dir": r["class_dir"],
                                     "stage": "image",
                                     "reason": "cv2.imread could not decode image (empty, corrupt or unsupported file)"})
            rows.append(base)
        table = pd.DataFrame(rows)
        for col in ("brightness", "contrast", "sharpness", "min_side"):
            if col not in table:
                table[col] = np.nan
        ok = table["status"] == "ok"
        table["dim_mismatch"] = ok & ((table["width_read"] != table["width_ann"]) |
                                      (table["height_read"] != table["height_ann"]))
        self.table = table
        return table

    # -- Part 4 -------------------------------------------------------------
    def assign_review_priority(self, pct=10, min_triggers=2):
        """Baseline flags and the sensitivity settings."""
        self.flagged, self.cutoffs = assign_review_priority(self.table, pct, min_triggers)
        for label, (p, m) in self.SENSITIVITY.items():
            t, c = assign_review_priority(self.table, p, m)
            self.sensitivity_tables[label] = t
            self.sensitivity_cutoffs[label] = c
        return self.flagged

    # -- Part 5 -------------------------------------------------------------
    def class_summary(self):
        """One row per selected class: counts, medians, flag rates (baseline)."""
        t = self.flagged
        full = self.annotations[self.annotations["ClassId"].isin(self.classes)] \
            .groupby("ClassId").size().rename("n_full")
        ok = t[t["status"] == "ok"]
        g = ok.groupby("class_id")
        s = pd.DataFrame({
            "n_sample": t.groupby("class_id").size(),
            "n_tracks_sample": t.groupby("class_id")["track_id"].nunique(),
            "median_min_side": g["min_side"].median(),
            "median_brightness": g["brightness"].median(),
            "median_contrast": g["contrast"].median(),
            "median_sharpness": g["sharpness"].median(),
            "n_flagged": t.groupby("class_id")["review_flag"].sum(),
        })
        s = s.join(full)
        s["flag_rate_pct"] = 100 * s["n_flagged"] / s["n_sample"]
        s.insert(0, "class_name", [CLASS_NAMES[i] for i in s.index])
        return s.sort_values("n_full", ascending=False)

    def sensitivity_summary(self):
        """Flag rate (%) by class for each cutoff setting + rank agreement."""
        rates = {}
        for label, t in self.sensitivity_tables.items():
            rates[label] = 100 * t.groupby("class_id")["review_flag"].mean()
        rates = pd.DataFrame(rates)
        base = rates.iloc[:, 0]
        agree = {c: float(rates[c].rank().corr(base.rank())) for c in rates.columns}
        top3 = {c: list(rates[c].sort_values(ascending=False).index[:3]) for c in rates.columns}
        overall = {l: int(t["review_flag"].sum()) for l, t in self.sensitivity_tables.items()}
        return rates, agree, top3, overall

    def feature_differences(self):
        """How big are between-class differences? (rank eta-squared and median gap in IQRs)."""
        ok = self.flagged[self.flagged["status"] == "ok"]
        rows = []
        for f in FEATURE_COLUMNS:
            ranks = ok[f].rank()
            grand = ranks.mean()
            ssb = ok.assign(r=ranks).groupby("class_id")["r"].agg(["count", "mean"])
            ssb = float((ssb["count"] * (ssb["mean"] - grand) ** 2).sum())
            sst = float(((ranks - grand) ** 2).sum())
            med = ok.groupby("class_id")[f].median()
            iqr = float(np.subtract(*np.percentile(ok[f], [75, 25])))
            rows.append({"feature": f, "rank_eta2": ssb / sst,
                         "median_gap_in_IQR": float((med.max() - med.min()) / iqr),
                         "class_with_min_median": CLASS_NAMES[int(med.idxmin())],
                         "class_with_max_median": CLASS_NAMES[int(med.idxmax())]})
        return pd.DataFrame(rows).set_index("feature")

    def association(self):
        """Spearman correlations between sharpness, contrast, brightness (no causal claim)."""
        ok = self.flagged[self.flagged["status"] == "ok"]
        ranks = ok[["sharpness", "contrast", "brightness", "min_side"]].rank()
        return ranks.corr()

    def roi_vs_full(self, n=200):
        """Compare ROI-crop and full-image measurements on a small seeded sample."""
        t = self.flagged if self.flagged is not None else self.table
        ok = t[(t["status"] == "ok") & t["roi_used"].fillna(False).astype(bool)]
        s = ok.sample(n=min(n, len(ok)), random_state=self.seed)
        out = {}
        for f in ("brightness", "contrast", "sharpness"):
            a, b = s[f].to_numpy(), s[f + "_full"].to_numpy()
            out[f] = {"median_crop": float(np.median(a)), "median_full": float(np.median(b)),
                      "spearman": float(pd.Series(a).rank().corr(pd.Series(b).rank())),
                      "n": int(len(s))}
        return out

    def pick_examples(self):
        """Three deterministic illustrative images (strong flag / borderline flag / clean)."""
        t = self.flagged[self.flagged["status"] == "ok"].copy()
        c = self.cutoffs
        pool = t[t["review_flag"]] if t["review_flag"].any() else t   # fall back if nothing flagged
        strong = pool.sort_values(["n_triggers", "sharpness"], ascending=[False, True]).iloc[0]
        flagged = pool[pool["n_triggers"] == pool["n_triggers"].min()].copy()
        if flagged.empty:
            flagged = pool.copy()
        # borderline: how close the triggering features are to their cutoffs
        flagged["margin"] = sum(((flagged[k] - c[k]).abs() / (abs(c[k]) + 1e-9)) *
                                flagged[f] for k, f in
                                [("brightness", "flag_dark"), ("contrast", "flag_low_contrast"),
                                 ("sharpness", "flag_blurry"), ("min_side", "flag_small")])
        borderline = flagged.sort_values("margin").iloc[0]
        clean = t[~t["review_flag"]].sort_values("n_triggers").copy()
        clean = clean[clean["n_triggers"] == clean["n_triggers"].min()] if len(clean) else t.copy()
        med = {k: clean[k].median() for k in ("brightness", "contrast", "sharpness")}
        clean["dist"] = sum(((clean[k] - m).abs() / (m + 1e-9)) for k, m in med.items())
        reference = clean.sort_values("dist").iloc[0]
        return [("Strongly flagged", strong), ("Borderline flag", borderline),
                ("Unflagged reference", reference)]

    # -- outputs ------------------------------------------------------------
    def save_tables(self):
        """Write all derived tables (never touches the raw dataset)."""
        p = self.proc_dir
        self.flagged.to_csv(p / "feature_table.csv", index=False)
        self.class_summary().to_csv(p / "class_summary.csv")
        rates, agree, top3, overall = self.sensitivity_summary()
        rates.to_csv(p / "sensitivity_flag_rates.csv")
        self.feature_differences().to_csv(p / "feature_differences.csv")
        self.association().to_csv(p / "spearman_association.csv")
        pd.DataFrame(self.skipped, columns=["filename", "class_dir", "stage", "reason"]) \
            .to_csv(p / "exclusion_log.csv", index=False)
        pd.Series(self.report["full_class_counts"], name="n_images") \
            .rename_axis("class_id").to_frame() \
            .assign(class_name=lambda d: [CLASS_NAMES[i] for i in d.index]) \
            .to_csv(p / "full_class_counts.csv")

    def save_results_json(self):
        """Machine-readable summary used by the report builder."""
        rates, agree, top3, overall = self.sensitivity_summary()
        t = self.flagged
        skipped_counts = Counter(s["reason"] for s in self.skipped)
        res = {
            "demo_data": self.demo, "seed": self.seed, "per_class_cap": self.per_class_cap,
            "classes": self.classes, "inspection": self.report,
            "n_sample": int(len(t)), "n_unreadable": int((t["status"] != "ok").sum()),
            "n_tracks_sample": int(t["track_id"].nunique()),
            "n_roi_fallback_sample": int((~t["roi_ok_annotation"]).sum()),
            "n_dim_mismatch": int(t["dim_mismatch"].sum()),
            "cutoffs": self.cutoffs, "sensitivity_cutoffs": self.sensitivity_cutoffs,
            "overall_flagged": overall, "rank_agreement": agree, "top3": top3,
            "n_flag_baseline": int(t["review_flag"].sum()),
            "tier_counts": {k: int(v) for k, v in t["priority_tier"].value_counts().items()},
            "trigger_counts": {k: int(t[k].sum()) for k in
                               ("flag_dark", "flag_low_contrast", "flag_blurry", "flag_small")},
            "skipped_counts": dict(skipped_counts),
            "roi_vs_full": self.roi_vs_full(),
            "examples": [{"label": l, "image_path": r["image_path"], "class_id": int(r["class_id"]),
                          "brightness": float(r["brightness"]), "contrast": float(r["contrast"]),
                          "sharpness": float(r["sharpness"]), "min_side": float(r["min_side"]),
                          "n_triggers": int(r["n_triggers"])} for l, r in self.pick_examples()],
        }
        with open(self.proc_dir / "results.json", "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=str)
        return res

    def plot_all(self):
        """Create and save every figure (see plotting.py)."""
        from plotting import make_all_figures      # local import keeps core light
        return make_all_figures(self)

    def run_all(self):
        """Run the complete pipeline from the raw annotations."""
        self.load_annotations()
        self.extract_features()
        self.assign_review_priority()
        self.save_tables()
        captions = self.plot_all()
        res = self.save_results_json()
        with open(self.fig_dir / "captions.json", "w", encoding="utf-8") as fh:
            json.dump(captions, fh, indent=2)
        return res
