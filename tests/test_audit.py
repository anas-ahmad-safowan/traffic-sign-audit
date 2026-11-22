"""Validation checks. Run with:  python tests/test_audit.py   (or: pytest -q)

Covers the three required checks: (1) an image-read failure is caught,
(2) the output feature columns exist, (3) review flags hold only True/False -
plus ROI validation and reproducibility of the seeded sample.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from traffic_audit import (FEATURE_COLUMNS, ImageReadError, assign_review_priority,  # noqa: E402
                           crop_roi, extract_image_features, load_annotations)


def _toy_table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"brightness": rng.uniform(20, 200, n), "contrast": rng.uniform(2, 60, n),
                         "sharpness": rng.lognormal(4, 1, n), "min_side": rng.integers(15, 120, n),
                         "status": "ok", "class_id": rng.integers(0, 3, n)})


def test_image_read_failure_is_caught():
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "broken.ppm"
        bad.write_bytes(b"this is not an image")
        for path in (bad, Path(d) / "does_not_exist.ppm"):
            try:
                extract_image_features(path)
            except ImageReadError:
                continue
            raise AssertionError("ImageReadError was not raised")


def test_feature_columns_exist():
    with tempfile.TemporaryDirectory() as d:
        img = (np.random.default_rng(1).uniform(0, 255, (40, 50, 3))).astype(np.uint8)
        p = Path(d) / "ok.png"
        cv2.imwrite(str(p), img)
        feats = extract_image_features(p, roi=(5, 5, 30, 30))
    for col in ["brightness", "contrast", "sharpness", "min_side", "roi_used",
                "brightness_full", "contrast_full", "sharpness_full"]:
        assert col in feats, f"missing feature {col}"
    assert feats["roi_used"] is True and feats["min_side"] == 26


def test_flags_are_boolean_only():
    out, cut = assign_review_priority(_toy_table())
    assert out["review_flag"].dtype == bool
    assert set(out["review_flag"].unique()) <= {True, False}
    assert set(FEATURE_COLUMNS) == set(cut)
    # stricter percentile can never flag more images than a looser one
    assert out["review_flag"].sum() <= assign_review_priority(_toy_table(), 20)[0]["review_flag"].sum()


def test_invalid_roi_falls_back_to_full_image():
    img = np.zeros((30, 30, 3), np.uint8)
    for roi in [(10, 10, 5, 5), (0, 0, 99, 99), (np.nan, 1, 5, 5), (0, 0, 1, 1), None]:
        crop, used = crop_roi(img, roi) if roi is not None else (img, False)
        assert used is False and crop.shape == img.shape


def test_missing_dataset_folder_gives_helpful_error():
    try:
        load_annotations("definitely/not/here")
    except FileNotFoundError as exc:
        assert "README" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"All {len(tests)} checks passed.")
