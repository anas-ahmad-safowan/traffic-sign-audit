"""Create a SYNTHETIC stand-in dataset in the GTSRB training-archive layout.

Purpose: let anyone smoke-test the pipeline without downloading GTSRB. The
images are drawn shapes, NOT real traffic-sign photographs; results computed
from them say nothing about GTSRB. A ``DEMO_DATA.txt`` marker file is written so
the audit and the report builder label every output as demo output.

Usage:  python tools/make_demo_data.py --out demo/data
Layout: <out>/Final_Training/Images/00000/{GT-00000.csv, 00000_00000.ppm, ...}
It also injects a few deliberate faults (zero-byte image, missing file, malformed
and invalid ROI rows, duplicate row) to exercise the error handling.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# images per class in the real training set (used only to mimic class imbalance)
REAL_COUNTS = [210, 2220, 2250, 1410, 1980, 1860, 420, 1440, 1410, 1470, 2010, 1320, 2100,
               2160, 780, 630, 420, 1110, 1200, 210, 360, 330, 390, 510, 270, 1500, 600, 240,
               540, 270, 450, 780, 240, 689, 420, 1200, 390, 210, 2070, 300, 360, 240, 240]
HEADER = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"


def draw_sign(cls, size, rng, dark, blur, flat):
    """Draw a simple coloured sign on a noisy road-like background."""
    margin = int(rng.integers(1, max(2, size // 8) + 1))
    w = size + 2 * margin
    h = int(size * rng.uniform(.9, 1.1)) + 2 * margin
    bg = rng.integers(60, 160, 3).astype(np.float32)
    img = np.ones((h, w, 3), np.float32) * bg + rng.normal(0, 10, (h, w, 3))
    cx, cy, r = w // 2, h // 2, max(3, min(w, h) // 2 - margin)
    kind = cls % 3
    rim = (40, 40, 200) if kind != 2 else (190, 90, 30)           # BGR
    if kind == 1:                                                  # triangle
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]])
        cv2.fillPoly(img, [pts.astype(np.int32)], rim)
        inner = ((pts - [cx, cy]) * .6 + [cx, cy + r * .12]).astype(np.int32)
        cv2.fillPoly(img, [inner], (235, 235, 235))
    else:
        cv2.circle(img, (cx, cy), r, rim, -1)
        cv2.circle(img, (cx, cy), int(r * .72), (235, 235, 235) if kind == 0 else (200, 120, 50), -1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.putText(img, str(cls % 10), (cx - r // 3, cy + r // 3), cv2.FONT_HERSHEY_SIMPLEX,
                max(.25, r / 28), (20, 20, 20), 1, cv2.LINE_AA)
    img = img.astype(np.float32) * dark                            # exposure
    img = img.mean() + (img - img.mean()) * flat                   # contrast loss
    img = np.clip(img, 0, 255).astype(np.uint8)
    if blur > 0.1:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img, (margin, margin, w - margin - 1, h - margin - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo/data")
    ap.add_argument("--scale", type=float, default=1 / 4, help="fraction of real class counts")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out)
    base = root / "Final_Training" / "Images"
    root.mkdir(parents=True, exist_ok=True)
    (root / "DEMO_DATA.txt").write_text(
        "SYNTHETIC DEMO DATA - drawn shapes, not GTSRB photographs. Do not report results from it.\n")
    for cls, real_n in enumerate(REAL_COUNTS):
        n = max(60, int(real_n * a.scale))
        d = base / f"{cls:05d}"
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        # class-level condition (some classes are darker / softer / smaller on average)
        c_dark = rng.uniform(.55, 1.15)
        c_blur = rng.uniform(0, 1.2)
        c_size = rng.uniform(24, 70)
        for i in range(n):
            if i % 30 == 0:                               # new "track" (one physical sign)
                t_dark = c_dark * rng.uniform(.75, 1.2)
                t_blur = max(0, c_blur + rng.normal(0, .5))
                t_flat = rng.uniform(.5, 1.0)
                t_size = c_size * rng.uniform(.7, 1.3)
            track = i // 30
            s = int(np.clip(t_size * (0.6 + 0.9 * (i % 30) / 30), 15, 220))
            img, roi = draw_sign(cls, s, rng, t_dark * rng.uniform(.95, 1.05),
                                 t_blur + rng.uniform(0, .3), t_flat)
            name = f"{track:05d}_{i % 30:05d}.ppm"
            h, w = img.shape[:2]
            cv2.imwrite(str(d / name), img)
            rows.append([name, w, h, *roi, cls])
        # --- deliberate faults in class 1 and 14 --------------------------------
        if cls == 1:
            (d / rows[3][0]).write_bytes(b"")                       # zero-byte image
            (d / rows[7][0]).write_bytes(b"not an image")           # corrupt image
            (d / rows[10][0]).unlink()                              # missing file
            rows[12][3], rows[12][5] = rows[12][5], rows[12][3]     # invalid ROI (x1 > x2)
            rows[15][4] = "??"                                      # malformed numeric
            rows.append(rows[20][:])                                # duplicate row
        if cls == 14:
            rows[5][3:7] = [0, 0, 0, 0]                             # degenerate ROI
        lines = [HEADER] + [";".join(map(str, r)) for r in rows]
        (d / f"GT-{cls:05d}.csv").write_text("\n".join(lines) + "\n")
    print(f"Synthetic demo data written to {base}")


if __name__ == "__main__":
    main()
