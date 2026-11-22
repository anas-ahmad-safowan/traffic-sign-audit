"""Recreate every table and figure from the raw annotations.

    python run_audit.py                       # uses data/GTSRB/Final_Training/Images
    python run_audit.py --data-dir demo/data/Final_Training/Images --out-dir demo/outputs
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from traffic_audit import TrafficSignAudit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Traffic Sign Visibility Audit (GTSRB)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--classes", type=int, nargs="+", default=[4, 11, 33, 36, 40])
    ap.add_argument("--cap", type=int, default=500, help="max images per class for OpenCV measures")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    try:
        audit = TrafficSignAudit(a.data_dir, a.out_dir, a.classes, a.cap, a.seed)
        res = audit.run_all()
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"Full dataset: {res['inspection']['n_rows_valid']} valid rows, "
          f"{res['inspection']['n_classes_valid']} classes")
    print(f"Sample: {res['n_sample']} images; unreadable: {res['n_unreadable']}; "
          f"flagged: {res['n_flag_baseline']}")
    print(f"Skipped/excluded records: {res['skipped_counts']}")
    print(f"Outputs written to {a.out_dir}/")


if __name__ == "__main__":
    main()
