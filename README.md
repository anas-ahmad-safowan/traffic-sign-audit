# Traffic Sign Visibility Audit (GTSRB)

An individual project that audits real traffic-sign photographs and answers one question:
**which images should a human review team look at first, and what evidence supports that priority?**
It measures brightness, contrast, sharpness and size with OpenCV, builds a transparent percentile-based
review-priority rule with NumPy, compares sign classes with pandas, and plots the evidence with Matplotlib.
**No deep-learning model is trained.** A flag is a triage aid, not a safety verdict.

## Dataset and citation
German Traffic Sign Recognition Benchmark (GTSRB), Real-Time Computer Vision Group, Institut für
Neuroinformatik, Ruhr-Universität Bochum (Stallkamp, Schlipsing, Salmen & Igel).
- Official page: https://benchmark.ini.rub.de/gtsrb_dataset.html
- Archival copy: https://zenodo.org/records/13741936

The dataset is **not** included in this repository (check the source terms before redistributing it).

### Getting the data
Download GTSRB (links above) and place it in `data/` in **either** layout (auto-detected):
- Kaggle-style: `data/Train.csv` plus `data/Train/<classId>/*.png` (the layout used for the submitted run), or
- Official archive: `data/GTSRB/Final_Training/Images/<00000..00042>/GT-000NN.csv` + `.ppm` images
  (then run with `--data-dir data/GTSRB/Final_Training/Images`).

The submitted run used the full `Train.csv` (39,209 rows) but only 22 locally available images of classes
4, 11, 33, 36 and 40, so the image measurements are a small illustration; add more images and re-run for stronger results.
The raw data is only ever read; every derived file goes to `outputs/`.

## Installation and run
**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_audit.py --data-dir data   # recreates all tables + figures from the raw annotations
python src/build_report.py          # optional: builds report/Traffic_Sign_Audit_Report.docx
```
**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate.bat   # (PowerShell: venv\Scripts\Activate.ps1)
pip install -r requirements.txt
python run_audit.py --data-dir data
python src\build_report.py
```
For the notebook also `pip install jupyter`, then open
`notebooks/traffic_sign_audit.ipynb` and use *Restart & Run All*.

Useful options: `python run_audit.py --classes 4 11 33 36 40 --cap 500 --seed 42`.
Validation checks: `python tests/test_audit.py`.

### Try it without the dataset (smoke test only)
```bash
python tools/make_demo_data.py --out demo/data
python run_audit.py --data-dir demo/data/Final_Training/Images --out-dir demo/outputs
```
This writes **synthetic drawn shapes** in the GTSRB layout (with a few deliberate faults to exercise the error
handling). Outputs from it are labelled DEMO and say nothing about GTSRB.

## Repository structure
```
traffic-sign-audit/
├── README.md  requirements.txt  run_audit.py      # entry point (main script)
├── data/                     # place the downloaded GTSRB here (instructions only in git)
├── notebooks/traffic_sign_audit.ipynb              # step-by-step analysis notebook
├── src/
│   ├── traffic_audit.py      # functions + TrafficSignAudit class
│   ├── plotting.py           # all Matplotlib figures
│   └── build_report.py       # report builder (.docx)
├── tests/test_audit.py       # validation checks
├── tools/make_demo_data.py   # synthetic smoke-test data
├── outputs/
│   ├── figures/              # 8 PNG figures + captions.json
│   └── processed/            # feature_table.csv, class_summary.csv, sensitivity, exclusion_log.csv, results.json
├── report/Traffic_Sign_Audit_Report
└── demo_example/             # example outputs from the SYNTHETIC data (not GTSRB results)
```

## Design (how the brief maps to the code)
| Requirement | Where |
|---|---|
| Reusable functions | `load_annotations`, `inspect_annotations`, `crop_roi`, `extract_image_features`, `assign_review_priority` |
| Class-based workflow | `TrafficSignAudit` (load, validate, sample, extract, flag, summarise, plot, run_all) |
| Validation of data / ROI / paths | `inspect_annotations`; every exclusion logged in `outputs/processed/exclusion_log.csv` |
| OpenCV features | brightness (gray mean), contrast (gray std), sharpness (Laplacian variance on a 48x48 patch), min_side; ROI vs full-image comparison |
| NumPy | `np.nanpercentile` cutoffs, boolean masks, `np.select` tiers, `np.sum`/`column_stack`, Wilson interval |
| pandas | groupby summaries, joins of measurements to labels, rank-based effect sizes, Spearman association |
| Matplotlib only | `src/plotting.py` (no Seaborn) |

Scope: all annotation rows are used for full-dataset class counts; image measurements use a seeded stratified
sample (<= 500 per selected class, seed 42). The two are reported separately.

