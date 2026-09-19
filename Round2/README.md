# DATA VORTEX A'26 — Round 2: "Rebuilding the Social Engine"
## NLP Sentiment Classification — Project README

---

## Repository Structure

```
Round2/
├── data/
│   ├── Labeled_Social_NLP_Training_Data.csv   # Raw dataset (byte-for-byte untouched)
│   ├── processed/
│   │   ├── train.csv    (6,279 rows)
│   │   ├── val.csv      (1,375 rows)
│   │   └── test.csv     (1,346 rows)
│   └── raw/             # Symlink / copy of original CSV
├── src/
│   ├── utils.py                    # Shared cleaning + label encoding (single source of truth)
│   ├── 01_audit.py                 # Phase 1: Data audit
│   ├── 02_preprocess.py            # Phase 2: Cleaning + group-aware splits
│   ├── 03_eda.py                   # Phase 3: EDA figures
│   ├── 04_baselines.py             # Phase 4: Baseline models
│   ├── 05_experiments.py           # Phase 5: Model experiments (fast path)
│   ├── 06_evaluate.py              # Phase 6: Final test evaluation (run ONCE)
│   ├── 07_error_analysis.py        # Phase 7: Quantified error analysis
│   └── 08_inference_pipeline.py    # Phase 8: Production inference interface
├── models/
│   └── final_sentiment_model.pkl   # Trained sklearn Pipeline (tfidf → LinearSVC)
├── artifacts/
│   ├── label_encoders.json         # Sentiment + topic label mappings
│   └── final_model_config.json     # Model config + val metrics
├── reports/
│   ├── audit_report.md             # Phase 1 output
│   ├── cleaning_log.md             # Phase 2 output
│   ├── eda_report.md               # Phase 3 output
│   ├── baseline_results.md/csv     # Phase 4 output
│   ├── model_selection.md          # Phase 5 output
│   ├── experiment_log.csv          # Phase 5 output
│   ├── evaluation_report.md        # Phase 6 output
│   ├── test_predictions.csv        # Phase 6 output
│   ├── error_analysis.md           # Phase 7 output
│   ├── reviewed_errors.csv         # Phase 7 output (20 hand-reviewed errors)
│   └── figures/                    # All generated figures (PNG)
├── notebooks/
│   └── Round2_NLP_Analysis.ipynb   # Full pipeline walkthrough notebook
├── docs/
│   └── Round2_Technical_Report.md  # Technical Report (Markdown → PDF)
├── submission/                      # Final submission package
├── requirements.txt                 # Pinned dependencies
└── README.md                        # This file
```

---

## Final Model

**`char(3,5) + LinearSVC C=1.0`** — TF-IDF character (3,5)-gram vectorizer + LinearSVC  
- **Test Macro-F1: 60.93%** (test.csv, evaluated once at 2026-09-19T01:05:10Z)  
- **Test Accuracy: 60.92%**  
- Retrained on train + val combined (7,654 rows)  
- Trains in **3.7 seconds** on CPU — fully reproducible

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Reproduce the full pipeline (from Round2/ root)

```bash
python src/01_audit.py          # Data audit
python src/02_preprocess.py     # Cleaning + splits
python src/03_eda.py            # EDA figures
python src/04_baselines.py      # Baseline models
python src/05_experiments.py    # Model experiments + selection
python src/06_evaluate.py       # Test evaluation (opens test.csv ONCE)
python src/07_error_analysis.py # Error analysis
python src/08_inference_pipeline.py  # Inference demo + verification
```

All scripts are deterministic with `RANDOM_SEED = 42`.

### 3. Use the inference pipeline directly

```python
import sys
sys.path.insert(0, 'src')
from inference_pipeline import load_model, predict

bundle  = load_model()
results = predict(["This is amazing!", "Terrible day :(", "Event at 3pm downtown"], bundle)
for r in results:
    print(r['label'], '|', r['clean_text'])
```

The pipeline automatically applies the same cleaning steps used during training (Unicode decode, HTML entity decode, @mention normalisation, URL replacement) via `src/utils.py::clean_text()`.

### 4. Run the notebook

```bash
jupyter notebook notebooks/Round2_NLP_Analysis.ipynb
```

Run all cells top-to-bottom. The notebook imports from `src/` — no code duplication.

---

## Technical Integrity Notes

- **One-shot evaluation**: `test.csv` was opened exactly once, in Phase 6, after all model selection decisions were locked. The test macro-F1 (60.93%) is a genuine held-out score.
- **No leakage**: Vectorizer fitted on `train.csv` only. Group-aware split ensures the 1,100 duplicate posts do not span train/val/test.
- **Cleaning parity**: `src/utils.py::clean_text()` is the single implementation of all preprocessing — identical at training time (Phase 2), evaluation time (Phase 6), and inference time (Phase 8). Verified by fresh-process test.
- **All metrics reproducible**: Every number in the Technical Report traces to a script-generated report file. Re-running the pipeline from scratch produces identical outputs.

---

## Environment

| Package | Version |
|---|---|
| Python | 3.10.11 |
| scikit-learn | 1.4.2 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| matplotlib | 3.8.2 |

---

## Submission Contents

See `submission/` directory for:
- `Round2_Technical_Report.md` (→ PDF)
- `Round2_NLP_Analysis.ipynb`
- `final_sentiment_model.pkl`
- `evaluation_report.md`
- `README.md`
