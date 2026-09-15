# STATE.md — DATA VORTEX submission status

Last updated: after full pipeline execution and notebook run (this session).

## Status: submission-ready

All steps below were actually executed (not just written) and their outputs verified.

| Step | Script | Status | Output |
|---|---|---|---|
| Audit | `src/01_audit_raw_data.py` | ✅ ran, verified | `reports/audit_report.md` |
| Clean | `src/02_clean_data.py` | ✅ ran, verified | `data/cleaned/*.csv`, `reports/cleaning_log.md` |
| Validate | `src/03_validate_cleaned_data.py` | ✅ ran, **15/15 checks passed** | `reports/validation_report.md` |
| EDA | `src/04_eda_and_visuals.py` | ✅ ran, verified | `reports/figures/*.png` (8 charts) |
| Insights | `src/05_insights.py` | ✅ ran, verified | `reports/insights.md` |
| Notebook | `notebooks/DataVortex_Analysis.ipynb` | ✅ executed end-to-end via `jupyter nbconvert --execute`, 0 errors across 22 cells | inline outputs + figures |
| Raw file integrity | — | ✅ checksums of `data/raw/*.csv` verified identical to the original uploads at every stage | — |

## Known issue found and fixed during this session

Early in cleaning, assigning `None` into a pandas `StringDtype` column silently
converted it to the literal string `"nan"` instead of a real missing value, corrupting
~15% of `text_content`. This was caught by re-loading the cleaned CSV and checking for
literal `"nan"` strings (not just `.isna()`, which would not have caught it). Fixed by
switching `utils.py` to load raw data with `dtype=object` instead of `dtype=str`, then
re-ran the full pipeline from scratch and re-verified. See `reports/cleaning_log.md`
and the audit trail above — this is why the "correct" pass here matters more than
any single check in isolation: **every claim in `insights.md` was validated against
the corrected cleaned output, not the first (buggy) run.**

## What's NOT included (explicit scope decisions, not oversights)

- No imputation of missing `likes`/`platform`/`text_content` — left as NaN by design.
- No trained sentiment/NLP model — insight #4 uses a fixed keyword list to demonstrate
  a data-quality problem, not to ship a sentiment score.
- No brand/product-level breakdown in the final cleaned schema — explored during EDA
  but the underlying text template wasn't parsed robustly enough to trust as a
  submission-grade feature.

## If continuing this project further

- Backfill missing `platform` values using a platform-ID lookup table, if one becomes
  available — insight #5 suggests these are recoverable, healthy posts.
- Build a small labeled sample to validate whether the "sign-flip" theory for negative
  `likes` holds up against ground truth, if the original source system is available.
- Extend `05_insights.py` with per-language or per-location engagement breakdowns using
  `users_cleaned.csv` — not done here to keep scope tight and every claim verifiable.
