# Round 3 — Final Validation Checklist
**Date validated:** 2026-09-22 (post final cleanup)
**All numbers verified from actual CSV outputs. Consistency check: ALL CLEAR (23/23).**

## A. Final Dataset / Results

| Item | Value |
|---|---|
| Original raw posts | 1,320 |
| Off-topic excluded (Fix 1) | 733 |
| **Posts used for analysis** | **587** |
| Future-dated anomalies | 6 (cause unknown; preserved unaltered) |
| Sentiment — Negative | 166 (28.3%) |
| Sentiment — Neutral | 213 (36.3%) |
| Sentiment — Positive | 208 (35.4%) |
| Hourly buckets | 22 |
| Source timestamp range | 2026-09-21T21:25:34Z — 2026-09-22T18:34:20.783992Z |
| Collection mode | Single-shot |
| Sentiment shifts detected | **10** (≥15pp polarity swing) |
| Engagement spikes detected | **1** (13:00 UTC; engagement=151; z=2.884; threshold=115.21) |
| Total flagged events | **11** |
| Events with entities | 11/11 |
| Trigger: Strong / Weak / Unexplained | 0 / 1 / 10 |

## B. Mandatory Requirements Checklist

- [x] 587 genuine Bluesky posts (no fabrication)
- [x] Original raw dataset preserved unchanged (`bluesky_raw_final.csv`, 1320 rows)
- [x] Source timestamps preserved (no clamping of `created_at`)
- [x] `future_dated_anomaly` flag present; 6 anomalies documented
- [x] Single-shot collection accurately described
- [x] Round 2 model unchanged — no retraining/modification
- [x] Relevance filtering documented (Fix 1): `exclusion_log.csv` with 733 rows + reasons
- [x] Engagement metric corrected (Fix 2): `likes + reposts + replies + quotes`
- [x] `<url>` artifact removed from TF-IDF topics (Fix 3)
- [x] Entity extraction (Fix 3): curated regex, 11/11 events have entities
- [x] 10 genuine sentiment shifts at ≥15pp threshold
- [x] 1 genuine engagement spike at mean+2σ on `total_engagement`
- [x] All 11 events investigated for external triggers
- [x] Trigger classifications: 0 Strong / 1 Weak / 10 Unexplained
- [x] No fabricated events or evidence
- [x] All 6 mandatory report sections present in `Round3_Analytical_Report.md`
- [x] Figures generated: `figure1_timeline.png`, `figure2_topics.png`
- [x] Notebook updated: `Round3_Analysis.ipynb` (corrected numbers, all fixes documented)
- [x] Notebook executed: `Round3_Analysis_executed.ipynb`
- [x] Notebook HTML export: `Round3_Analysis_notebook.html` (624,519 bytes)
- [x] Trigger table regenerated for 11 events: `trigger_table.md` (0 Strong / 1 Weak / 10 Unexplained)
- [x] Analytical report PDF: `Round3_Analytical_Report.pdf` (634,255 bytes)
- [x] `.env` excluded from git tracking via root `.gitignore`
- [x] Credentials never printed

## C. Known Remaining Issues

None. All three cleanup items from the previous validation are resolved:
- ~~Trigger table reflected 8-event set~~ → **Fixed**: rebuilt for 11 events
- ~~Notebook cells unexecuted~~ → **Fixed**: executed + HTML exported
- ~~Figure 2 had no entity data~~ → **Accepted limitation**: entity data in CSV; figure shows topics only

## D. Submission Files

| File | Path | Status |
|---|---|---|
| Analytical Report (MD) | `Round3/reports/Round3_Analytical_Report.md` | ✅ |
| Analytical Report (PDF) | `Round3/reports/Round3_Analytical_Report.pdf` | ✅ 634 KB |
| Figure 1 — Timeline | `Round3/reports/figure1_timeline.png` | ✅ |
| Figure 2 — Topics | `Round3/reports/figure2_topics.png` | ✅ |
| Notebook (source) | `Round3/notebooks/Round3_Analysis.ipynb` | ✅ |
| Notebook (executed) | `Round3/notebooks/Round3_Analysis_executed.ipynb` | ✅ |
| Notebook (HTML) | `Round3/reports/Round3_Analysis_notebook.html` | ✅ 624 KB |
| Trigger table | `Round3/reports/trigger_table.md` | ✅ 11 events |
| Trigger search log | `Round3/reports/trigger_search_log.md` | ✅ |
| Raw posts (original) | `Round3/data/raw/bluesky_raw_final.csv` | ✅ 1,320 rows |
| Relevant posts | `Round3/data/raw/bluesky_relevant.csv` | ✅ 587 rows |
| Exclusion log | `Round3/data/raw/exclusion_log.csv` | ✅ 733 rows |
| Enriched posts | `Round3/data/processed/bluesky_enriched.csv` | ✅ 587 rows |
| Timeline | `Round3/data/processed/hourly_timeline.csv` | ✅ 22 buckets |
| Flagged events | `Round3/data/processed/flagged_events.csv` | ✅ 11 events |
| Topic/entity data | `Round3/data/processed/topic_entity_by_event.csv` | ✅ 11 events |
| Trigger candidates | `Round3/data/processed/trigger_candidates.csv` | ✅ 79 candidates |
| Collection log | `Round3/reports/collection_log.md` | ✅ |
| `.gitignore` (root) | `.gitignore` | ✅ excludes `.env` |
| Credentials | `Round3/.env` | 🔒 not tracked |

## E. What You Must Do Manually Before Submitting

1. **Open `Round3_Analytical_Report.pdf`** and confirm figures render + all 6 sections are readable.
2. **Open `Round3_Analysis_notebook.html`** and confirm all cell outputs are visible.
3. **Do not commit `Round3/.env`** — verify `git status` shows it as untracked before any `git push`.
