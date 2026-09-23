# PHASE 6 — FINAL REPORT + VALIDATION

Execute **only Phase 6**.

## Objective

Assemble the Round 3 analysis into the final notebook, report, figures, and validation files.

Read all outputs from Phases 1–5, especially:

* `Round3/data/raw/bluesky_raw_final.csv`
* `Round3/data/processed/bluesky_enriched.csv`
* `Round3/data/processed/hourly_timeline.csv`
* `Round3/data/processed/flagged_events.csv`
* `Round3/data/processed/topic_entity_by_event.csv`
* `Round3/data/processed/trigger_candidates.csv`
* `Round3/reports/collection_log.md`
* `Round3/reports/phase3_sanity_check.md`
* `Round3/reports/phase4_findings_notes.md`
* `Round3/reports/trigger_table.md`

## 1. Figures

Create:

`Round3/src/07_build_report_figures.py`

Generate:

**Fig 1:** `Round3/reports/figures/fig1_sentiment_engagement_timeline.png`

* sentiment percentages over time
* activity/post-count bars
* clearly mark genuine flagged shifts/spikes
* use actual observed timeline only

**Fig 2:** `Round3/reports/figures/fig2_topic_entity_before_after.png`

* use the strongest genuinely flagged event based on Phase 4 evidence
* show before vs event topics/entities

Optional Fig 3 only if an existing Round 2 confusion matrix is readily available. **Do not retrain the model.**

## 2. Notebook

Create:

`Round3/notebooks/Round3_Analysis.ipynb`

Provide a clean top-to-bottom walkthrough covering:

1. configuration/data source
2. collection
3. raw data inspection
4. Round 2 model loading
5. sentiment prediction
6. timeline
7. event detection
8. topic/entity analysis
9. trigger investigation
10. figures
11. final validation

Import/use the Phase 2–5 scripts rather than duplicating their logic.

Execute top-to-bottom where possible. If external API access prevents reproduction, document it clearly.

## 3. Final Report

Create:

`Round3/reports/Round3_Analytical_Report.md`

Convert it to a PDF.

Include exactly these mandatory sections:

### 1. Data Collection Method

Bluesky public API, search terms, fields collected, deduplication, and collection procedure.

### 2. Time Window

Report **actual** collection timestamps, source `created_at` range, collector `collected_at` range, number of runs, gaps, and collection mode.

### 3. Sentiment Analysis

Explain the frozen Round 2 model, three sentiment classes, overall distribution, timeline behavior, and genuine sentiment shifts.

### 4. Activity Analysis

Explain post volume, engagement/activity metric, detected spikes, and threshold.

### 5. Topic/Entity Analysis

Explain topics/entities around flagged events and the before-vs-event comparison.

### 6. Trigger Explanations

Include the Phase 5 trigger findings for every flagged event, with Strong/Weak/Unexplained evidence classification.

Use only actual results. If a required shift/spike was not found, state the shortfall honestly.

## 4. Model Limitations

Briefly state:

* Round 2 model was reused without retraining.
* Known limitations, including weaker handling of negation.
* Phase 3 manual sanity check was performed.
* No new accuracy/F1 is claimed for Round 3.

## 5. Final Validation

Create:

`Round3/reports/final_validation.md`

Verify:

* genuine self-collected Bluesky dataset
* real source/collection timestamps
* no fabricated data
* ≥2 sentiment shifts **or documented shortfall**
* ≥1 engagement spike **or documented shortfall**
* topics/entities tied to genuine flagged events
* trigger investigation completed for every flagged event
* Round 2 model reused unchanged
* raw data preserved
* all 6 mandatory report sections present
* notebook runs successfully or limitations are documented
* final submission files are present

## Final Output

Ensure the submission package contains:

* `bluesky_raw_final.csv`
* `01_collect_bluesky.py`
* `Round3_Analysis.ipynb`
* `Round3_Analytical_Report.pdf`
* `Round3_Analytical_Report.md`
* required figures
* `final_validation.md`

**Do not submit or claim that any external form has been submitted. Do not fabricate missing results.**
