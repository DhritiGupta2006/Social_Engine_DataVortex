# PHASE 3 — SENTIMENT ANALYSIS + TIMELINE

Execute **only Phase 3**. Do not start Phase 4.

## Objective

Apply the **existing frozen Round 2 sentiment model** to the collected Bluesky data and build an hourly activity/sentiment timeline.

## 1. Sentiment Processing

Create/update:

`Round3/src/02_process_and_predict.py`

Input:
`Round3/data/raw/bluesky_raw_final.csv`

* Preserve all raw columns.
* Apply the **exact Round 2 `clean_text()`** to the Bluesky `text`.
* Reuse the existing Round 2 inference/model-loading code from `Round3/src/round2_reuse/`.
* Do **not retrain, refit, or modify** the Round 2 model/vectorizer.
* Generate `sentiment_label`: `Negative`, `Neutral`, or `Positive`.
* Save:

`Round3/data/processed/bluesky_enriched.csv`

Row count must match the raw dataset. Report any prediction failures; never silently drop rows.

## 2. Build Timeline

Create/update:

`Round3/src/03_build_timeline.py`

Use **`created_at`**, not `collected_at`, as the event/time-series timestamp.

Before processing, print the actual:

* minimum `created_at`
* maximum `created_at`
* number of posts

Create hourly UTC buckets covering the **actual observed source-time range**.

For each hour calculate:

* `hour`
* `post_count`
* `total_engagement`
* `avg_engagement`
* `pct_negative`
* `pct_neutral`
* `pct_positive`

Calculate engagement only from real fields:

`like_count + repost_count + reply_count + quote_count`

Include zero-post hours with zero metrics.

Save:

`Round3/data/processed/hourly_timeline.csv`

## 3. Manual Sanity Check

Create:

`Round3/reports/phase3_sanity_check.md`

Inspect 15–20 random predictions and record:

* original text
* cleaned text
* predicted sentiment
* brief correctness observation

Pay particular attention to negation, sarcasm, short/ambiguous posts, and mixed sentiment. Do not change the model based on this check.

## Constraints

* Round 2 model remains frozen.
* Do not invent sentiment, timestamps, engagement, or missing data.
* Do not assume the collection covered 20–22 September.
* Use actual timestamps from the collected dataset.
* Do not modify Round 2 files.

## Final Check

Verify:

* enriched CSV exists
* row counts match
* all sentiment labels are valid
* timeline covers the actual source-time range
* engagement uses only real fields
* sanity check is saved

Then **stop after Phase 3** and report the files created/changed and the actual data/time range processed.
