# Phase 4 — Detect Shifts/Spikes & Analyze Topics/Entities

## Objective
Statistically detect genuine sentiment shifts and engagement spikes from `hourly_timeline.csv` (thresholds set BEFORE looking at results), then analyze what topics/entities changed around each flagged event.

## Context / Input
- `Round3/data/processed/hourly_timeline.csv` and `reddit_enriched.csv` from Phase 3.
- Rulebook requires ≥2 genuine sentiment shifts and ≥1 genuine engagement spike — these must emerge from the data, not be forced.

## Exact Tasks
1. Write `Round3/src/04_detect_events.py`:
   - **MINIMUM SAMPLE GUARD**:
     - Check the count of non-zero hourly buckets in `hourly_timeline.csv`.
     - If the number of non-zero hourly buckets is too sparse (e.g., < 6 non-zero hours), fall back to 3-hour buckets. If still insufficient for statistical detection, fall back to a descriptive-only mode and document the rationale in `Round3/reports/phase4_findings_notes.md`.
   - **Before looking at the data**, fix thresholds and hard floors in code comments:
     - Sentiment shift = |Δ(pct_positive − pct_negative)| between adjacent hours ≥ 15 percentage points (adjust once, with a documented reason, if the data genuinely never crosses it — do not repeatedly lower it until something appears).
       - **Hard floor**: 10 percentage points (must NOT be crossed downward under any circumstances).
     - Engagement spike = `post_count` (or `total_score`) in an hour ≥ mean + 2×standard deviation of the full window.
       - **Hard floor**: mean + 1σ (must NOT be crossed downward under any circumstances).
   - Scan `hourly_timeline.csv` and output every hour that crosses either threshold into `Round3/data/processed/flagged_events.csv` with columns: `timestamp, event_type (shift/spike), metric_value, delta_or_zscore`.
2. If fewer than 2 shifts or 0 spikes are found:
   - Try a 2-hour or 3-hour bucket instead of hourly (documented, not silently swapped).
   - If still insufficient, clearly flag this in `Round3/reports/phase4_findings_notes.md` as a genuine data limitation and report the strongest available shifts/spikes honestly, rather than lowering thresholds below the hard floors until something trivial qualifies.
3. Write `Round3/src/05_topic_entity_analysis.py`:
   - For each flagged event's hour window, take posts from that window vs. the immediately preceding window of equal length.
   - Compute top TF-IDF terms (unigrams/bigrams) for each window using `sklearn.feature_extraction.text.TfidfVectorizer` (reuse Round 2's vectorizer approach/settings for consistency, do not import Round 2's fitted vectorizer object — fit fresh on this new text).
   - Run a lightweight NER pass (spaCy `en_core_web_sm`, install if missing) on the same windows to extract named entities (people, orgs, products).
   - Save a before/after term + entity table per flagged event to `Round3/data/processed/topic_entity_by_event.csv`.

## Files to Inspect / Modify
Read: `Round3/data/processed/hourly_timeline.csv`, `reddit_enriched.csv`.
Create: `Round3/src/04_detect_events.py`, `Round3/src/05_topic_entity_analysis.py`, `Round3/data/processed/flagged_events.csv`, `Round3/data/processed/topic_entity_by_event.csv`, `Round3/reports/phase4_findings_notes.md`.

## Expected Outputs
- `flagged_events.csv`: every genuinely detected shift/spike with its numeric evidence.
- `topic_entity_by_event.csv`: top terms/entities before vs. after each flagged event.
- `phase4_findings_notes.md`: honest note on whether ≥2 shifts / ≥1 spike were found, and any threshold adjustments made (with reasons).

## Validation / Checks
- [ ] Thresholds are documented in code BEFORE results were computed (no evidence of post-hoc tuning to hit the required count).
- [ ] `flagged_events.csv` contains ≥2 shift rows and ≥1 spike row, OR `phase4_findings_notes.md` clearly explains the shortfall and what was reported instead.
- [ ] Topic/entity tables show genuinely different top terms before vs. after (not identical lists).
- [ ] No fabricated rows anywhere in these files — every value traces back to `hourly_timeline.csv` / `reddit_enriched.csv`.

## Completion Criteria
`flagged_events.csv` and `topic_entity_by_event.csv` exist, are threshold-based (not cherry-picked), and findings notes honestly state whether requirements were met.

## Must NOT Do
- Do not lower thresholds repeatedly or cross below the hard floors (10pp / mean+1σ) just to manufacture the required number of shifts/spikes.
- Do not hand-pick "interesting" hours outside the threshold rule.
- Do not fabricate topic/entity terms that don't appear in the actual text.
- Do not skip documenting a shortfall if the data doesn't support the full requirement.
