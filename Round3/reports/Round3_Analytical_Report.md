# DATA VORTEX A'26 — Round 3 Analytical Report
**Topic: Public Reaction to a Viral Product Launch**
**Data Source: Bluesky Social**
**Analysis Date: 2026-09-22**

---

## 1. Data Collection Method

Data was collected from **Bluesky Social** (`bsky.social`) as the sole source for Round 3. Reddit was abandoned prior to this phase because Reddit API access required approval that was not granted. No fabricated, synthetic, or backfilled data was used at any point.

**Authentication:** Bluesky's authenticated `createSession` endpoint was used with credentials stored in `Round3/.env` (never hardcoded or printed). An initial unauthenticated attempt via `public.api.bsky.app` returned HTTP 403 (geographic block) and was documented in `collection_log.md`.

**Collection mode:** This was a **single-shot collection** — a single script run against the Bluesky API. No multi-run or continuous collection occurred. The `collection_log.md` records the single successful run.

**Search queries used:**
- `Apple iPhone`
- `iphone`
- `technology` *(later filtered out — see relevance filtering below)*

**Fields collected per post:** URI, text, `created_at` (source timestamp), `collected_at` (UTC timestamp at time of script execution), author handle, like count, repost count, reply count, quote count, permalink, query term.

**Deduplication:** Posts were deduplicated by stable post URI. No post appears more than once.

**Relevance filtering (Fix 1):** The `technology` query introduced 496 off-topic posts with no product keyword in their text. An additional 237 posts from the `Apple iPhone`/`iphone` queries also lacked product-specific keywords. A keyword regex filter (`00_filter_relevant.py`) was applied to the raw dataset. The **original raw dataset is preserved unchanged** at `bluesky_raw_final.csv`. The filtered dataset `bluesky_relevant.csv` is the input to all downstream analysis.

| Metric | Count |
|---|---|
| Original posts collected (raw) | 1,320 |
| Posts excluded (off-topic) | 733 |
| **Posts used for analysis** | **587** |

---

## 2. Time Window

**Actual source timestamp range** (from `created_at` field, UTC):
- **Earliest:** `2026-09-21T21:25:34+00:00`
- **Latest:** `2026-09-22T18:34:20.783992+00:00`

**Distinction of timestamps:**
- `created_at` — the timestamp when the Bluesky user authored the post (source clock)
- `collected_at` — the UTC timestamp when our collection script retrieved the post (approximately `2026-09-22T16:43:30Z`)

**This was a single-shot collection.** The source timestamp range reflects the age of posts returned by the Bluesky search API, not a continuous monitoring window. There is no guarantee of complete coverage between the earliest and latest timestamps.

**Future-dated timestamp anomalies:** 6 posts in the filtered dataset have `created_at` later than their `collected_at`. The cause of these anomalies is unknown — they are preserved unaltered in all datasets. The `future_dated_anomaly` flag is set to `True` for these posts in `bluesky_enriched.csv`. They are included in all analyses using their original timestamps.

> [!NOTE]
> Do not interpret the source timestamp range as evidence of continuous real-time monitoring. This was a single retrieval operation.

---

## 3. Sentiment Analysis

### Model
The **frozen Round 2 TF-IDF + LinearSVC model** was reused without any retraining, refitting, or modification. Round 2 training and evaluation are entirely independent of Round 3.

- **Round 2 Validation Macro-F1:** 60.11%
- **Round 2 Test Macro-F1:** 60.93%
- **Classes:** Negative, Neutral, Positive

### Preprocessing
The exact `clean_text()` function from Round 2 (`08_inference_pipeline.py`) was applied via dynamic import (`importlib`). No modifications were made to the preprocessing logic.

### Round 3 Results (587 posts)

| Sentiment | Count | Percentage |
|---|---|---|
| Neutral | 213 | 36.3% |
| Positive | 208 | 35.4% |
| Negative | 166 | 28.3% |

*(See **Figure 1** for the hourly breakdown of sentiment percentages over time.)*

### Detected Sentiment Shifts
**10 sentiment shifts** were detected across the observation window. A shift is defined as an absolute change of ≥ 15 percentage points in polarity (`pct_positive − pct_negative`) between two adjacent non-empty hourly buckets.

The model's known limitations apply: it is English-centric, struggles with cross-language text, and may misclassify mixed-entity sentiment (e.g., posts praising a competitor while criticising Apple).

---

## 4. Activity Analysis

Posts were aggregated into **22 hourly UTC buckets** spanning the observation window.

**Engagement definition:** `likes + reposts + replies + quotes` (total per bucket)

**Spike detection threshold:** `mean_engagement + 2 × std_engagement`
- Mean engagement per bucket: **34.23**
- Std deviation: **40.49**
- Threshold: **115.21 total engagement units**

All 22 buckets were scanned; no cherry-picking occurred.

### Detected Engagement Spike

**1 genuine engagement spike** was detected:

| Metric | Value |
|---|---|
| Bucket | `2026-09-22T13:00:00+00:00` |
| Total engagement | 151 |
| Z-score | 2.884 |
| Threshold | 115.21 |

*(See **Figure 1** for engagement volume by hour with flagged events marked.)*

> [!IMPORTANT]
> Prior to Fix 2, the engagement spike detector incorrectly used post volume (post count) instead of the defined engagement metric. This has been corrected. The corrected detector found 1 genuine spike; the earlier incorrect implementation found 2.

---

## 5. Topic / Entity Analysis

### Topic Extraction
A fresh **Round 3 TF-IDF** vectorizer (unigrams + bigrams, scikit-learn) was applied to each event bucket's posts and compared against the immediately preceding equal-duration bucket.

**Preprocessing artifacts removed:** Before topic extraction, the following artifacts were stripped from `clean_text` values to prevent them surfacing as spurious topics: `<url>` placeholder, raw HTTP(S) links, `<amp>`/`&amp;` HTML entities, `@mentions`, and isolated digits. This ensures topics reflect genuine discussion content.

**Key topics by event:** `iphone pro`, `iphone duo`, `ios`, `apple`, `charging`, `android`, `renewed`, `unlocked`. The `url` artifact that appeared in the pre-Fix-3 output has been eliminated.

*(See **Figure 2** for a topic comparison for EVT_003, the most informative flagged event.)*

### Entity Extraction
**Method:** Curated regex dictionary (33 patterns) covering:
- **Companies:** Apple, Samsung, Google, Xiaomi, Microsoft, Amazon, Meta
- **Products:** iPhone 17, iPhone 18, iPhone Duo, iPhone Pro/Pro Max, AirPods, Apple Watch, Vision Pro, Apple Pencil, iPad, MacBook
- **OS/Platforms:** iOS 26, iOS 27, Android, App Store, Bluesky
- **Chips:** A18, A19, M5, M6
- **People:** Tim Cook, Mark Gurman

**Results:** Genuine entities were extracted for **all 11 flagged events**.

Dominant entities across events:
- **Apple** — present in every event bucket (4–35 mentions)
- **iPhone 18** — prominent in EVT_005 through EVT_010 (mid-morning through afternoon)
- **iPhone Duo** — appears in EVT_003, EVT_005–EVT_011
- **iOS 27** — consistent across most events
- **Amazon** — present in several events, likely affiliate/price-alert posts
- **Samsung, Android, Xiaomi** — appear in comparison/competitive posts (EVT_006, EVT_008, EVT_011)

**Limitation:** Both spaCy and NLTK `ne_chunk` were attempted but hung indefinitely in the current Windows environment and were abandoned. The regex dictionary approach only matches entities in its curated vocabulary; novel, misspelled, or abbreviated entities are not captured.

---

## 6. Trigger Explanations

All **11 flagged events** were individually investigated using the **HackerNews Algolia Search API** (`hn.algolia.com`). Search window: 24 hours before to 6 hours after each event timestamp. 79 candidate articles were retrieved in total.

**Important:** No social-media post, repost, or article merely repeating a social-media claim was treated as independent evidence. No external article was claimed to have "caused" any social-media shift — the language used is "consistent with" or "candidate explanation."

### Evidence Classification Rules
- **Strong:** Independent credible source retrieved; published before/during the event; specifically relevant to the event's product/topics; timing consistent
- **Weak:** Plausible connection but evidence is incomplete, timing is ambiguous, or topic match is partial
- **Unexplained:** No credible independent explanation found, or evidence is irrelevant/post-event

### Final Classification

| Event | Type | Timestamp (UTC) | Topics | Entities | Classification |
|---|---|---|---|---|---|
| EVT_001 | Sentiment shift | 2026-09-21T22:00 | iphone, apple, pro, ios | Apple, iOS 27, iPhone 18 | Unexplained |
| EVT_002 | Sentiment shift | 2026-09-22T00:00 | apple iphone, renewed, unlocked | Apple, Amazon, iPhone 18 | Unexplained |
| EVT_003 | Sentiment shift | 2026-09-22T02:00 | iphone, apple, iphone pro, duo | Apple, iPhone 18, iPhone Duo | Weak |
| EVT_004 | Sentiment shift | 2026-09-22T03:00 | iphone, charging, charger | Apple, Amazon, iPhone 18 | Unexplained |
| EVT_005 | Sentiment shift | 2026-09-22T04:00 | iphone, apple, iphone pro | Apple, iPhone 18, Apple Watch | Unexplained |
| EVT_006 | Sentiment shift | 2026-09-22T07:00 | iphone, apple, pro, duo | Apple, iPhone 17, iPhone Duo | Unexplained |
| EVT_007 | Sentiment shift | 2026-09-22T08:00 | iphone, apple, pro, ios | Apple, iOS, iOS 27 | Unexplained |
| EVT_008 | Sentiment shift | 2026-09-22T11:00 | iphone, apple, ios, iphone pro | Apple, iPhone 18, iPhone Duo | Unexplained |
| EVT_009 | Sentiment shift | 2026-09-22T12:00 | iphone, pro, apple, iphone pro | Apple, iPhone 18, iOS 27 | Unexplained |
| EVT_010 | Engagement spike | 2026-09-22T13:00 | iphone, pro, iphone pro, duo | Apple, iPhone 18, iPhone Duo | Unexplained |
| EVT_011 | Sentiment shift | 2026-09-22T16:00 | iphone, apple, pro, android | Apple, Amazon, iPhone 18 | Unexplained |

**Final evidence classification:**
- **Strong: 0**
- **Weak: 1** (EVT_003 — iPhone teardown video published ~20 hours prior, topic overlap with "iPhone 18" and "duo")
- **Unexplained: 10**

Most events remain unexplained because: (1) HackerNews skews toward developer/technical content and misses consumer news triggers; (2) the single-shot collection provides no pre-event baseline for comparison; and (3) the narrow observation window limits causal inference.

---

## Final Limitations

1. **Single-shot collection:** Data was pulled in one API call. Bucket post volumes are heavily skewed toward the collection time window rather than representing a continuous real-time stream.
2. **Bluesky source characteristics:** Bluesky is smaller than Twitter/Reddit; the dataset may not represent broader public reaction.
3. **Future-dated timestamp anomalies:** 6 posts have `created_at` later than `collected_at`. Cause unknown; timestamps preserved unaltered.
4. **Round 2 model limitations:** Frozen TF-IDF + LinearSVC (Macro-F1 ~60%) is English-centric and handles cross-language text and mixed-entity sentiment poorly.
5. **Entity extraction:** Regex dictionary covers known entities only; NLTK/spaCy unavailable in this environment.
6. **Trigger evidence:** HackerNews API is developer-oriented; 10 of 11 events remain unexplained.
7. **Limited observation window:** ~21 hours of effective source data.
8. **No pre-event baseline:** Single-shot collection cannot establish baseline sentiment before the launch event.

---

## Figures

**Figure 1 — Sentiment & Activity Timeline**
Shows hourly post volume (bars), sentiment percentages (lines), and all 11 flagged events (vertical markers).
Blue markers = sentiment shifts. Orange markers = engagement spikes.
*(See `reports/figure1_timeline.png`)*

**Figure 2 — Event Topic Analysis (EVT_003)**
Compares top TF-IDF terms in the event bucket (02:00 UTC) against the preceding bucket.
Event topics: `iphone, apple, iphone pro, pro, duo`
Previous topics: `apple, iphone, charging station, charger, station`
Does not imply causation.
*(See `reports/figure2_topics.png`)*
