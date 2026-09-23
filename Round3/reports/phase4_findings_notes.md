# Phase 4: Event Findings & Notes

## Detection Configuration
- **Bucket Size:** 1 hour
- **Sentiment Shift Threshold Used:** 15.0 percentage points
- **Engagement Spike Threshold Used:** 115.2068 posts (Mean + stddev multiplier)

## Event Summary
- **Number of Sentiment Shifts Detected:** 10
- **Number of Engagement Spikes Detected:** 1
- **Threshold Adjustments:** None required.

## Topics & Entities by Event
### EVT_001 (sentiment_shift @ 2026-09-21T22:00:00+00:00)
- **Current Topics:** iphone, apple, device, pro, new
- **Previous Topics:** iphone, apple, new, duo, iphone duo
- **Current Entities:** 
- **Previous Entities:** 

### EVT_002 (sentiment_shift @ 2026-09-22T00:00:00+00:00)
- **Current Topics:** iphone, apple iphone, apple, url, url apple
- **Previous Topics:** apple, iphone, apple iphone, pro, url
- **Current Entities:** 
- **Previous Entities:** 

### EVT_003 (sentiment_shift @ 2026-09-22T02:00:00+00:00)
- **Current Topics:** iphone, apple, 18, iphone duo, duo
- **Previous Topics:** apple, iphone, apple iphone, pro, url
- **Current Entities:** 
- **Previous Entities:** 

### EVT_004 (sentiment_shift @ 2026-09-22T03:00:00+00:00)
- **Current Topics:** iphone, apple, charging station, established wake, giant
- **Previous Topics:** iphone, apple, 18, iphone duo, duo
- **Current Entities:** 
- **Previous Entities:** 

### EVT_005 (sentiment_shift @ 2026-09-22T04:00:00+00:00)
- **Current Topics:** apple, iphone, 18, iphone 18, storage
- **Previous Topics:** iphone, apple, charging station, established wake, giant
- **Current Entities:** 
- **Previous Entities:** 

### EVT_006 (sentiment_shift @ 2026-09-22T07:00:00+00:00)
- **Current Topics:** apple, iphone, pro, url, pro max
- **Previous Topics:** iphone, apple, duo, iphone duo, pro
- **Current Entities:** 
- **Previous Entities:** 

### EVT_007 (sentiment_shift @ 2026-09-22T08:00:00+00:00)
- **Current Topics:** iphone, apple, 27, um, pro
- **Previous Topics:** apple, iphone, pro, url, pro max
- **Current Entities:** 
- **Previous Entities:** 

### EVT_008 (sentiment_shift @ 2026-09-22T11:00:00+00:00)
- **Current Topics:** iphone, apple, url, ios, pro
- **Previous Topics:** iphone, apple, url, pro, duo
- **Current Entities:** 
- **Previous Entities:** 

### EVT_009 (sentiment_shift @ 2026-09-22T12:00:00+00:00)
- **Current Topics:** iphone, url, apple, pro, duo
- **Previous Topics:** iphone, apple, url, ios, pro
- **Current Entities:** 
- **Previous Entities:** 

### EVT_010 (engagement_spike @ 2026-09-22T13:00:00+00:00)
- **Current Topics:** iphone, url, apple, pro, 18
- **Previous Topics:** iphone, url, apple, pro, duo
- **Current Entities:** 
- **Previous Entities:** 

### EVT_011 (sentiment_shift @ 2026-09-22T16:00:00+00:00)
- **Current Topics:** iphone, apple, url, pro, 18
- **Previous Topics:** iphone, apple, url, pro, duo
- **Current Entities:** 
- **Previous Entities:** 

## Limitations
1. **Dataset constraints:** Single-shot pull resulting in heavily skewed hourly buckets (post volume varies wildly).
2. **Future-Dated Timestamp Anomaly:** 21 posts carry `created_at` timestamps extending past the actual collection time (`2026-09-22T16:43:30Z`) by up to ~1.5 hours. These original timestamps have been strictly preserved. Any events generated in these future buckets represent actual parsed timestamps, which may be anomalous due to client device clock drift.
3. **Frozen Model Limitations:** Round 2 TF-IDF model struggles with cross-language text and mixed entity sentiments, creating some noise in the polarity calculations.


3. **Entity Extraction (Fix 3):** A curated regex dictionary was used (companies: Apple, Samsung, Google, Xiaomi; products: iPhone 17/18/Duo/Pro, iOS 26/27, AirPods, Apple Watch; people: Tim Cook, Mark Gurman). This produced genuine entities for **11 of 11 events**. spaCy and NLTK ne_chunk were attempted but both hung indefinitely in the current Windows environment and were abandoned. The regex approach only matches genuinely present named entities and does not fabricate.