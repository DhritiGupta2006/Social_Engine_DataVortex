# DATA VORTEX — Insights: Rebuilding the Social Engine

All figures below are computed directly from `data/cleaned/*.csv` by `src/05_insights.py`.

## 1. Follower count is not a usable proxy for engagement or reach

- Correlation between author `follower_count` and total post engagement (likes+shares+comments): **r = -0.011**
- Correlation between `follower_count` and `likes` alone: r = 0.008
- Mean engagement for the top 10% most-followed authors (follower_count >= 44363): **3587**
- Mean engagement for the bottom 10% least-followed authors (follower_count <= 5446): **3638**
- The two groups differ by only 1.4%, despite a >100x difference in audience size. **Implication:** a rebuilt recommendation/ranking engine should not weight follower count as a proxy for expected engagement or influence — in this data it carries no signal.

## 2. Engagement is nearly platform-agnostic

Mean total engagement per post, by platform:

- Instagram: 3669
- Reddit: 3647
- YouTube: 3638
- Facebook: 3631
- Twitter: 3564

- Spread between the highest- and lowest-engagement platform is only **3.0%**, far smaller than the platform-level differences typically assumed in social strategy. **Implication:** for this data, platform-specific engagement models add little value versus a single shared model — effort is better spent elsewhere (e.g. content-level features, timing).

## 3. Data corruption is uniform noise, not a single faulty source

- Negative-`likes` rate ranges only from 4.2% to 6.4% across all 12 months of the campaign, and from 3.9% to 5.9% across platforms.
- Duplicate-row rate ranges only from 4.2% to 6.7% month over month.
- If the corruption came from one broken ingestion pipeline (e.g. a single platform integration or a bad deploy in one month), we'd expect the error rate to spike in that slice and be near-zero elsewhere. Instead every slice shows almost the same error rate. **Implication:** when rebuilding the ingestion layer, prioritize a general field-level validation/sanitization step (schema checks, sign checks, dedup keys) over auditing one specific platform connector or time window — the evidence points to systemic, randomly-injected noise rather than one localized bug.

## 4. A meaningful share of post text mixes contradictory sentiment cues

- Of 10289 posts with usable text: 2826 contain only positive sentiment phrases, 1024 contain only negative phrases, **310 contain both** in the same post, and 6129 contain neither.
- That means roughly **3.0%** of posts with detectable sentiment language would be mis-scored by a naive keyword-matching sentiment classifier (it would need to pick one polarity when the text asserts both). **Implication:** a rebuilt sentiment/insight layer needs phrase-order- or clause-aware sentiment modeling, not simple keyword lookups, or it will silently produce contradictory sentiment labels for a non-trivial slice of content.

## 5. Missing `platform` looks like a metadata-tagging gap, not corrupted records

- Posts WITH a platform tag have usable `likes` data 84.8% of the time.
- Posts WITHOUT a platform tag have usable `likes` data 85.4% of the time — essentially the same rate.
- If missing-platform rows were generally broken/incomplete records, we'd expect their other fields to be missing far more often too. They aren't. **Implication:** the ~14.9% of posts missing a platform tag are most likely healthy posts that lost one specific metadata field (e.g. a platform-ID lookup failure), not failed ingestions — worth a targeted backfill effort rather than being discarded.

## 6. A small set of campaign themes dominates hashtag usage

Top 5 hashtags by occurrence:

- #Reviews: 750 posts (7.3% of posts with text)
- #Fitness: 749 posts (7.3% of posts with text)
- #BestValue: 732 posts (7.1% of posts with text)
- #SpecialOffer: 723 posts (7.0% of posts with text)
- #Eco: 723 posts (7.0% of posts with text)

