# PHASE 2 — LIVE DATA COLLECTION

Execute **only Phase 2**. Do not start Phase 3.

## Objective

Collect genuine public data for the Round 3 topic **“Public Reaction to a Viral Product Launch”** using **Bluesky’s public API** instead of Reddit.

## Source

Use the official Bluesky public endpoint:

`https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts`

No Reddit/PRAW, API key, or fabricated fields.

Read the product name and search terms from `Round3/config.py`. Remove/ignore Reddit-specific settings such as subreddits.

## Implement

Create/update:

`Round3/src/01_collect_bluesky.py`

For each returned post, preserve available real fields including:

* `post_uri`
* `text`
* `created_at`
* `indexed_at`
* `collected_at` (UTC timestamp added by collector)
* `author_handle`
* `like_count`
* `repost_count`
* `reply_count`
* `quote_count`
* `permalink`
* `query_term`

Use actual API fields only. Do not invent Reddit-style `score`, `subreddit`, or `num_comments`.

## Storage

Save:

* `Round3/data/raw/bluesky_raw.csv`
* `Round3/data/raw/bluesky_raw_final.csv`

Append results and deduplicate using the stable post URI/ID. Preserve the raw post text and source timestamps.

## Collection Log

Create/update:

`Round3/reports/collection_log.md`

Record for **every run**:

* UTC run time
* query/term
* API results
* rows added
* total unique rows
* errors/retries
* source `created_at` range
* collector timestamp

At the end record:
`ACTUAL_FIRST_RUN_UTC`, `ACTUAL_LAST_RUN_UTC`, `TOTAL_RUNS_COMPLETED`, `COVERAGE_GAPS`, and `COLLECTION_MODE`.

Use `multi-run collection` only if multiple actual runs occurred. If only one run occurred, record `single-shot pull`. **Do not claim continuous collection or backfill historical data.**

## Constraints

* Do not fabricate posts, timestamps, engagement, or events.
* Do not modify Round 2 files.
* Do not overwrite existing raw data incorrectly.
* If the API fails, document the failure rather than creating substitute data.
* Preserve actual collection/source timestamps so later phases can distinguish `created_at` from `collected_at`.

## Final Check

Verify the final CSV exists, is deduplicated, contains the expected raw fields, and has non-empty text/timestamps.

Then **stop after Phase 2** and report:

1. files created/changed
2. total unique posts
3. actual source timestamp range
4. actual collection run count
5. collection mode
6. any gaps/errors
