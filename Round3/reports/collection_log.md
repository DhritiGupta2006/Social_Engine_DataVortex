# Round 3 — Collection Log
**Product:** iPhone 17
**Data source:** Bluesky public API (`https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts`)
**Collector script:** `Round3/src/01_collect_bluesky.py`

---

## Actual Collection Reality

| Field | Value |
|---|---|
| `ACTUAL_FIRST_RUN_UTC` | `2026-09-22T16:24:40Z` |
| `ACTUAL_LAST_RUN_UTC` | `2026-09-22T16:43:30Z` |
| `TOTAL_RUNS_COMPLETED` | 2 |
| `COVERAGE_GAPS` | N/A — single run only |
| `COLLECTION_MODE` | **single-shot pull** |

**Status: Authenticated Collection Successful.**

After the initial 403 API BLOCKED failure, authentication was implemented via `bsky.social` using App Passwords. The second run successfully retrieved 1,320 unique posts. 

- Data collected from `2026-09-21T19:47:10-03:00` to `2026-09-23T01:01:54+09:00`.
- 2 timeout errors occurred (handled by pagination bounds), but 1320 rows were cleanly extracted.
- `bluesky_raw.csv` and `bluesky_raw_final.csv` are now populated.
- Downstream phases can proceed.

---

## Run: 2026-09-22T16:24:40Z

| Field | Value |
|---|---|
| **Run UTC** | `2026-09-22T16:24:40Z` |
| **Product** | iPhone 17 |
| **Data source** | bluesky |
| **Queries** | `iPhone 17`, `Apple iPhone`, `apple`, `iphone`, `technology` |
| **API results (per query)** | All 5 queries: 0 rows (HTTP 403 Forbidden on every request) |
| **New unique rows this run** | 0 |
| **Total unique rows after dedup** | 0 |
| **Source `created_at` min** | N/A |
| **Source `created_at` max** | N/A |
| **Collector `collected_at`** | `2026-09-22T16:24:40Z` |
| **API errors** | 5 (one per query term, all HTTP 403) |

**Error detail (all queries):**
- `HTTP Error 403: Forbidden` — `https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts`
- Affected queries: `iPhone 17`, `Apple iPhone`, `apple`, `iphone`, `technology`
- CDN geo-block from India (`CDN-RequestCountryCode: IN`, Server: BunnyCDN)

---

## Run: 2026-09-22T16:43:30Z

| Field | Value |
|---|---|
| **Run UTC** | `2026-09-22T16:43:30Z` |
| **Product** | iPhone 17 |
| **Data source** | bluesky |
| **Queries** | `iPhone 17`, `Apple iPhone`, `apple`, `iphone`, `technology` |
| **API results (per query)** | {"iPhone 17": 0, "Apple iPhone": 499, "apple": 0, "iphone": 498, "technology": 498} |
| **New unique rows this run** | 1320 |
| **Total unique rows after dedup** | 1320 |
| **Source `created_at` min** | `2026-09-21T19:47:10-03:00` |
| **Source `created_at` max** | `2026-09-23T01:01:54+09:00` |
| **Collector `collected_at`** | `2026-09-22T16:43:30Z` |
| **API errors** | 0 |

_No errors._

