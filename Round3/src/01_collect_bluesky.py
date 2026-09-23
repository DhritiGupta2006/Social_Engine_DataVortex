"""
Round3/src/01_collect_bluesky.py — DATA VORTEX A'26 Round 3
Phase 2: Live Data Collection via Bluesky Public API

Endpoint: https://bsky.social/xrpc/app.bsky.feed.searchPosts
Requires credentials (BLUESKY_HANDLE, BLUESKY_APP_PASSWORD) in Round3/.env.

Usage (from repo root):
    python Round3/src/01_collect_bluesky.py

Each run appends to bluesky_raw.csv and deduplicates by post_uri.
A final consolidated file is written as bluesky_raw_final.csv.
Run details are appended to Round3/reports/collection_log.md.

Must NOT Do:
- Do not fabricate posts, timestamps, or engagement counts.
- Do not modify Round2/ files.
- If the API fails, log the error and exit cleanly without creating substitute data.
"""

import sys
import csv
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import urllib.parse
import os
from dotenv import load_dotenv

# ── Force UTF-8 stdout on Windows (avoids cp1252 UnicodeEncodeError) ──────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' not installed. Run: pip install requests")
    sys.exit(1)

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROUND3_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROUND3_DIR))

import config as cfg

# ── Constants ──────────────────────────────────────────────────────────────────
API_BASE    = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
AUTH_URL    = "https://bsky.social/xrpc/com.atproto.server.createSession"
LIMIT       = 100           # max results per API page (Bluesky max)
MAX_PAGES   = 5             # max pagination pages per query term
SLEEP_SEC   = 1.0           # polite delay between page requests

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "DataVortex-Round3/0.1 (academic research project)",
}

# CSV column order — must match Phase 3 expectations
COLUMNS = [
    "post_uri",
    "text",
    "created_at",       # source timestamp (UTC ISO8601, from record.createdAt)
    "indexed_at",       # Bluesky indexing timestamp (from post.indexedAt)
    "collected_at",     # UTC timestamp of when this collector ran this row
    "author_handle",
    "author_did",
    "like_count",
    "repost_count",
    "reply_count",
    "quote_count",
    "permalink",
    "query_term",
]


# =============================================================================
# API helpers
# =============================================================================

def fetch_page(query: str, cursor: str | None, session: requests.Session) -> dict:
    """Fetch one page of search results. Returns raw API response dict."""
    params = {"q": query, "limit": LIMIT}
    if cursor:
        params["cursor"] = cursor
    resp = session.get(API_BASE, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def extract_post(raw_post: dict, query_term: str, collected_at: str) -> dict:
    """
    Extract the required fields from a raw Bluesky post object.
    Only uses real API fields — never invents values.
    Returns a dict matching COLUMNS.
    """
    uri    = raw_post.get("uri", "")
    record = raw_post.get("record", {})
    author = raw_post.get("author", {})

    # Permalink: convert at:// URI → bsky.app URL
    # at://did:plc:xxx/app.bsky.feed.post/yyy → https://bsky.app/profile/handle/post/yyy
    handle = author.get("handle", "")
    rkey   = uri.rsplit("/", 1)[-1] if "/" in uri else ""
    permalink = f"https://bsky.app/profile/{handle}/post/{rkey}" if handle and rkey else ""

    return {
        "post_uri":     uri,
        "text":         record.get("text", ""),
        "created_at":   record.get("createdAt", ""),   # source post timestamp
        "indexed_at":   raw_post.get("indexedAt", ""), # Bluesky index timestamp
        "collected_at": collected_at,                   # THIS run's UTC time
        "author_handle": handle,
        "author_did":    author.get("did", ""),
        "like_count":    raw_post.get("likeCount",   0),
        "repost_count":  raw_post.get("repostCount", 0),
        "reply_count":   raw_post.get("replyCount",  0),
        "quote_count":   raw_post.get("quoteCount",  0),
        "permalink":     permalink,
        "query_term":    query_term,
    }


# =============================================================================
# Collection
# =============================================================================

def collect_query(query: str, session: requests.Session, collected_at: str) -> list[dict]:
    """
    Collect all pages for a single query term.
    Returns list of extracted post dicts.
    Respects MAX_PAGES and SLEEP_SEC.
    """
    posts   = []
    cursor  = None
    page    = 0

    while page < MAX_PAGES:
        try:
            data   = fetch_page(query, cursor, session)
        except requests.exceptions.HTTPError as e:
            print(f"  [HTTP ERROR] {e} — stopping pagination for '{query}'")
            break
        except Exception as e:
            print(f"  [ERROR] {e} — stopping pagination for '{query}'")
            break

        raw_posts = data.get("posts", [])
        if not raw_posts:
            break

        for rp in raw_posts:
            posts.append(extract_post(rp, query, collected_at))

        cursor = data.get("cursor")
        page  += 1

        print(f"  Page {page}: {len(raw_posts)} posts (cursor={'yes' if cursor else 'none'})")

        if not cursor:
            break
        time.sleep(SLEEP_SEC)

    return posts


# =============================================================================
# Deduplication & persistence
# =============================================================================

def load_existing(path: Path) -> dict[str, dict]:
    """Load existing CSV into a dict keyed by post_uri. Returns {} if file absent."""
    existing = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uri = row.get("post_uri", "")
                if uri:
                    existing[uri] = row
    return existing


def save_csv(rows: list[dict], path: Path) -> None:
    """Write rows to CSV, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def append_to_log(log_path: Path, entry: str) -> None:
    """Append a text entry to collection_log.md."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    run_utc     = datetime.now(timezone.utc)
    collected_at = run_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    print("=" * 65)
    print("DATA VORTEX A'26 Round 3 — Bluesky Data Collector")
    print(f"Run UTC: {collected_at}")
    print("=" * 65)

    raw_csv   = cfg.RAW_POSTS_CSV          # bluesky_raw.csv   (append)
    final_csv = cfg.RAW_FINAL_CSV          # bluesky_raw_final.csv
    log_md    = cfg.COLLECTION_LOG_MD      # collection_log.md
    queries   = cfg.search_query_terms

    # ── Load existing deduplicated set ────────────────────────────────────────
    existing = load_existing(raw_csv)
    pre_count = len(existing)
    print(f"[1/4] Existing rows in raw file: {pre_count}")

    # ── Run collection ────────────────────────────────────────────────────────
    # ── Authenticate ──────────────────────────────────────────────────────────
    load_dotenv(_ROUND3_DIR / ".env")
    handle = os.getenv("BLUESKY_HANDLE")
    password = os.getenv("BLUESKY_APP_PASSWORD")

    if not handle or not password:
        print("[ERROR] Missing BLUESKY_HANDLE or BLUESKY_APP_PASSWORD in Round3/.env")
        sys.exit(1)

    session = requests.Session()
    print("[1.5/4] Authenticating with bsky.social...")
    try:
        auth_resp = session.post(
            AUTH_URL,
            json={"identifier": handle, "password": password},
            headers={"Content-Type": "application/json"}
        )
        auth_resp.raise_for_status()
        token = auth_resp.json().get("accessJwt")
        session.headers.update({"Authorization": f"Bearer {token}"})
        print("  [OK] Authenticated successfully.")
    except Exception as e:
        print(f"  [ERROR] Authentication failed: {e}")
        sys.exit(1)

    new_posts   = {}
    errors      = []
    per_query   = {}

    print(f"[2/4] Querying {len(queries)} search terms ...")
    for q in queries:
        print(f"\n  -> Query: '{q}'")
        try:
            posts = collect_query(q, session, collected_at)
            per_query[q] = len(posts)
            for p in posts:
                uri = p["post_uri"]
                if uri and uri not in existing:
                    new_posts[uri] = p
                    existing[uri]  = p
        except Exception as e:
            err_msg = f"FATAL error on query '{q}': {traceback.format_exc()}"
            errors.append(err_msg)
            print(f"  [FATAL] {e}")

    # ── Persist ───────────────────────────────────────────────────────────────
    print(f"\n[3/4] Persisting data ...")
    post_count_after = len(existing)
    new_count        = len(new_posts)
    print(f"  New unique posts this run : {new_count}")
    print(f"  Total unique posts        : {post_count_after}")

    if post_count_after > 0:
        # Append raw CSV (all rows, sorted by created_at)
        all_rows = sorted(existing.values(), key=lambda r: r.get("created_at", ""))
        save_csv(all_rows, raw_csv)

        # Write final (identical at end of single-shot run)
        save_csv(all_rows, final_csv)
        print(f"  Saved: {raw_csv.name}")
        print(f"  Saved: {final_csv.name}")
    else:
        print("  No data collected — raw files NOT written (no fabrication).")

    # ── Compute source timestamp range ────────────────────────────────────────
    created_ats = [r.get("created_at","") for r in existing.values() if r.get("created_at")]
    src_min     = min(created_ats) if created_ats else "N/A"
    src_max     = max(created_ats) if created_ats else "N/A"

    # ── Append to collection_log.md ───────────────────────────────────────────
    print(f"\n[4/4] Writing collection log ...")
    log_entry = f"""
---

## Run: {collected_at}

| Field | Value |
|---|---|
| **Run UTC** | `{collected_at}` |
| **Product** | {cfg.chosen_product} |
| **Data source** | {cfg.data_source} |
| **Queries** | {', '.join(f'`{q}`' for q in queries)} |
| **API results (per query)** | {json.dumps(per_query)} |
| **New unique rows this run** | {new_count} |
| **Total unique rows after dedup** | {post_count_after} |
| **Source `created_at` min** | `{src_min}` |
| **Source `created_at` max** | `{src_max}` |
| **Collector `collected_at`** | `{collected_at}` |
| **API errors** | {len(errors)} |

{"**Errors:**" + chr(10) + chr(10).join(f"- {e}" for e in errors) if errors else "_No errors._"}
"""
    append_to_log(log_md, log_entry)

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\n=== VALIDATION ===")
    if post_count_after == 0:
        print("  [FAIL] No data collected — check API reachability and log for errors.")
    else:
        print(f"  [OK] {post_count_after} unique posts in final CSV")
        missing_text = sum(1 for r in existing.values() if not r.get("text","").strip())
        print(f"  [{'WARN' if missing_text else 'OK'}] Rows with empty text: {missing_text}")
        missing_ts   = sum(1 for r in existing.values() if not r.get("created_at","").strip())
        print(f"  [{'WARN' if missing_ts else 'OK'}] Rows missing created_at: {missing_ts}")
        print(f"  [OK] Source timestamp range: {src_min}  →  {src_max}")

    print("\n=== DONE ===")
    return post_count_after, new_count, errors, src_min, src_max


if __name__ == "__main__":
    total, new, errs, src_min, src_max = main()
