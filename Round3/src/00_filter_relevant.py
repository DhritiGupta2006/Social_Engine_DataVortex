"""
Round3/src/00_filter_relevant.py
Fix 1: Data Relevance Filtering

Filters bluesky_raw_final.csv to posts genuinely relevant to
the product launch topic ("iPhone 17" / "Apple iPhone"), discarding
posts that arrived solely via the generic 'technology' query and
contain no on-topic keywords in their text.

The original raw dataset is NEVER modified.
Output: Round3/data/raw/bluesky_relevant.csv
"""

import sys
import csv
import re
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROUND3_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROUND3_DIR))
import config as cfg

# ---------------------------------------------------------------------------
# Relevance keywords — at least one must appear in the post text (case-insensitive)
# Chosen to be specific to the product-launch topic, not generic tech commentary.
# ---------------------------------------------------------------------------
RELEVANCE_PATTERN = re.compile(
    r'\b(iphone\s*1[67]|iphone\s*18|iphone\s*duo|apple\s*iphone|iphone\s+pro|'
    r'iphone\s+launch|ios\s*2[67]|alarmkit|ios\s+26|ios\s+27|'
    r'iphone\s+preorder|iphone\s+order|iphone\s+release|'
    r'apple\s+launch|apple\s+event|apple\s+announcement|'
    r'iphone\s+camera|iphone\s+battery|iphone\s+price|iphone\s+review|'
    r'iphone\s+chip|iphone\s+feature|iphone\s+upgrade|iphone\s+buy|'
    r'iphone\s+unbox|iphone\s+teardown|apple\s+silicon|a19|a20|'
    r'iphone)\b',
    re.IGNORECASE
)


def is_relevant(row: dict) -> tuple[bool, str]:
    """Return (True, reason) if the post is on-topic, else (False, reason)."""
    text = (row.get("text") or "").strip()
    query = (row.get("query_term") or "").strip().lower()

    # Posts retrieved by specific product queries are presumed relevant
    # (they still need to contain the keyword as a sanity check).
    if query in ("apple iphone", "iphone"):
        if RELEVANCE_PATTERN.search(text):
            return True, f"specific-query={query!r} + keyword-match"
        else:
            # Rare edge case: query matched but text doesn't contain any keyword
            return False, f"specific-query={query!r} but no keyword in text"

    # Posts from the generic 'technology' query need an explicit keyword in text.
    if query == "technology":
        if RELEVANCE_PATTERN.search(text):
            return True, "technology-query + keyword-match"
        else:
            return False, "technology-query, no product keyword in text"

    # Catch-all for any unexpected query term
    if RELEVANCE_PATTERN.search(text):
        return True, f"unknown-query={query!r} + keyword-match"
    return False, f"unknown-query={query!r}, no product keyword"


def main():
    print("=" * 65)
    print("DATA VORTEX A'26 Round 3 — Relevance Filter (Fix 1)")
    print("=" * 65)

    in_csv = cfg.RAW_FINAL_CSV
    out_csv = cfg.RAW_DIR / "bluesky_relevant.csv"

    if not in_csv.exists():
        print(f"  [ERROR] {in_csv.name} not found.")
        sys.exit(1)

    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total = len(rows)
    kept = []
    excluded = []

    for row in rows:
        ok, reason = is_relevant(row)
        if ok:
            kept.append(row)
        else:
            excluded.append((row.get("uri", ""), row.get("query_term", ""), reason,
                             (row.get("text") or "")[:120]))

    # Write filtered dataset
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    # Write exclusion log
    excl_log = cfg.RAW_DIR / "exclusion_log.csv"
    with open(excl_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["uri", "query_term", "reason", "text_preview"])
        writer.writerows(excluded)

    print(f"  Original rows  : {total}")
    print(f"  Kept (relevant): {len(kept)}")
    print(f"  Excluded       : {len(excluded)}")
    print(f"  Output CSV     : {out_csv.name}")
    print(f"  Exclusion log  : {excl_log.name}")

    # Breakdown by query_term
    from collections import Counter
    kept_qt = Counter(r.get("query_term", "") for r in kept)
    excl_qt = Counter(e[1] for e in excluded)
    print("\n  Kept by query_term:")
    for k, v in kept_qt.most_common():
        print(f"    {k!r}: {v}")
    print("  Excluded by query_term:")
    for k, v in excl_qt.most_common():
        print(f"    {k!r}: {v}")

    print("=" * 65)


if __name__ == "__main__":
    main()
