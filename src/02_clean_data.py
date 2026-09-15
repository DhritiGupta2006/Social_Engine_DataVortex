"""
02_clean_data.py
Cleans the raw CSVs and writes:
  - data/cleaned/users_cleaned.csv
  - data/cleaned/posts_cleaned.csv
  - reports/cleaning_log.md   (every decision + before/after counts)
Never writes to data/raw/. Never fabricates values: missing data stays
missing (NaN) rather than being guessed or imputed.
Run from the project root: python src/02_clean_data.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from utils import (
    load_raw_users, load_raw_posts, parse_timestamp, normalize_missing,
    clean_text, CLEAN_USERS_PATH, CLEAN_POSTS_PATH,
)

log = []


def clean_users():
    users = load_raw_users()
    n0 = len(users)
    log.append("## Users cleaning\n")
    log.append(f"- Raw rows: {n0}")

    users['location'] = users['location'].str.strip()
    users['language'] = users['language'].str.strip()
    users['account_created'] = pd.to_datetime(users['account_created']).dt.strftime('%Y-%m-%d')
    users['follower_count'] = pd.to_numeric(users['follower_count'], errors='raise').astype(int)

    n_dupe = users.duplicated().sum()
    users = users.drop_duplicates()
    log.append(f"- Whitespace trimmed on `location`/`language`; `account_created` "
                f"normalized to ISO `YYYY-MM-DD`; `follower_count` cast to int.")
    log.append(f"- Exact duplicate rows removed: {n_dupe}")
    log.append(f"- Final rows: {len(users)} (no other changes needed — see audit report)\n")

    os.makedirs("data/cleaned", exist_ok=True)
    users.to_csv(CLEAN_USERS_PATH, index=False)
    return users


def clean_posts(valid_user_ids):
    posts = load_raw_posts()
    n0 = len(posts)
    log.append("## Posts cleaning\n")
    log.append(f"- Raw rows: {n0}")

    # 1. Normalize placeholder missing markers to real NaN
    for col in ['platform', 'text_content', 'likes']:
        posts[col] = posts[col].apply(normalize_missing)
    n_platform_missing = posts['platform'].isna().sum()
    n_text_missing_raw = posts['text_content'].isna().sum()
    n_likes_missing = posts['likes'].isna().sum()
    log.append(f"- Normalized literal `NULL`/empty-string placeholders to real missing "
                f"values: platform={n_platform_missing}, text_content={n_text_missing_raw}, "
                f"likes={n_likes_missing}. **These rows are kept** (not dropped) — the "
                f"missing field just isn't usable for analyses that need it.")

    # 2. Decode HTML entities + trim whitespace in free text (restores original text,
    #    does not alter wording/meaning)
    posts['text_content'] = posts['text_content'].apply(clean_text)
    posts['platform'] = posts['platform'].apply(lambda v: v.strip() if isinstance(v, str) else v)
    log.append("- Decoded HTML entities (e.g. `&amp;` -> `&`) and trimmed stray whitespace "
                "in `text_content` and `platform`.")

    # 3. Standardize the 3 mixed timestamp formats into one ISO 8601 column
    posts['event_time'] = posts['timestamp'].apply(parse_timestamp)
    n_unparsed = posts['event_time'].isna().sum()
    log.append(f"- Parsed 3 mixed timestamp formats (unix epoch, ISO 8601, `DD-MM-YYYY`) "
                f"into a single standardized `event_time` column. Unparseable: {n_unparsed}.")

    # 4. Fix likes: cast to numeric, correct sign-flip corruption (see audit report for the
    #    distributional evidence that negative likes are the same underlying population with
    #    a flipped sign, not distinct/garbage values)
    posts['likes'] = pd.to_numeric(posts['likes'], errors='coerce')
    n_negative = (posts['likes'] < 0).sum()
    posts['likes'] = posts['likes'].abs()
    log.append(f"- Corrected {n_negative} negative `likes` values by taking the absolute "
                f"value. Justification: `abs(negative likes)` and positive `likes` have "
                f"near-identical mean/std/range (see audit report) — this is a sign-flip "
                f"corruption, not a separate population, so recovering the magnitude is a "
                f"correction, not a fabrication. `likes` is still missing (NaN) where the "
                f"raw value was itself missing.")

    posts['shares'] = pd.to_numeric(posts['shares'], errors='raise').astype(int)
    posts['comments'] = pd.to_numeric(posts['comments'], errors='raise').astype(int)

    # 5. Drop exact full-row duplicates (verified in audit: every repeated post_id group is
    #    byte-identical across all columns, so this is safe re-ingestion cleanup, not loss
    #    of distinct records)
    n_dup_groups = posts[posts.duplicated(keep=False)]['post_id'].nunique()
    dup_mask_full = posts.duplicated(keep='first')
    n_dupe = dup_mask_full.sum()
    posts = posts[~dup_mask_full].copy()
    log.append(f"- Removed {n_dupe} exact duplicate rows (all fields identical, confirmed "
                f"in the audit to be re-ingestion artifacts affecting {n_dup_groups} "
                f"distinct post_ids). Rows were matched on every column INCLUDING the raw "
                f"`timestamp`, so only true re-ingested copies were removed.")

    # 6. Referential integrity check against users (report only — none found to remove)
    orphan_mask = ~posts['user_id'].isin(valid_user_ids)
    n_orphan = orphan_mask.sum()
    log.append(f"- Rows referencing a `user_id` not present in the users table: {n_orphan} "
                f"(none found; no rows removed for this reason).")

    # 7. Final column selection / ordering. Keep the original raw `timestamp` string for
    #    full traceability alongside the new standardized `event_time`.
    posts = posts.rename(columns={'timestamp': 'timestamp_raw'})
    final_cols = ['post_id', 'user_id', 'platform', 'text_content', 'timestamp_raw',
                  'event_time', 'likes', 'shares', 'comments']
    posts = posts[final_cols].sort_values('event_time').reset_index(drop=True)

    log.append(f"- Final rows: {len(posts)} (from {n0} raw rows)")
    log.append(f"- Final missingness retained as NaN: platform={posts['platform'].isna().sum()}, "
                f"text_content={posts['text_content'].isna().sum()}, "
                f"likes={posts['likes'].isna().sum()}\n")

    posts.to_csv(CLEAN_POSTS_PATH, index=False)
    return posts


def main():
    users = clean_users()
    posts = clean_posts(set(users['user_id']))

    os.makedirs("reports", exist_ok=True)
    header = ["# DATA VORTEX — Cleaning Decision Log\n",
              "Generated by `src/02_clean_data.py`. Every transformation below is "
              "deterministic and reproducible from the raw files in `data/raw/`. "
              "No values were invented; missing data is preserved as missing.\n"]
    with open("reports/cleaning_log.md", "w") as f:
        f.write("\n".join(header + log) + "\n")

    print(f"Cleaned users -> {CLEAN_USERS_PATH} ({len(users)} rows)")
    print(f"Cleaned posts -> {CLEAN_POSTS_PATH} ({len(posts)} rows)")
    print("Cleaning log -> reports/cleaning_log.md")


if __name__ == "__main__":
    main()
