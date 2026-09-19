"""
08_validate_phase2_results.py
Independently validates all three Phase 2 queries using pandas only (no SQL),
then compares results to the SQL output in data/datavortex.db.

Checks:
  E3 — NULL-platform exclusion, post count, per-platform averages,
        reconciliation with Phase 1 Insight #2 (platform spread ~3%).
  M1 — location count, total post count, top-3 location match.
  H2 — rank ordering, no duplicate (user, location), top-user cross-check,
        locations with fewer than 3 active users, rank-3 ties.

All results written to reports/phase2/phase2_validation.md.

Run from the project root:
    python src/08_validate_phase2_results.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import sqlite3
import pandas as pd
import numpy as np
from utils import CLEAN_USERS_PATH, CLEAN_POSTS_PATH

DB_PATH    = "data/datavortex.db"
REPORT_DIR = "reports/phase2"
OUT_PATH   = os.path.join(REPORT_DIR, "phase2_validation.md")

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []   # list of (check_label, status, actual, expected, note)


def record(label, passed, actual="", expected="", note=""):
    status = PASS if passed else FAIL
    results.append((label, status, str(actual), str(expected), note))
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {label}")
    if not passed:
        print(f"        actual={actual!r}  expected={expected!r}  {note}")


def load_data():
    users = pd.read_csv(CLEAN_USERS_PATH)
    posts = pd.read_csv(CLEAN_POSTS_PATH)
    posts['engagement'] = posts[['likes', 'shares', 'comments']].apply(
        lambda r: (0 if pd.isna(r['likes']) else r['likes']) + r['shares'] + r['comments'],
        axis=1
    )
    merged = posts.merge(users, on='user_id', how='left')
    return users, posts, merged


def fetch_sql(query: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query(query, con)
    con.close()
    return df


# ── E3 validation ─────────────────────────────────────────────────────────────

def validate_e3(posts, merged):
    print("\n== E3 ==================================================================")

    # pandas reference
    posts_with_plat = posts[posts['platform'].notna()].copy()

    # 1. NULL exclusion
    n_null_plat = posts['platform'].isna().sum()
    record("E3.1 NULL-platform rows excluded",
           n_null_plat == 1784,
           actual=n_null_plat, expected=1784,
           note="dataset should have 1,784 NULL-platform posts")

    # 2. Post count in SQL result
    sql_e3 = fetch_sql("""
        SELECT platform, COUNT(*) AS post_count,
               AVG(likes) AS avg_likes, AVG(shares) AS avg_shares,
               AVG(comments) AS avg_comments,
               AVG(COALESCE(likes,0)+shares+comments) AS avg_total_engagement
        FROM posts WHERE platform IS NOT NULL
        GROUP BY platform ORDER BY avg_total_engagement DESC
    """)
    sql_total = sql_e3['post_count'].sum()
    pd_total  = len(posts_with_plat)
    record("E3.2 Total post count in SQL matches pandas",
           sql_total == pd_total,
           actual=sql_total, expected=pd_total)

    # 3. Platform row count
    record("E3.3 SQL returns 5 platform rows",
           len(sql_e3) == 5,
           actual=len(sql_e3), expected=5)

    # 4. Per-platform avg_total_engagement vs pandas
    pd_plat = posts_with_plat.groupby('platform').agg(
        avg_total_engagement=('engagement', 'mean')
    ).reset_index()
    tol = 0.01   # within 0.01 units
    all_match = True
    mismatch_detail = []
    for _, row in sql_e3.iterrows():
        plat = row['platform']
        pd_val = pd_plat.loc[pd_plat['platform'] == plat, 'avg_total_engagement'].values
        if len(pd_val) == 0:
            all_match = False
            mismatch_detail.append(f"{plat}: not found in pandas")
            continue
        diff = abs(row['avg_total_engagement'] - pd_val[0])
        if diff > tol:
            all_match = False
            mismatch_detail.append(f"{plat}: SQL={row['avg_total_engagement']:.4f} pandas={pd_val[0]:.4f}")
    record("E3.4 Per-platform avg_total_engagement matches pandas (tol 0.01)",
           all_match,
           actual="see detail" if not all_match else "all match",
           expected="all match",
           note="; ".join(mismatch_detail))

    # 5. Reconcile with Phase 1 Insight #2 (~3% spread)
    spread_pct = (sql_e3['avg_total_engagement'].max() - sql_e3['avg_total_engagement'].min()) \
                 / sql_e3['avg_total_engagement'].min() * 100
    record("E3.5 Platform engagement spread is narrow (< 10%) — consistent with Phase 1 Insight #2",
           spread_pct < 10,
           actual=f"{spread_pct:.2f}%", expected="< 10%",
           note="Phase 1 reported ~3% spread")

    return sql_e3


# ── M1 validation ─────────────────────────────────────────────────────────────

def validate_m1(users, posts, merged):
    print("\n== M1 ==================================================================")

    sql_m1 = fetch_sql("""
        SELECT u.location,
               COUNT(p.post_id) AS post_count,
               SUM(COALESCE(p.likes,0)+p.shares+p.comments) AS total_engagement,
               ROUND(AVG(COALESCE(p.likes,0)+p.shares+p.comments),1) AS avg_engagement_per_post
        FROM users u JOIN posts p ON u.user_id=p.user_id
        GROUP BY u.location ORDER BY total_engagement DESC
    """)

    # 1. Location count
    n_locs = sql_m1['location'].nunique()
    pd_locs = merged['location'].nunique()
    record("M1.1 Location count matches pandas",
           n_locs == pd_locs,
           actual=n_locs, expected=pd_locs)

    # 2. Total post count
    sql_posts = sql_m1['post_count'].sum()
    record("M1.2 Total post count across all locations is 12,000",
           sql_posts == 12000,
           actual=sql_posts, expected=12000)

    # 3. pandas top-3 comparison
    pd_m1 = merged.groupby('location').agg(
        total_engagement=('engagement', 'sum')
    ).sort_values('total_engagement', ascending=False).reset_index()
    sql_top3  = list(sql_m1.head(3)['location'])
    pd_top3   = list(pd_m1.head(3)['location'])
    record("M1.3 Top-3 locations match pandas independently",
           sql_top3 == pd_top3,
           actual=sql_top3, expected=pd_top3)

    # 4. total_engagement per location within 1 unit of pandas
    pd_eng = pd_m1.set_index('location')['total_engagement']
    mismatches = []
    for _, row in sql_m1.iterrows():
        loc = row['location']
        if loc not in pd_eng:
            mismatches.append(f"{loc}: not in pandas")
            continue
        diff = abs(row['total_engagement'] - pd_eng[loc])
        if diff > 1:
            mismatches.append(f"{loc}: SQL={row['total_engagement']} pandas={pd_eng[loc]}")
    record("M1.4 Per-location total_engagement matches pandas (tol 1)",
           len(mismatches) == 0,
           actual="all match" if not mismatches else mismatches[:3],
           expected="all match")

    return sql_m1


# ── H2 validation ─────────────────────────────────────────────────────────────

def validate_h2(merged):
    print("\n== H2 ==================================================================")

    sql_h2 = fetch_sql("""
        WITH user_engagement AS (
            SELECT u.user_id, u.location, u.follower_count,
                   COUNT(p.post_id) AS post_count,
                   SUM(COALESCE(p.likes,0)+p.shares+p.comments) AS total_engagement
            FROM users u JOIN posts p ON u.user_id=p.user_id
            GROUP BY u.user_id, u.location, u.follower_count
        ),
        ranked AS (
            SELECT *,
                   RANK() OVER (PARTITION BY location ORDER BY total_engagement DESC) AS location_rank
            FROM user_engagement
        )
        SELECT location, location_rank, user_id, follower_count, post_count, total_engagement
        FROM ranked WHERE location_rank <= 3
        ORDER BY location, location_rank
    """)

    # 1. No duplicate (user_id, location)
    dupes = sql_h2.duplicated(subset=['user_id', 'location']).sum()
    record("H2.1 No duplicate (user_id, location) pairs in result",
           dupes == 0,
           actual=dupes, expected=0)

    # 2. Rank ordering: within each location, total_engagement is non-increasing
    bad_order = []
    for loc, grp in sql_h2.groupby('location', sort=False):
        vals = list(grp['total_engagement'])
        if vals != sorted(vals, reverse=True):
            bad_order.append(loc)
    record("H2.2 Within each location, rows are ordered DESC by total_engagement",
           len(bad_order) == 0,
           actual=bad_order[:3] if bad_order else "none",
           expected="none")

    # 3. All location_rank values are 1, 2, or 3
    bad_rank = sql_h2[~sql_h2['location_rank'].isin([1, 2, 3])]
    record("H2.3 All returned location_rank values are 1, 2, or 3",
           len(bad_rank) == 0,
           actual=len(bad_rank), expected=0)

    # 4. Locations with fewer than 3 active users (should have fewer rows)
    pd_user_loc = merged.groupby(['user_id', 'location'])['engagement'].sum().reset_index()
    users_per_loc = pd_user_loc.groupby('location').size()
    small_locs = users_per_loc[users_per_loc < 3].index.tolist()
    if small_locs:
        for loc in small_locs:
            n_rows = (sql_h2['location'] == loc).sum()
            max_rank = sql_h2.loc[sql_h2['location'] == loc, 'location_rank'].max() if n_rows > 0 else 0
            record(f"H2.4 Location '{loc}' has < 3 active users — rows={n_rows}, max_rank={max_rank}",
                   n_rows <= users_per_loc.get(loc, 0),
                   actual=n_rows, expected=f"≤ {users_per_loc.get(loc, 0)}")
    else:
        record("H2.4 Locations with fewer than 3 active users",
               True, actual="none found (all locations have ≥ 3 active users)", expected="any value")

    # 5. Rank-3 ties
    rank3 = sql_h2[sql_h2['location_rank'] == 3]
    ties  = rank3.groupby('location').size()
    n_tied_locs = (ties > 1).sum()
    record(f"H2.5 Rank-3 tie check: {n_tied_locs} location(s) have >1 user at rank 3",
           True,   # informational — ties are allowed and reported, not a failure
           actual=f"{n_tied_locs} location(s)",
           expected="any (ties allowed)",
           note="RANK() used, not ROW_NUMBER()")

    # 6. Top-1 user per location cross-check with pandas
    pd_user_eng = merged.groupby(['location', 'user_id'])['engagement'].sum().reset_index()
    pd_top1 = pd_user_eng.sort_values(['location', 'engagement'], ascending=[True, False]) \
                          .groupby('location').first().reset_index()[['location', 'user_id']]
    sql_top1 = sql_h2[sql_h2['location_rank'] == 1][['location', 'user_id']]
    merged_top = pd_top1.merge(sql_top1, on='location', suffixes=('_pd', '_sql'))
    mismatch_top = merged_top[merged_top['user_id_pd'] != merged_top['user_id_sql']]
    record("H2.6 Top-1 user per location matches pandas independently",
           len(mismatch_top) == 0,
           actual=f"{len(mismatch_top)} mismatch(es)", expected="0 mismatches",
           note=list(mismatch_top['location']) if len(mismatch_top) else "")

    return sql_h2


# ── report writer ─────────────────────────────────────────────────────────────

def write_validation_report():
    os.makedirs(REPORT_DIR, exist_ok=True)
    lines = [
        "# Phase 2 Validation Report\n",
        "All checks are computed independently in pandas and compared against "
        "the SQLite query results in `data/datavortex.db`.\n",
        "| Check | Status | Actual | Expected | Note |",
        "|---|---|---|---|---|",
    ]
    for label, status, actual, expected, note in results:
        row = f"| {label} | {status} | {actual} | {expected} | {note} |"
        lines.append(row)

    n_pass = sum(1 for _, s, *_ in results if "PASS" in s)
    n_fail = sum(1 for _, s, *_ in results if "FAIL" in s)
    lines += [
        "",
        f"## Summary\n",
        f"- **PASS:** {n_pass}",
        f"- **FAIL:** {n_fail}",
        f"- **Total checks:** {len(results)}",
        "",
    ]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nValidation report -> {OUT_PATH}")
    print(f"  PASS: {n_pass}   FAIL: {n_fail}   Total: {len(results)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Run src/06_build_sql_database.py first.")
        sys.exit(1)

    users, posts, merged = load_data()

    validate_e3(posts, merged)
    validate_m1(users, posts, merged)
    validate_h2(merged)

    write_validation_report()

    failed = [r for r in results if "FAIL" in r[1]]
    if failed:
        print(f"\n{len(failed)} check(s) FAILED - see {OUT_PATH} for details.")
        sys.exit(1)
    else:
        print("\nAll checks PASSED [OK]")


if __name__ == "__main__":
    main()
