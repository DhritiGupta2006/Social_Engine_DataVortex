# Data Vortex — Phase 2 SQL Implementation

## Objective

Implement Phase 2 for exactly:

* **E3** — Average Engagement by Platform
* **M1** — Which Locations Generate the Most Engagement
* **H2** — Rank Users Within Their Location

Phase 1 is already complete. **Phase 1 used Python/pandas, NOT SQL. Do not modify Phase 1.**

Read `README.md`, `STATE.md`, `src/utils.py`, `src/05_insights.py`, and the cleaned CSVs before implementing.

## Global Rule

For all queries:

```text
engagement = COALESCE(likes,0) + shares + comments
```

Reuse `CLEAN_USERS_PATH` and `CLEAN_POSTS_PATH` from `src/utils.py`.

Use SQLite via Python `sqlite3`. No unnecessary dependencies.

## E3

Create `sql/E3_platform_avg_engagement.sql`.

```sql
SELECT platform,
       COUNT(*) AS post_count,
       AVG(likes) AS avg_likes,
       AVG(shares) AS avg_shares,
       AVG(comments) AS avg_comments,
       AVG(COALESCE(likes,0) + shares + comments) AS avg_total_engagement
FROM posts
WHERE platform IS NOT NULL
GROUP BY platform
ORDER BY avg_total_engagement DESC;
```

Validate against Phase 1 `reports/insights.md` Insight #2.

Expected: 5 platforms and 10,216 non-NULL-platform posts if the dataset confirms 1,784 NULL platforms.

## M1

Create `sql/M1_location_engagement.sql`.

```sql
SELECT u.location,
       COUNT(p.post_id) AS post_count,
       SUM(COALESCE(p.likes,0) + p.shares + p.comments) AS total_engagement,
       ROUND(AVG(COALESCE(p.likes,0) + p.shares + p.comments), 1) AS avg_engagement_per_post
FROM users u
JOIN posts p ON u.user_id = p.user_id
GROUP BY u.location
ORDER BY total_engagement DESC;
```

Validate independently with pandas.

Expected: 33 locations and 12,000 posts, if confirmed by the actual dataset.

## H2

Create `sql/H2_rank_users_by_location.sql`.

```sql
WITH user_engagement AS (
    SELECT u.user_id,
           u.location,
           u.follower_count,
           COUNT(p.post_id) AS post_count,
           SUM(COALESCE(p.likes,0) + p.shares + p.comments) AS total_engagement
    FROM users u
    JOIN posts p ON u.user_id = p.user_id
    GROUP BY u.user_id, u.location, u.follower_count
),
ranked AS (
    SELECT *,
           RANK() OVER (
               PARTITION BY location
               ORDER BY total_engagement DESC
           ) AS location_rank
    FROM user_engagement
)
SELECT location,
       location_rank,
       user_id,
       follower_count,
       post_count,
       total_engagement
FROM ranked
WHERE location_rank <= 3
ORDER BY location, location_rank;
```

Use `RANK()`, not `ROW_NUMBER()`.

Users with zero posts are excluded by the INNER JOIN; document this.

If ties occur at rank 3, allow extra rows and report them.

## Files to Create

Create:

```text
sql/
├── E3_platform_avg_engagement.sql
├── M1_location_engagement.sql
└── H2_rank_users_by_location.sql

src/
├── 06_build_sql_database.py
├── 07_run_phase2_queries.py
└── 08_validate_phase2_results.py

data/
└── datavortex.db

reports/phase2/
├── E3_platform_avg_engagement.md
├── M1_location_engagement.md
├── H2_rank_users_by_location.md
└── phase2_validation.md
```

## Script Requirements

### 06_build_sql_database.py

* Load cleaned CSVs only.
* Reuse paths from `src/utils.py`.
* Build `data/datavortex.db`.
* Create `users` and `posts` tables.
* Make it re-runnable/idempotent.

### 07_run_phase2_queries.py

* Execute all three `.sql` files.
* Print clean, readable actual result tables.
* Save actual results to the three Phase 2 Markdown reports.
* Add short plain-English interpretations.
* Never fabricate results.

### 08_validate_phase2_results.py

Independently validate using pandas.

Check:

**E3**

* NULL-platform exclusion
* post count
* platform averages
* reconciliation with Phase 1 Insight #2

**M1**

* location count
* total post count
* independent pandas top-3 comparison

**H2**

* rank ordering
* duplicate user/location integrity
* top user cross-check
* locations with fewer than 3 active users
* rank-3 ties

Write PASS/FAIL and actual values to `reports/phase2/phase2_validation.md`.

## README / STATE

Append Phase 2 information to `README.md` and `STATE.md`.

Do not overwrite existing Phase 1 content.

## Restrictions

* Do not modify `data/raw/`.
* Do not modify `data/cleaned/`.
* Do not modify `notebooks/`.
* Do not modify existing Phase 1 scripts.
* Do not modify existing Phase 1 reports.
* Do not create PDFs.
* Do not create screenshots.
* Do not invent data or columns.

## Final Requirement

Run everything end-to-end:

```text
06 → 07 → 08
```

Fix any errors and rerun.

At the end, show:

1. Files created
2. Files modified
3. PASS/FAIL validation results
4. Actual E3, M1 and H2 output tables
5. Exact screenshot instructions

I will personally take the JPEG screenshots.
