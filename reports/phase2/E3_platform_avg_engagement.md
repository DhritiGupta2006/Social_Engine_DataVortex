# E3 — Average Engagement by Platform

**Query:** `sql/E3_platform_avg_engagement.sql`  
**Database:** `data/datavortex.db`  
**Rows returned:** 5 (one per non-NULL platform)

## Results

| platform | post_count | avg_likes | avg_shares | avg_comments | avg_total_engagement |
| --- | --- | --- | --- | --- | --- |
| Instagram | 1,989 | 2,500.93 | 1,040.84 | 499.80 | 3,669.38 |
| Reddit | 2,031 | 2,488.11 | 1,002.23 | 511.18 | 3,647.47 |
| YouTube | 2,073 | 2,517.77 | 1,011.83 | 504.38 | 3,638.03 |
| Facebook | 2,074 | 2,528.86 | 984.17 | 506.94 | 3,631.00 |
| Twitter | 2,049 | 2,437.69 | 1,005.39 | 506.13 | 3,563.75 |

## Interpretation

- `engagement = COALESCE(likes, 0) + shares + comments` (global rule).
- Posts with a NULL platform tag (1,784 rows) are excluded; the remaining 10,216 posts span 5 platforms.
- The spread in `avg_total_engagement` across all platforms is **105.6 units (3.0%)** — consistent with Phase 1 Insight #2 (platform is a weak differentiator).
- `avg_likes` is computed over non-NULL likes rows only (SQLite AVG ignores NULLs), matching the Phase 1 methodology.
