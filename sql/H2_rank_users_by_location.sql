-- H2: Rank Users Within Their Location by Total Engagement
-- engagement = COALESCE(likes, 0) + shares + comments (global rule)
-- Uses RANK() (not ROW_NUMBER()) so ties at rank 3 yield extra rows.
-- INNER JOIN excludes users with zero posts; this is intentional and documented.
-- Expected: ~99 rows (top-3 per location × 33 locations), ties may produce more.

WITH user_engagement AS (
    SELECT u.user_id,
           u.location,
           u.follower_count,
           COUNT(p.post_id)                                              AS post_count,
           SUM(COALESCE(p.likes, 0) + p.shares + p.comments)           AS total_engagement
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
