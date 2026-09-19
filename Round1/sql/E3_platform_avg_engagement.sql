-- E3: Average Engagement by Platform
-- engagement = COALESCE(likes, 0) + shares + comments (global rule)
-- NULL-platform rows are intentionally excluded (1,784 rows missing platform tag).
-- Expected: 5 rows (Facebook, Instagram, Reddit, Twitter, YouTube).

SELECT platform,
       COUNT(*)                                             AS post_count,
       AVG(likes)                                          AS avg_likes,
       AVG(shares)                                         AS avg_shares,
       AVG(comments)                                       AS avg_comments,
       AVG(COALESCE(likes, 0) + shares + comments)        AS avg_total_engagement
FROM posts
WHERE platform IS NOT NULL
GROUP BY platform
ORDER BY avg_total_engagement DESC;
