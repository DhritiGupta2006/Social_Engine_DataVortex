-- M1: Which Locations Generate the Most Engagement
-- engagement = COALESCE(likes, 0) + shares + comments (global rule)
-- INNER JOIN: users with zero posts are excluded (none expected given 100% user_id resolution).
-- Expected: 33 distinct locations, 12,000 total posts.

SELECT u.location,
       COUNT(p.post_id)                                                        AS post_count,
       SUM(COALESCE(p.likes, 0) + p.shares + p.comments)                      AS total_engagement,
       ROUND(AVG(COALESCE(p.likes, 0) + p.shares + p.comments), 1)            AS avg_engagement_per_post
FROM users u
JOIN posts p ON u.user_id = p.user_id
GROUP BY u.location
ORDER BY total_engagement DESC;
