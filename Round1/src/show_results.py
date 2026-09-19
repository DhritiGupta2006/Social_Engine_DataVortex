import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3, pandas as pd

con = sqlite3.connect('data/datavortex.db')

print('=== E3: Average Engagement by Platform ===')
e3 = pd.read_sql_query(
    "SELECT platform, COUNT(*) AS post_count, "
    "AVG(likes) AS avg_likes, AVG(shares) AS avg_shares, AVG(comments) AS avg_comments, "
    "AVG(COALESCE(likes,0)+shares+comments) AS avg_total_engagement "
    "FROM posts WHERE platform IS NOT NULL "
    "GROUP BY platform ORDER BY avg_total_engagement DESC", con)
print(e3.to_string(index=False))

print()
print('=== M1: Locations by Total Engagement (top 10 shown) ===')
m1 = pd.read_sql_query(
    "SELECT u.location, COUNT(p.post_id) AS post_count, "
    "SUM(COALESCE(p.likes,0)+p.shares+p.comments) AS total_engagement, "
    "ROUND(AVG(COALESCE(p.likes,0)+p.shares+p.comments),1) AS avg_engagement_per_post "
    "FROM users u JOIN posts p ON u.user_id=p.user_id "
    "GROUP BY u.location ORDER BY total_engagement DESC", con)
print(m1.head(10).to_string(index=False))
n_locs = len(m1)
print(f'... ({n_locs} total locations, {int(m1["post_count"].sum())} total posts)')

print()
print('=== H2: Top-3 Users per Location (first 15 rows shown) ===')
h2 = pd.read_sql_query(
    "WITH user_engagement AS ("
    "SELECT u.user_id, u.location, u.follower_count, COUNT(p.post_id) AS post_count, "
    "SUM(COALESCE(p.likes,0)+p.shares+p.comments) AS total_engagement "
    "FROM users u JOIN posts p ON u.user_id=p.user_id "
    "GROUP BY u.user_id, u.location, u.follower_count), "
    "ranked AS (SELECT *, RANK() OVER (PARTITION BY location ORDER BY total_engagement DESC) AS location_rank "
    "FROM user_engagement) "
    "SELECT location, location_rank, user_id, follower_count, post_count, total_engagement "
    "FROM ranked WHERE location_rank <= 3 ORDER BY location, location_rank", con)
print(h2.head(15).to_string(index=False))
n_h2 = len(h2)
n_locs_h2 = h2['location'].nunique()
print(f'... ({n_h2} total rows across {n_locs_h2} locations)')

con.close()
