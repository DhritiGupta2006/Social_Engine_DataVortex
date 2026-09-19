"""
04_eda_and_visuals.py
Runs exploratory data analysis on the cleaned data and saves figures to
reports/figures/. All numbers printed here are computed directly from
data/cleaned/*.csv — nothing is hardcoded.
Run from the project root: python src/04_eda_and_visuals.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import CLEAN_USERS_PATH, CLEAN_POSTS_PATH

FIG_DIR = "reports/figures"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                      "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    users = pd.read_csv(CLEAN_USERS_PATH, parse_dates=['account_created'])
    posts = pd.read_csv(CLEAN_POSTS_PATH, parse_dates=['event_time'])
    posts['engagement'] = posts[['likes', 'shares', 'comments']].sum(axis=1, min_count=1)
    merged = posts.merge(users, on='user_id', how='left')

    # 1. Post volume per platform
    fig, ax = plt.subplots(figsize=(6, 4))
    order = posts['platform'].value_counts()
    order.plot(kind='bar', ax=ax, color='#4C72B0')
    ax.set_title("Post volume by platform (missing platform excluded)")
    ax.set_ylabel("number of posts")
    save(fig, "01_posts_by_platform.png")
    print("Posts by platform:\n", order, "\n")

    # 2. Engagement metric distributions
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ['likes', 'shares', 'comments']):
        posts[col].dropna().plot(kind='hist', bins=40, ax=ax, color='#55A868')
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
    fig.tight_layout()
    save(fig, "02_engagement_distributions.png")

    # 3. Monthly post volume trend
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly = posts.set_index('event_time').resample('MS').size()
    monthly.plot(ax=ax, marker='o', color='#C44E52')
    ax.set_title("Post volume over time (monthly)")
    ax.set_ylabel("number of posts")
    save(fig, "03_monthly_post_volume.png")

    # 4. Mean engagement by platform
    fig, ax = plt.subplots(figsize=(7, 4))
    plat_means = posts.groupby('platform')[['likes', 'shares', 'comments']].mean()
    plat_means.plot(kind='bar', ax=ax)
    ax.set_title("Mean engagement per post, by platform")
    ax.set_ylabel("mean value")
    fig.tight_layout()
    save(fig, "04_engagement_by_platform.png")
    print("Mean engagement by platform:\n", plat_means, "\n")

    # 5. Follower count vs total engagement (scatter)
    fig, ax = plt.subplots(figsize=(6, 5))
    sample = merged.dropna(subset=['follower_count', 'engagement']).sample(
        n=min(3000, merged.dropna(subset=['follower_count', 'engagement']).shape[0]),
        random_state=42)
    ax.scatter(sample['follower_count'], sample['engagement'], alpha=0.15, s=10, color='#8172B2')
    ax.set_xlabel("author follower_count")
    ax.set_ylabel("post engagement (likes+shares+comments)")
    ax.set_title("Follower count vs. post engagement")
    save(fig, "05_followers_vs_engagement.png")
    corr = merged[['follower_count', 'engagement']].corr().iloc[0, 1]
    print(f"Correlation(follower_count, engagement) = {corr:.4f}\n")

    # 6. Top hashtags
    tags = posts['text_content'].dropna().str.findall(r'#(\w+)')
    from collections import Counter
    counts = Counter(t for lst in tags for t in lst)
    top15 = pd.Series(dict(counts.most_common(15))).sort_values()
    fig, ax = plt.subplots(figsize=(6, 6))
    top15.plot(kind='barh', ax=ax, color='#DD8452')
    ax.set_title("Top 15 hashtags")
    ax.set_xlabel("occurrences")
    save(fig, "06_top_hashtags.png")

    # 7. Users by language
    fig, ax = plt.subplots(figsize=(6, 4))
    users['language'].value_counts().sort_values().plot(kind='barh', ax=ax, color='#64B5CD')
    ax.set_title("Registered users by language code")
    save(fig, "07_users_by_language.png")

    # 8. Missingness heatmap-style bar (data quality overview post-cleaning)
    fig, ax = plt.subplots(figsize=(6, 4))
    miss = posts[['platform', 'text_content', 'likes']].isna().mean() * 100
    miss.plot(kind='bar', ax=ax, color='#937860')
    ax.set_ylabel("% missing (retained as NaN)")
    ax.set_title("Residual missingness in cleaned posts data")
    fig.tight_layout()
    save(fig, "08_residual_missingness.png")

    print("EDA complete. Figures saved to reports/figures/.")


if __name__ == "__main__":
    main()
