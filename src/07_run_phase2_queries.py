"""
07_run_phase2_queries.py
Executes E3, M1, and H2 SQL queries against data/datavortex.db and writes
readable Markdown reports to reports/phase2/.

Run from the project root:
    python src/07_run_phase2_queries.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import sqlite3
import pandas as pd

DB_PATH      = "data/datavortex.db"
SQL_DIR      = "sql"
REPORT_DIR   = "reports/phase2"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_sql(name: str) -> str:
    path = os.path.join(SQL_DIR, name)
    with open(path) as f:
        return f.read()


def run_query(con: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, con)


def df_to_md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-Flavored Markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:,.2f}" if not pd.isna(v) else "")
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v) if v is not None else "")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_report(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  -> {path}")


# ── E3 ───────────────────────────────────────────────────────────────────────

def run_e3(con):
    print("\n[E3] Average Engagement by Platform")
    sql = load_sql("E3_platform_avg_engagement.sql")
    df  = run_query(con, sql)
    print(df.to_string(index=False))

    lines = [
        "# E3 — Average Engagement by Platform\n",
        "**Query:** `sql/E3_platform_avg_engagement.sql`  ",
        "**Database:** `data/datavortex.db`  ",
        f"**Rows returned:** {len(df)} (one per non-NULL platform)\n",
        "## Results\n",
        df_to_md_table(df),
        "\n## Interpretation\n",
        "- `engagement = COALESCE(likes, 0) + shares + comments` (global rule).",
        "- Posts with a NULL platform tag (1,784 rows) are excluded; the remaining "
        f"{df['post_count'].sum():,} posts span {len(df)} platforms.",
        "- The spread in `avg_total_engagement` across all platforms is "
        f"**{(df['avg_total_engagement'].max() - df['avg_total_engagement'].min()):.1f} units "
        f"({(df['avg_total_engagement'].max() - df['avg_total_engagement'].min()) / df['avg_total_engagement'].min() * 100:.1f}%)** — "
        "consistent with Phase 1 Insight #2 (platform is a weak differentiator).",
        "- `avg_likes` is computed over non-NULL likes rows only (SQLite AVG ignores NULLs), "
        "matching the Phase 1 methodology.",
        "",
    ]
    write_report(os.path.join(REPORT_DIR, "E3_platform_avg_engagement.md"), "\n".join(lines))
    return df


# ── M1 ───────────────────────────────────────────────────────────────────────

def run_m1(con):
    print("\n[M1] Locations by Total Engagement")
    sql = load_sql("M1_location_engagement.sql")
    df  = run_query(con, sql)
    print(df.to_string(index=False))

    top3 = df.head(3)
    lines = [
        "# M1 — Which Locations Generate the Most Engagement\n",
        "**Query:** `sql/M1_location_engagement.sql`  ",
        "**Database:** `data/datavortex.db`  ",
        f"**Rows returned:** {len(df)} (one per distinct location)\n",
        "## Results\n",
        df_to_md_table(df),
        "\n## Interpretation\n",
        f"- {len(df)} distinct locations found across {df['post_count'].sum():,} posts.",
        "- `total_engagement` uses `COALESCE(likes, 0)` so missing likes are treated "
        "as 0 (conservative lower bound — not imputed).",
        "- Top 3 locations by total engagement:",
    ]
    for _, row in top3.iterrows():
        lines.append(
            f"  1. **{row['location']}** — {int(row['total_engagement']):,} total engagement "
            f"across {int(row['post_count']):,} posts (avg {row['avg_engagement_per_post']:,.1f}/post)"
        )
    lines.append(
        "- Locations with fewer posts naturally accumulate less total engagement, "
        "so `avg_engagement_per_post` is the fairer cross-location comparator."
    )
    lines.append("")
    write_report(os.path.join(REPORT_DIR, "M1_location_engagement.md"), "\n".join(lines))
    return df


# ── H2 ───────────────────────────────────────────────────────────────────────

def run_h2(con):
    print("\n[H2] Top-3 Users by Engagement Within Each Location")
    sql = load_sql("H2_rank_users_by_location.sql")
    df  = run_query(con, sql)
    print(df.to_string(index=False))

    n_locs  = df['location'].nunique()
    tie_locs = (df.groupby('location').size() > 3).sum()

    lines = [
        "# H2 — Rank Users Within Their Location\n",
        "**Query:** `sql/H2_rank_users_by_location.sql`  ",
        "**Database:** `data/datavortex.db`  ",
        f"**Rows returned:** {len(df)} across {n_locs} locations\n",
        "## Notes\n",
        "- `RANK()` is used (not `ROW_NUMBER()`), so tied users at rank 3 both appear.",
        "- Users with zero posts are excluded by the INNER JOIN in the CTE — "
        "none are expected given 100% `user_id` resolution in Phase 1 validation.",
        f"- Locations with more than 3 rows due to rank-3 ties: **{tie_locs}**.\n",
        "## Results\n",
        df_to_md_table(df),
        "\n## Interpretation\n",
        "- Rankings are based on `total_engagement = SUM(COALESCE(likes, 0) + shares + comments)` "
        "per user within each location.",
        "- `follower_count` is shown for reference; as established in Phase 1 Insight #1, "
        "it carries no engagement signal (r = −0.011).",
        "- The top-ranked user per location can be used to seed a 'local influencer' "
        "feature for a rebuilt recommendation engine.",
        "",
    ]
    write_report(os.path.join(REPORT_DIR, "H2_rank_users_by_location.md"), "\n".join(lines))
    return df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Run src/06_build_sql_database.py first.")
        sys.exit(1)

    con = sqlite3.connect(DB_PATH)
    try:
        e3 = run_e3(con)
        m1 = run_m1(con)
        h2 = run_h2(con)
    finally:
        con.close()

    print("\nAll Phase 2 reports written to reports/phase2/")
    print(f"  E3 rows: {len(e3)}   M1 rows: {len(m1)}   H2 rows: {len(h2)}")


if __name__ == "__main__":
    main()
