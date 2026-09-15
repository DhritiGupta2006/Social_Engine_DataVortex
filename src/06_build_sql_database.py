"""
06_build_sql_database.py
Loads the Phase 1 cleaned CSVs and builds data/datavortex.db (SQLite).
Creates 'users' and 'posts' tables.  Re-runnable / idempotent: tables are
dropped and recreated each time so the DB always reflects the current
cleaned CSVs without manual cleanup.

Run from the project root:
    python src/06_build_sql_database.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import sqlite3
import pandas as pd
from utils import CLEAN_USERS_PATH, CLEAN_POSTS_PATH

DB_PATH = "data/datavortex.db"


def build_database():
    # ── 1. Load cleaned CSVs ────────────────────────────────────────────────
    print("Loading cleaned CSVs …")
    users = pd.read_csv(CLEAN_USERS_PATH)
    posts = pd.read_csv(CLEAN_POSTS_PATH)
    print(f"  users: {len(users):,} rows × {len(users.columns)} columns")
    print(f"  posts: {len(posts):,} rows × {len(posts.columns)} columns")

    # ── 2. Connect / create DB ──────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # ── 3. Drop + recreate tables (idempotent) ──────────────────────────────
    cur.executescript("""
        DROP TABLE IF EXISTS posts;
        DROP TABLE IF EXISTS users;

        CREATE TABLE users (
            user_id         TEXT PRIMARY KEY,
            location        TEXT,
            language        TEXT,
            account_created TEXT,
            follower_count  INTEGER
        );

        CREATE TABLE posts (
            post_id       TEXT PRIMARY KEY,
            user_id       TEXT,
            platform      TEXT,
            text_content  TEXT,
            timestamp_raw TEXT,
            event_time    TEXT,
            likes         REAL,
            shares        INTEGER,
            comments      INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE INDEX idx_posts_user_id  ON posts(user_id);
        CREATE INDEX idx_posts_platform ON posts(platform);
    """)

    # ── 4. Insert data ──────────────────────────────────────────────────────
    print("Inserting users …")
    users.to_sql("users", con, if_exists="append", index=False)

    print("Inserting posts …")
    posts.to_sql("posts", con, if_exists="append", index=False)

    con.commit()

    # ── 5. Quick sanity check ───────────────────────────────────────────────
    n_users = cur.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    n_posts = cur.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    con.close()

    print(f"\nDatabase written to {DB_PATH}")
    print(f"  users table: {n_users:,} rows")
    print(f"  posts table: {n_posts:,} rows")

    assert n_users == len(users), f"Row count mismatch in users: {n_users} vs {len(users)}"
    assert n_posts == len(posts), f"Row count mismatch in posts: {n_posts} vs {len(posts)}"
    print("Row-count assertions passed [OK]")


if __name__ == "__main__":
    build_database()
