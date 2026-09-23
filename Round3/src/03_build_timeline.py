"""
Round3/src/03_build_timeline.py
Phase 3: Sentiment Timeline & Engagement Analysis

Builds an hourly UTC time series from the enriched dataset.
Calculates post volume, total engagement, and sentiment percentages.
Preserves original source timestamps for all records, including future-dated anomalies.
"""

import sys
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
import dateutil.parser
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROUND3_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROUND3_DIR))
import config as cfg

def parse_dt(ts_str):
    if not ts_str:
        return None
    try:
        return dateutil.parser.isoparse(ts_str).astimezone(timezone.utc)
    except Exception:
        return None

def main():
    print("=================================================================")
    print("DATA VORTEX A'26 Round 3 — Build Timeline")
    print("=================================================================")

    in_csv = cfg.PROCESSED_POSTS_CSV
    print(f"[1/4] Loading enriched dataset: {in_csv.name}...")
    if not in_csv.exists():
        print("  [ERROR] Enriched dataset not found. Run 02_process_and_predict.py first.")
        sys.exit(1)

    rows = []
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  [OK] Loaded {len(rows)} posts.")

    # Parse and validate timestamps
    print("[2/4] Validating timestamps...")
    valid_posts = []
    anomalies_future = 0

    for r in rows:
        created_at = parse_dt(r.get("created_at"))
        collected_at = parse_dt(r.get("collected_at"))
        
        if not created_at or not collected_at:
            continue
            
        # Detect future-dated anomaly and count it, but do NOT modify the raw timestamp
        if created_at > collected_at:
            anomalies_future += 1

        # Calculate total engagement
        try:
            likes = int(r.get("like_count", 0) or 0)
            reposts = int(r.get("repost_count", 0) or 0)
            replies = int(r.get("reply_count", 0) or 0)
            quotes = int(r.get("quote_count", 0) or 0)
            engagement = likes + reposts + replies + quotes
        except ValueError:
            engagement = 0
            
        valid_posts.append({
            "ts": created_at,
            "sentiment": r.get("sentiment_label", "Neutral"),
            "engagement": engagement
        })

    if not valid_posts:
        print("  [ERROR] No valid posts with timestamps found.")
        sys.exit(1)

    # Print bounds
    timestamps = [p["ts"] for p in valid_posts]
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    print(f"  [INFO] Actual min created_at: {min_ts.isoformat()}")
    print(f"  [INFO] Actual max created_at: {max_ts.isoformat()}")
    if anomalies_future > 0:
        print(f"  [WARN] Found {anomalies_future} future-dated anomalie(s). Timestamps left unaltered.")

    # Group into hourly buckets
    print("[3/4] Building hourly buckets...")
    
    # floor to hour
    start_hour = min_ts.replace(minute=0, second=0, microsecond=0)
    end_hour = max_ts.replace(minute=0, second=0, microsecond=0)
    
    # Initialize all hours in range with 0s
    buckets = {}
    curr = start_hour
    while curr <= end_hour:
        buckets[curr] = {
            "post_count": 0,
            "total_engagement": 0,
            "negative_cnt": 0,
            "neutral_cnt": 0,
            "positive_cnt": 0
        }
        curr += timedelta(hours=1)

    # Fill buckets
    for p in valid_posts:
        hour_key = p["ts"].replace(minute=0, second=0, microsecond=0)
        buckets[hour_key]["post_count"] += 1
        buckets[hour_key]["total_engagement"] += p["engagement"]
        
        lbl = p["sentiment"]
        if lbl == "Negative":
            buckets[hour_key]["negative_cnt"] += 1
        elif lbl == "Positive":
            buckets[hour_key]["positive_cnt"] += 1
        else:
            buckets[hour_key]["neutral_cnt"] += 1

    # Calculate percentages and averages
    out_rows = []
    for h in sorted(buckets.keys()):
        b = buckets[h]
        pc = b["post_count"]
        if pc > 0:
            avg_eng = round(b["total_engagement"] / pc, 2)
            pct_neg = round((b["negative_cnt"] / pc) * 100, 2)
            pct_neu = round((b["neutral_cnt"] / pc) * 100, 2)
            pct_pos = round((b["positive_cnt"] / pc) * 100, 2)
        else:
            avg_eng = 0.0
            pct_neg = 0.0
            pct_neu = 0.0
            pct_pos = 0.0
            
        out_rows.append({
            "hour": h.isoformat(),
            "post_count": pc,
            "total_engagement": b["total_engagement"],
            "avg_engagement": avg_eng,
            "pct_negative": pct_neg,
            "pct_neutral": pct_neu,
            "pct_positive": pct_pos
        })

    # Save
    out_csv = cfg.TIMELINE_CSV
    print(f"[4/4] Saving timeline: {out_csv.name}...")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "hour", "post_count", "total_engagement", "avg_engagement", 
            "pct_negative", "pct_neutral", "pct_positive"
        ])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"  [OK] Saved {len(out_rows)} hourly buckets.")
    print("=================================================================")

if __name__ == "__main__":
    main()
