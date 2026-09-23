"""
Round3/src/04_detect_events.py
Phase 4: Event Detection

Scans the hourly timeline for sentiment shifts and engagement spikes.
Hard-floor thresholds: 10pp for sentiment, mean+1std for engagement.
Saves results to flagged_events.csv
"""

import sys
import csv
import math
import statistics
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROUND3_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROUND3_DIR))
import config as cfg

def main():
    print("=================================================================")
    print("DATA VORTEX A'26 Round 3 — Detect Events")
    print("=================================================================")

    in_csv = cfg.TIMELINE_CSV
    if not in_csv.exists():
        print("  [ERROR] Timeline CSV not found.")
        sys.exit(1)

    with open(in_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("  [ERROR] Timeline is empty.")
        sys.exit(1)

    # Convert types and calculate polarity
    for r in rows:
        r["post_count"] = int(r["post_count"])
        r["total_engagement"] = int(float(r.get("total_engagement", 0) or 0))
        r["avg_engagement"] = float(r.get("avg_engagement", 0) or 0)
        r["pct_negative"] = float(r["pct_negative"])
        r["pct_neutral"] = float(r["pct_neutral"])
        r["pct_positive"] = float(r["pct_positive"])
        r["polarity"] = r["pct_positive"] - r["pct_negative"]

    # Engagement spike stats — computed on total_engagement (likes+reposts+replies+quotes)
    # per bucket, NOT on post_count.
    engagements = [r["total_engagement"] for r in rows]
    mean_eng = statistics.mean(engagements)
    std_eng = statistics.stdev(engagements) if len(engagements) > 1 else 0

    # Post-count stats are reported for context only
    post_counts = [r["post_count"] for r in rows]
    mean_count = statistics.mean(post_counts)
    std_count = statistics.stdev(post_counts) if len(post_counts) > 1 else 0

    print(f"  [INFO] Timeline stats: {len(rows)} buckets")
    print(f"  [INFO] Post Count   — Mean: {mean_count:.2f}, StdDev: {std_count:.2f}")
    print(f"  [INFO] Engagement   — Mean: {mean_eng:.2f}, StdDev: {std_eng:.2f}")

    # Event detection thresholds
    shift_thresh = 15.0
    shift_floor = 10.0
    # Spike threshold is now on TOTAL ENGAGEMENT (likes+reposts+replies+quotes)
    spike_thresh = mean_eng + (2 * std_eng)
    spike_floor  = mean_eng + (1 * std_eng)

    events = []
    event_id_counter = 1

    # First pass: try strict thresholds
    shifts = []
    spikes = []

    def detect_events(s_thresh, sp_thresh):
        res_shifts = []
        res_spikes = []
        for i, r in enumerate(rows):
            # Engagement spikes — metric is total_engagement (likes+reposts+replies+quotes)
            eng = r["total_engagement"]
            if eng >= sp_thresh:
                z = (eng - mean_eng) / std_eng if std_eng > 0 else 0
                prev = rows[i-1] if i > 0 else {}
                res_spikes.append({
                    "timestamp": r["hour"],
                    "event_type": "engagement_spike",
                    "metric_value": eng,          # total_engagement, NOT post_count
                    "delta_or_zscore": round(z, 4),
                    "threshold": round(sp_thresh, 4),
                    "bucket_size": "1 hour",
                    "previous_timestamp": prev.get("hour", ""),
                    "post_count": r["post_count"],
                    "pct_negative": r["pct_negative"],
                    "pct_neutral": r["pct_neutral"],
                    "pct_positive": r["pct_positive"],
                    "polarity": r["polarity"]
                })
            
            # Shifts
            if i > 0:
                prev = rows[i-1]
                # Only consider shifts if there's actual data in both buckets
                if r["post_count"] > 0 and prev["post_count"] > 0:
                    delta = r["polarity"] - prev["polarity"]
                    if abs(delta) >= s_thresh:
                        res_shifts.append({
                            "timestamp": r["hour"],
                            "event_type": "sentiment_shift",
                            "metric_value": r["polarity"],
                            "delta_or_zscore": delta,
                            "threshold": s_thresh,
                            "bucket_size": "1 hour",
                            "previous_timestamp": prev["hour"],
                            "post_count": r["post_count"],
                            "pct_negative": r["pct_negative"],
                            "pct_neutral": r["pct_neutral"],
                            "pct_positive": r["pct_positive"],
                            "polarity": r["polarity"]
                        })
        return res_shifts, res_spikes

    shifts, spikes = detect_events(shift_thresh, spike_thresh)
    print(f"  [INFO] Initial pass (thresh=15.0pp, mean+2std): {len(shifts)} shifts, {len(spikes)} spikes.")

    # Threshold adjustment if 0 events found
    if len(shifts) == 0:
        print(f"  [WARN] No sentiment shifts found at >=15pp. Adjusting to hard floor >=10pp.")
        shift_thresh = shift_floor
        shifts, _ = detect_events(shift_thresh, spike_thresh)

    if len(spikes) == 0:
        print(f"  [WARN] No engagement spikes found at mean+2std. Adjusting to hard floor mean+1std.")
        spike_thresh = spike_floor
        _, spikes = detect_events(shift_thresh, spike_thresh)

    print(f"  [OK] Final detection: {len(shifts)} shifts, {len(spikes)} spikes.")

    # Combine and add event IDs
    all_events = shifts + spikes
    
    # Sort chronologically
    all_events.sort(key=lambda x: x["timestamp"])

    for ev in all_events:
        ev["event_id"] = f"EVT_{event_id_counter:03d}"
        event_id_counter += 1

    out_csv = _ROUND3_DIR / "data" / "processed" / "flagged_events.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    fields = [
        "event_id", "timestamp", "event_type", "metric_value", "delta_or_zscore", 
        "threshold", "bucket_size", "previous_timestamp", "post_count", 
        "pct_negative", "pct_neutral", "pct_positive", "polarity"
    ]
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_events)
    
    print(f"  [OK] Wrote {len(all_events)} events to {out_csv.name}")

    # Write summary for next script
    with open(_ROUND3_DIR / "scratch" / "phase4_thresholds.json", "w") as f:
        import json
        json.dump({
            "shift_threshold_used": shift_thresh,
            "spike_threshold_used": round(spike_thresh, 4),
            "spike_metric": "total_engagement (likes+reposts+replies+quotes)",
            "mean_engagement": round(mean_eng, 4),
            "std_engagement": round(std_eng, 4),
            "mean_post_count": round(mean_count, 4),
            "std_post_count": round(std_count, 4),
            "num_shifts": len(shifts),
            "num_spikes": len(spikes)
        }, f)

    print("=================================================================")

if __name__ == "__main__":
    main()
