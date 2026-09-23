"""
Round3/src/07_build_report_figures.py
Phase 6: Build Report Figures

Generates:
1. Figure 1: Sentiment + Activity Timeline
2. Figure 2: Event Topic Analysis
"""

import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import dateutil.parser

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROUND3_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROUND3_DIR))
import config as cfg

def main():
    print("=================================================================")
    print("DATA VORTEX A'26 Round 3 — Build Report Figures")
    print("=================================================================")

    # Setup directories
    reports_dir = _ROUND3_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timeline_csv = cfg.TIMELINE_CSV
    events_csv = _ROUND3_DIR / "data" / "processed" / "flagged_events.csv"
    topics_csv = _ROUND3_DIR / "data" / "processed" / "topic_entity_by_event.csv"

    # 1. Read Timeline
    hours = []
    post_counts = []
    pct_pos = []
    pct_neg = []
    pct_neu = []

    if timeline_csv.exists():
        with open(timeline_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hours.append(dateutil.parser.isoparse(row["hour"]))
                post_counts.append(int(row["post_count"]))
                pct_pos.append(float(row["pct_positive"]))
                pct_neg.append(float(row["pct_negative"]))
                pct_neu.append(float(row["pct_neutral"]))

    # 2. Read Events
    events = []
    if events_csv.exists():
        with open(events_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                events.append({
                    "id": row["event_id"],
                    "timestamp": dateutil.parser.isoparse(row["timestamp"]),
                    "type": row["event_type"]
                })

    # --- Figure 1: Sentiment + Activity Timeline ---
    if hours:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Activity volume (bars)
        ax1.bar(hours, post_counts, width=0.03, color='lightgray', label="Post Volume (Activity)", alpha=0.6)
        ax1.set_xlabel("Time (UTC)", fontsize=12)
        ax1.set_ylabel("Post Volume", fontsize=12)
        ax1.tick_params(axis='y', labelcolor='gray')

        # Sentiment (lines)
        ax2 = ax1.twinx()
        ax2.plot(hours, pct_pos, color='green', marker='o', label="% Positive", linewidth=2)
        ax2.plot(hours, pct_neg, color='red', marker='x', label="% Negative", linewidth=2)
        ax2.plot(hours, pct_neu, color='gray', marker='.', label="% Neutral", linewidth=1, linestyle='--')
        ax2.set_ylabel("Sentiment (%)", fontsize=12)
        ax2.set_ylim(0, 100)

        # Mark events
        for ev in events:
            color = 'blue' if ev["type"] == 'sentiment_shift' else 'orange'
            ax1.axvline(x=ev["timestamp"], color=color, linestyle='--', alpha=0.8)
            ax1.text(ev["timestamp"], max(post_counts)*0.9, ev["id"], rotation=90, verticalalignment='bottom', color=color, fontsize=9)

        # Formatting
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
        fig.autofmt_xdate()
        
        # Legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.title("Figure 1: Sentiment & Activity Timeline (Round 3 Bluesky Data)", fontsize=14)
        plt.tight_layout()
        f1_path = reports_dir / "figure1_timeline.png"
        plt.savefig(f1_path, dpi=300)
        plt.close()
        print(f"  [OK] Saved {f1_path.name}")

    # --- Figure 2: Event Topic Analysis ---
    # We will pick EVT_003 (Weak explanation) as the most informative event
    if topics_csv.exists():
        with open(topics_csv, "r", encoding="utf-8") as f:
            topics_data = list(csv.DictReader(f))
            
        evt003 = next((r for r in topics_data if r["event_id"] == "EVT_003"), None)
        if evt003:
            curr = evt003["current_topics"].split(", ")
            prev = evt003["previous_topics"].split(", ")
            
            # Simple bar chart comparing rank or presence
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # For visualization, we just list them out since TF-IDF values aren't saved
            # We'll do a simple text-based diagram in matplotlib
            ax.axis('off')
            ax.text(0.1, 0.8, "Event: EVT_003 (Sentiment Shift @ 02:00 UTC)", fontsize=14, fontweight='bold')
            ax.text(0.1, 0.6, "Previous Hour Topics:", fontsize=12, fontweight='bold', color='gray')
            ax.text(0.1, 0.5, ", ".join(prev), fontsize=12)
            
            ax.text(0.1, 0.3, "Event Hour Topics:", fontsize=12, fontweight='bold', color='blue')
            ax.text(0.1, 0.2, ", ".join(curr), fontsize=12, color='blue')
            
            ax.text(0.1, 0.05, "*Note: Figure displays rank-ordered TF-IDF terms. Does not imply causation.", fontsize=9, style='italic')
            
            plt.title("Figure 2: Event Topic Analysis Comparison", fontsize=14)
            plt.tight_layout()
            f2_path = reports_dir / "figure2_topics.png"
            plt.savefig(f2_path, dpi=300)
            plt.close()
            print(f"  [OK] Saved {f2_path.name}")

    print("=================================================================")

if __name__ == "__main__":
    main()
