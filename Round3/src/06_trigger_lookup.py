"""
Round3/src/06_trigger_lookup.py
Phase 5: Trigger Investigation

Investigates every flagged event individually by searching an independent news API
(HackerNews Algolia API) for the product and strongest event-specific topics.
Search window: 24h before to 6h after the event.
Classifies findings as Strong, Weak, or Unexplained.
"""

import sys
import csv
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import dateutil.parser

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROUND3_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROUND3_DIR))
import config as cfg

def get_hn_news(query, start_dt, end_dt):
    """
    Search HackerNews Algolia API.
    query: str
    start_dt, end_dt: datetime objects (UTC)
    """
    url = "http://hn.algolia.com/api/v1/search_by_date"
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    
    # We search for 'story' type to get news articles/links
    params = {
        "query": query,
        "tags": "story",
        "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts}",
        "hitsPerPage": 20
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("hits", [])
    except Exception as e:
        print(f"  [ERROR] HN API request failed: {e}")
        return []

def main():
    print("=================================================================")
    print("DATA VORTEX A'26 Round 3 — Trigger Lookup")
    print("=================================================================")

    events_csv = _ROUND3_DIR / "data" / "processed" / "flagged_events.csv"
    topics_csv = _ROUND3_DIR / "data" / "processed" / "topic_entity_by_event.csv"

    if not events_csv.exists() or not topics_csv.exists():
        print("  [ERROR] Required phase 4 CSVs not found.")
        sys.exit(1)

    with open(events_csv, "r", encoding="utf-8") as f:
        events = list(csv.DictReader(f))
        
    with open(topics_csv, "r", encoding="utf-8") as f:
        topics_rows = list(csv.DictReader(f))
        topics_dict = {r["event_id"]: r for r in topics_rows}

    if not events:
        print("  [WARN] No events to investigate.")
        sys.exit(0)

    print(f"  [INFO] Investigating {len(events)} events using HackerNews API...")
    
    candidates_all = []
    log_entries = []
    table_rows = []

    for ev in events:
        eid = ev["event_id"]
        ts_str = ev["timestamp"]
        ev_type = ev["event_type"]
        ev_dt = dateutil.parser.isoparse(ts_str)
        
        # Determine search window (24h before to 6h after)
        start_dt = ev_dt - timedelta(hours=24)
        end_dt = ev_dt + timedelta(hours=6)
        
        t_row = topics_dict.get(eid, {})
        curr_topics = t_row.get("current_topics", "")
        
        # Build query
        # We always search for the product ("iPhone 17" or "iPhone") + top topic
        base_query = cfg.chosen_product
        top_topic = curr_topics.split(",")[0].strip() if curr_topics else ""
        if top_topic and top_topic.lower() not in base_query.lower():
            query = f"{base_query} {top_topic}"
        else:
            query = base_query
            
        print(f"  -> {eid} ({ev_type} @ {ts_str}): Query '{query}'")
        
        # Make API call
        hits = get_hn_news(query, start_dt, end_dt)
        time.sleep(0.5) # Polite delay
        
        best_candidate = None
        strength = "Unexplained"
        
        # Evaluate hits
        rel_candidates = []
        for hit in hits:
            pub_dt_str = hit.get("created_at")
            pub_dt = dateutil.parser.isoparse(pub_dt_str)
            title = hit.get("title", "")
            url = hit.get("url", "")
            
            # Basic relevance check
            if "iphone" not in title.lower() and "apple" not in title.lower():
                continue
                
            c_dict = {
                "event_id": eid,
                "event_timestamp": ts_str,
                "event_type": ev_type,
                "query": query,
                "title": title,
                "source": "HackerNews",
                "published_at": pub_dt_str,
                "url": url,
                "description": hit.get("story_text", "") or "",
                "search_window_start": start_dt.isoformat(),
                "search_window_end": end_dt.isoformat()
            }
            rel_candidates.append(c_dict)
            candidates_all.append(c_dict)
            
            # Classification
            if pub_dt <= ev_dt:
                # Pre-event or during-event
                # Mark Strong if it mentions the specific topic deeply or is a direct iPhone news piece
                if top_topic and top_topic.lower() in title.lower():
                    strength = "Strong"
                    best_candidate = c_dict
                elif strength != "Strong":
                    strength = "Weak"
                    best_candidate = c_dict
            else:
                # Post-event evidence cannot be Strong, and only Weak if nothing else exists
                if strength == "Unexplained":
                    # Doesn't explain it, but we can note it
                    best_candidate = c_dict

        # Logging
        pre_event_ct = sum(1 for c in rel_candidates if dateutil.parser.isoparse(c["published_at"]) <= ev_dt)
        post_event_ct = len(rel_candidates) - pre_event_ct
        
        log_entries.append({
            "event_id": eid,
            "queries": query,
            "window": f"{start_dt.isoformat()} to {end_dt.isoformat()}",
            "raw_results": len(hits),
            "relevant_candidates": len(rel_candidates),
            "timing": f"Pre/During: {pre_event_ct}, Post: {post_event_ct}",
            "strength": strength
        })
        
        # Table Row
        if strength == "Unexplained" or not best_candidate:
            table_rows.append(
                f"| {ev_type} | {ts_str} | {curr_topics} | No credible independent evidence found | N/A | Unexplained |"
            )
        else:
            title_escaped = best_candidate["title"].replace("|", " ")
            url = best_candidate["url"]
            evidence_link = f"[{title_escaped}]({url})" if url else title_escaped
            table_rows.append(
                f"| {ev_type} | {ts_str} | {curr_topics} | {evidence_link} | Published {best_candidate['published_at']} | {strength} |"
            )

    # 1. Save trigger_candidates.csv
    c_csv = _ROUND3_DIR / "data" / "processed" / "trigger_candidates.csv"
    c_csv.parent.mkdir(parents=True, exist_ok=True)
    if candidates_all:
        with open(c_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=candidates_all[0].keys())
            writer.writeheader()
            writer.writerows(candidates_all)
    else:
        # Empty file with headers
        with open(c_csv, "w", newline="", encoding="utf-8") as f:
            f.write("event_id,event_timestamp,event_type,query,title,source,published_at,url,description,search_window_start,search_window_end\n")
    print(f"  [OK] Saved {len(candidates_all)} candidates to {c_csv.name}")

    # 2. Save trigger_table.md
    t_md = _ROUND3_DIR / "reports" / "trigger_table.md"
    with open(t_md, "w", encoding="utf-8") as f:
        f.write("# Phase 5: Trigger Investigation Table\n\n")
        f.write("| What changed | When | Topics/entities | Candidate explanatory event | Independent evidence | Strength/limitation |\n")
        f.write("| ------------ | ---- | --------------- | --------------------------- | -------------------- | ------------------- |\n")
        for tr in table_rows:
            f.write(tr + "\n")
    print(f"  [OK] Saved trigger_table.md")

    # 3. Save trigger_search_log.md
    l_md = _ROUND3_DIR / "reports" / "trigger_search_log.md"
    with open(l_md, "w", encoding="utf-8") as f:
        f.write("# Phase 5: Trigger Search Log\n\n")
        for lg in log_entries:
            f.write(f"### {lg['event_id']}\n")
            f.write(f"- **Queries Used:** `{lg['queries']}`\n")
            f.write(f"- **Search Window:** {lg['window']}\n")
            f.write(f"- **Raw Results:** {lg['raw_results']}\n")
            f.write(f"- **Relevant Candidates:** {lg['relevant_candidates']}\n")
            f.write(f"- **Evidence Timing:** {lg['timing']}\n")
            f.write(f"- **Final Classification:** **{lg['strength']}**\n\n")
    print(f"  [OK] Saved trigger_search_log.md")
    
    print("=================================================================")

if __name__ == "__main__":
    main()
