"""
Round3/src/05_topic_entity_analysis.py
Phase 4: Topic & Entity Analysis (Fix 3 — Final)

Topic extraction:
  - Strips <url>, <amp>, http links and other preprocessing artifacts
    before TF-IDF so they cannot appear as topics.

Entity extraction:
  - Uses a curated regex dictionary of known product-launch entities
    (companies, products, people, platforms) relevant to the iPhone launch topic.
  - Fast, deterministic, no external NLP dependencies.
  - Only counts an entity if it genuinely appears in the post text.
  - Does NOT fabricate entities.

NLTK/spaCy were attempted but both hang in this environment; documented as limitation.
"""

import sys
import csv
import re
import json
from pathlib import Path
from collections import defaultdict
import dateutil.parser

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROUND3_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROUND3_DIR))
import config as cfg

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print("  [ERROR] scikit-learn not installed.")
    sys.exit(1)

# ── Artifact patterns to strip before TF-IDF ────────────────────────────────
# Removes: <url>, <amp>, bare http(s) links, @mentions, #hashtag-symbol,
#          isolated digits, and 1–2 char tokens.
_STRIP_RE = re.compile(
    r'<url>|<amp>|&amp;|https?://\S+|www\.\S+|(?<!\w)#(?=\S)|@\w+',
    re.IGNORECASE
)

# ── Curated entity dictionary ─────────────────────────────────────────────────
# Pattern: (display_name, compiled_regex)
# Entities are grouped into: Companies, Products, People, Platforms/OS
_ENTITIES = [
    # ── Companies / Brands ──────────────────────────────────────────────────
    ("Apple",         re.compile(r'\bApple\b', re.I)),
    ("Samsung",       re.compile(r'\bSamsung\b', re.I)),
    ("Google",        re.compile(r'\bGoogle\b', re.I)),
    ("Xiaomi",        re.compile(r'\bXiaomi\b', re.I)),
    ("Microsoft",     re.compile(r'\bMicrosoft\b', re.I)),
    ("Qualcomm",      re.compile(r'\bQualcomm\b', re.I)),
    ("TSMC",          re.compile(r'\bTSMC\b', re.I)),
    ("Amazon",        re.compile(r'\bAmazon\b', re.I)),
    ("Meta",          re.compile(r'\bMeta\b(?!\s+Quest)', re.I)),

    # ── iPhone / iPad Models ─────────────────────────────────────────────────
    ("iPhone 17",     re.compile(r'\biPhone\s*17\b', re.I)),
    ("iPhone 18",     re.compile(r'\biPhone\s*18\b', re.I)),
    ("iPhone Duo",    re.compile(r'\biPhone\s*Duo\b', re.I)),
    ("iPhone Pro",    re.compile(r'\biPhone\s*Pro\b', re.I)),
    ("iPhone Pro Max",re.compile(r'\biPhone\s*Pro\s*Max\b', re.I)),
    ("iPad",          re.compile(r'\biPad\b', re.I)),
    ("MacBook",       re.compile(r'\bMacBook\b', re.I)),
    ("Apple Watch",   re.compile(r'\bApple\s*Watch\b', re.I)),
    ("AirPods",       re.compile(r'\bAirPods\b', re.I)),
    ("Vision Pro",    re.compile(r'\bVision\s*Pro\b', re.I)),
    ("Apple Pencil",  re.compile(r'\bApple\s*Pencil\b', re.I)),

    # ── Operating Systems / Platforms ────────────────────────────────────────
    ("iOS 26",        re.compile(r'\biOS\s*26\b', re.I)),
    ("iOS 27",        re.compile(r'\biOS\s*27\b', re.I)),
    ("iOS",           re.compile(r'\biOS\b', re.I)),
    ("Android",       re.compile(r'\bAndroid\b', re.I)),
    ("App Store",     re.compile(r'\bApp\s*Store\b', re.I)),
    ("Bluesky",       re.compile(r'\bBluesky\b', re.I)),

    # ── Chips / Hardware ─────────────────────────────────────────────────────
    ("A19",           re.compile(r'\bA19\b', re.I)),
    ("A18",           re.compile(r'\bA18\b', re.I)),
    ("M6",            re.compile(r'\bM6\b')),
    ("M5",            re.compile(r'\bM5\b')),

    # ── People ───────────────────────────────────────────────────────────────
    ("Tim Cook",      re.compile(r'\bTim\s*Cook\b', re.I)),
    ("Craig Federighi", re.compile(r'\bFederighi\b', re.I)),
    ("Mark Gurman",   re.compile(r'\bGurman\b', re.I)),
]


def clean_for_topics(text: str) -> str:
    """Strip artifacts; return cleaned string for TF-IDF."""
    t = _STRIP_RE.sub(' ', text)
    # Remove isolated digits and 1–2 char tokens
    t = re.sub(r'\b\d+\b', ' ', t)
    t = re.sub(r'\b\w{1,2}\b', ' ', t)
    return ' '.join(t.split())


def get_top_terms(texts: list, n: int = 5) -> str:
    """Top-N TF-IDF unigrams/bigrams after artifact removal."""
    cleaned = [clean_for_topics(t) for t in texts]
    cleaned = [t for t in cleaned if len(t) > 10]
    if not cleaned:
        return ""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=150,
        min_df=1,
        token_pattern=r"[a-zA-Z][a-zA-Z0-9']{2,}"   # alpha-starting, ≥3 chars
    )
    try:
        X = vectorizer.fit_transform(cleaned)
    except ValueError:
        return ""
    scores = X.sum(axis=0).A1
    indices = scores.argsort()[::-1]
    features = vectorizer.get_feature_names_out()
    return ", ".join(features[i] for i in indices[:n])


def get_entities_regex(texts: list, min_count: int = 1) -> str:
    """
    Count curated entity occurrences across a list of post texts.
    Returns top entities by frequency (>=min_count mentions).
    Only counts genuinely present entities; does not fabricate.
    """
    counts: dict = defaultdict(int)
    for text in texts:
        for name, pattern in _ENTITIES:
            if pattern.search(text):
                counts[name] += 1
    if not counts:
        return ""
    filtered = {k: v for k, v in counts.items() if v >= min_count}
    if not filtered:
        return ""
    sorted_ents = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return ", ".join(f"{k} ({v})" for k, v in sorted_ents[:10])


def main():
    print("=" * 65)
    print("DATA VORTEX A'26 Round 3 — Topic & Entity Analysis (Fix 3)")
    print("  Entity method: curated regex dictionary")
    print("=" * 65)

    events_csv = _ROUND3_DIR / "data" / "processed" / "flagged_events.csv"
    if not events_csv.exists():
        print("  [ERROR] flagged_events.csv not found.")
        sys.exit(1)

    with open(events_csv, "r", encoding="utf-8") as f:
        events = list(csv.DictReader(f))

    if not events:
        print("  [WARN] No flagged events.")
        sys.exit(0)

    print(f"  [INFO] Analysing {len(events)} events...")

    enriched_csv = cfg.PROCESSED_POSTS_CSV
    if not enriched_csv.exists():
        print("  [ERROR] bluesky_enriched.csv not found.")
        sys.exit(1)

    with open(enriched_csv, "r", encoding="utf-8") as f:
        posts = list(csv.DictReader(f))

    # Group posts by UTC hour bucket
    posts_by_hour = defaultdict(list)
    for p in posts:
        ts_str = p.get("created_at") or ""
        if not ts_str:
            continue
        try:
            dt = dateutil.parser.isoparse(ts_str)
            hour_key = dt.replace(minute=0, second=0, microsecond=0).isoformat()
            posts_by_hour[hour_key].append(p)
        except Exception:
            pass

    results = []
    events_with_entities = 0

    for ev in events:
        curr_ts = ev["timestamp"]
        prev_ts = ev.get("previous_timestamp", "")

        curr_posts = posts_by_hour.get(curr_ts, [])
        prev_posts = posts_by_hour.get(prev_ts, []) if prev_ts else []

        # TF-IDF uses clean_text (Round 2 preprocessed); artifacts stripped on top
        curr_topics = get_top_terms([p.get("clean_text", "") for p in curr_posts])
        prev_topics = get_top_terms([p.get("clean_text", "") for p in prev_posts])

        # Regex entity extraction uses raw text for proper capitalisation matching
        curr_ents = get_entities_regex([p.get("text", "") for p in curr_posts])
        prev_ents = get_entities_regex([p.get("text", "") for p in prev_posts])

        if curr_ents or prev_ents:
            events_with_entities += 1

        results.append({
            "event_id": ev["event_id"],
            "timestamp": curr_ts,
            "event_type": ev["event_type"],
            "bucket_post_count": len(curr_posts),
            "current_topics": curr_topics,
            "previous_topics": prev_topics,
            "current_entities": curr_ents,
            "previous_entities": prev_ents,
        })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = _ROUND3_DIR / "data" / "processed" / "topic_entity_by_event.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"  [OK] Saved {len(results)} rows → {out_csv.name}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"  [OK] Events with genuine entities: {events_with_entities} / {len(results)}")
    print()
    for r in results:
        print(f"  {r['event_id']} ({r['event_type']} @ {r['timestamp']})")
        print(f"    Topics   : {r['current_topics']}")
        print(f"    Entities : {r['current_entities'] or '(none matched)'}")

    # ── Update phase4 thresholds metadata ─────────────────────────────────────
    thresh_file = _ROUND3_DIR / "scratch" / "phase4_thresholds.json"
    if thresh_file.exists():
        with open(thresh_file) as f:
            td = json.load(f)
        td["entity_extraction_method"] = (
            "Curated regex dictionary (companies, products, people, platforms). "
            "NLTK/spaCy attempted but both hang in this environment."
        )
        td["topic_url_artifact_removed"] = True
        td["events_with_entities"] = events_with_entities
        with open(thresh_file, "w") as f:
            json.dump(td, f, indent=2)

    # ── Update phase4_findings_notes.md limitation section ───────────────────
    findings_md = _ROUND3_DIR / "reports" / "phase4_findings_notes.md"
    if findings_md.exists():
        text = findings_md.read_text(encoding="utf-8")
        # Replace old NER limitation lines
        old_ner = ("3. **Missing Entity Extraction:** `spaCy` NER was unavailable "
                   "in the environment, so no named entities were extracted. "
                   "The empty entity fields for each event do not indicate that "
                   "no entities existed in the text.")
        new_ner = (
            f"3. **Entity Extraction (Fix 3):** A curated regex dictionary was used "
            f"(companies: Apple, Samsung, Google, Xiaomi; products: iPhone 17/18/Duo/Pro, "
            f"iOS 26/27, AirPods, Apple Watch; people: Tim Cook, Mark Gurman). "
            f"This produced genuine entities for **{events_with_entities} of {len(results)} events**. "
            f"spaCy and NLTK ne_chunk were attempted but both hung indefinitely in the "
            f"current Windows environment and were abandoned. "
            f"The regex approach only matches genuinely present named entities and does not fabricate."
        )
        if old_ner in text:
            text = text.replace(old_ner, new_ner)
        else:
            text += f"\n\n{new_ner}"
        findings_md.write_text(text, encoding="utf-8")
        print(f"\n  [OK] Updated {findings_md.name}")

    print("=" * 65)


if __name__ == "__main__":
    main()
