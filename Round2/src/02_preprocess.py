"""
02_preprocess.py — DATA VORTEX A'26, Round 2
Phase 2: Preprocessing Pipeline

Consumes:
  data/raw/Labeled_Social_NLP_Training_Data.csv

Produces:
  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv
  artifacts/label_encoders.json
  reports/cleaning_log.md

Run from the Round2/ project root:
    python src/02_preprocess.py
"""

import re
import sys
import json
import hashlib
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

# ── Import shared utilities (same cleaning function Phase 8 will use) ─────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    clean_text,
    encode_labels,
    save_label_encoders,
    SENTIMENT_LABEL_MAP,
    TOPIC_LABEL_MAP,
    LABEL_ENCODERS,
    PAT_UNICODE_ESCAPE,
    PAT_HTML_ENTITY,
)

# ── Reproducibility ───────────────────────────────────────────────────────
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────
RAW_CSV     = PROJECT_ROOT / "data" / "raw" / "Labeled_Social_NLP_Training_Data.csv"
SOURCE_CSV  = PROJECT_ROOT / "data" / "Labeled_Social_NLP_Training_Data.csv"
PROC_DIR    = PROJECT_ROOT / "data" / "processed"
ART_DIR     = PROJECT_ROOT / "artifacts"
REPORT_PATH = PROJECT_ROOT / "reports" / "cleaning_log.md"

# Ensure directories exist
for d in [PROC_DIR, ART_DIR, PROJECT_ROOT / "reports"]:
    d.mkdir(parents=True, exist_ok=True)

# Copy raw CSV to data/raw/ if needed (byte-for-byte; never edited in place)
if SOURCE_CSV.exists() and not RAW_CSV.exists():
    shutil.copy2(SOURCE_CSV, RAW_CSV)
    print("[INFO] Copied raw CSV -> data/raw/")
elif not RAW_CSV.exists():
    print("[ERROR] Raw CSV not found at " + str(RAW_CSV))
    sys.exit(1)

# =============================================================================
# STEP 0 — Load raw CSV with dtype=str / keep_default_na=False
# =============================================================================
print("\n[STEP 0] Loading raw CSV ...")
df = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False)
n_raw = len(df)
print("  Loaded " + str(n_raw) + " rows, " + str(len(df.columns)) + " columns.")

# Inline assertion: no nulls
assert df.isnull().sum().sum() == 0, "Unexpected nulls in raw CSV!"
print("  Null check: OK (0 nulls)")

# =============================================================================
# STEP 1 — Smoke-test clean_text on a known example from the audit report
# =============================================================================
print("\n[STEP 1] Smoke-test clean_text (Unicode escape + HTML + @mention + URL) ...")

_raw_u = r"Tune into @user for live coverage of tonight\u2019s NNS race."
_cln_u = clean_text(_raw_u)
assert "\u2019" in _cln_u, "FAIL: \\u2019 not decoded to right-single-quote. Got: " + _cln_u
assert r"\u2019" not in _cln_u, "FAIL: literal \\u2019 still present. Got: " + _cln_u
print("  Unicode decode  OK: '" + _raw_u[:50] + "' -> '" + _cln_u[:50] + "'")

_raw_h = "Federer &amp; Murray &quot;march&quot; on"
_cln_h = clean_text(_raw_h)
assert "&amp;" not in _cln_h, "FAIL: &amp; not decoded."
assert "&quot;" not in _cln_h, "FAIL: &quot; not decoded."
print("  HTML decode     OK: '" + _raw_h + "' -> '" + _cln_h + "'")

_raw_m = "@dhrit replied to @newsbot about something"
_cln_m = clean_text(_raw_m)
assert _cln_m == "@user replied to @user about something", "FAIL: mention norm. Got: " + _cln_m
print("  Mention norm    OK: '" + _raw_m + "' -> '" + _cln_m + "'")

_raw_url = "See https://example.com for details"
_cln_url = clean_text(_raw_url)
assert "<url>" in _cln_url, "FAIL: URL not replaced. Got: " + _cln_url
print("  URL replace     OK: '" + _raw_url + "' -> '" + _cln_url + "'")

print("  All smoke-tests PASSED.")

# =============================================================================
# STEP 2 — Apply text cleaning
# =============================================================================
print("\n[STEP 2] Applying clean_text to post_text -> text_clean ...")

df["text_clean"] = df["post_text"].apply(clean_text)

# Count how many rows were actually changed by each sub-step
# (for the cleaning_log — measure on raw column)
n_unicode_rows = int(df["post_text"].str.contains(PAT_UNICODE_ESCAPE, regex=True, na=False).sum())
n_html_rows    = int(df["post_text"].str.contains(PAT_HTML_ENTITY,    regex=True, na=False).sum())
n_atall_rows   = int(df["post_text"].str.contains(r'@[A-Za-z0-9_]+', regex=True, na=False).sum())
n_url_rows     = int(df["post_text"].str.contains(r'https?://\S+|www\.\S+', regex=True, na=False).sum())
n_changed      = int((df["post_text"] != df["text_clean"]).sum())
n_unchanged    = n_raw - n_changed

print("  Rows changed by cleaning      : " + str(n_changed) + " / " + str(n_raw))
print("  Rows unchanged                : " + str(n_unchanged))

# Validation: assert no literal \\uXXXX or HTML entities remain in text_clean
remaining_unicode = int(df["text_clean"].str.contains(PAT_UNICODE_ESCAPE, regex=True, na=False).sum())
remaining_html    = int(df["text_clean"].str.contains(PAT_HTML_ENTITY,    regex=True, na=False).sum())
assert remaining_unicode == 0, \
    "VALIDATION FAILED: " + str(remaining_unicode) + " rows still contain literal \\uXXXX in text_clean!"
assert remaining_html == 0, \
    "VALIDATION FAILED: " + str(remaining_html) + " rows still contain HTML entities in text_clean!"
print("  Post-clean validation: 0 remaining Unicode escapes, 0 remaining HTML entities. OK")

# Collect before/after examples for the cleaning log
def get_before_after_examples(mask, n=3):
    """Return list of (raw, clean) pairs for rows matching mask."""
    idxs = df[mask].index[:n]
    return [(df.loc[i, "post_text"], df.loc[i, "text_clean"]) for i in idxs]

unicode_mask  = df["post_text"].str.contains(PAT_UNICODE_ESCAPE, regex=True, na=False)
html_mask     = df["post_text"].str.contains(PAT_HTML_ENTITY,    regex=True, na=False)
mention_mask  = df["post_text"].str.contains(r'@(?!user\b)[A-Za-z0-9_]+', regex=True, na=False)
url_mask      = df["post_text"].str.contains(r'https?://\S+|www\.\S+',     regex=True, na=False)

unicode_examples  = get_before_after_examples(unicode_mask)
html_examples     = get_before_after_examples(html_mask)
mention_examples  = get_before_after_examples(mention_mask)
url_examples      = get_before_after_examples(url_mask)

# =============================================================================
# STEP 3 — Label encoding
# =============================================================================
print("\n[STEP 3] Encoding labels ...")

df["sentiment_label_id"] = encode_labels(df["sentiment_label"], SENTIMENT_LABEL_MAP).astype(int)
df["topic_category_id"]  = encode_labels(df["topic_category"],  TOPIC_LABEL_MAP).astype(int)

print("  sentiment_label mapping : " + str(SENTIMENT_LABEL_MAP))
print("  topic_category mapping  : " + str(TOPIC_LABEL_MAP))

# Save encoders
save_label_encoders(ART_DIR / "label_encoders.json")

# =============================================================================
# STEP 4 — Group-aware stratified split (70 / 15 / 15)
# =============================================================================
print("\n[STEP 4] Group-aware stratified split (target 70/15/15) ...")

# Assign each unique text_clean to a group id (group = all rows with same cleaned text)
# Then stratify split at GROUP level by majority sentiment_label of the group.
# Since Phase 1 confirmed all duplicate groups are 100% label-consistent,
# every group has a single label — using .first() is safe.

groups = df.groupby("text_clean").agg(
    group_label   =("sentiment_label", "first"),
    group_label_id=("sentiment_label_id", "first"),
    row_count     =("text_id", "count"),
).reset_index()

n_groups = len(groups)
print("  Total unique text groups : " + str(n_groups))

# Stratified shuffle of groups by group_label
# We shuffle within each stratum then take 70% train, 15% val, 15% test
def stratified_group_split(groups_df, val_frac=0.15, test_frac=0.15, seed=42):
    rng_local = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for lbl, stratum in groups_df.groupby("group_label"):
        idx = stratum.index.tolist()
        rng_local.shuffle(idx)
        n = len(idx)
        n_test  = max(1, round(n * test_frac))
        n_val   = max(1, round(n * val_frac))
        n_train = n - n_val - n_test

        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    return (
        groups_df.loc[train_idx, "text_clean"].tolist(),
        groups_df.loc[val_idx,   "text_clean"].tolist(),
        groups_df.loc[test_idx,  "text_clean"].tolist(),
    )

train_texts, val_texts, test_texts = stratified_group_split(
    groups, val_frac=0.15, test_frac=0.15, seed=RANDOM_SEED
)

train_set = set(train_texts)
val_set   = set(val_texts)
test_set  = set(test_texts)

# Assign split to each row
def assign_split(tc):
    if tc in train_set:
        return "train"
    elif tc in val_set:
        return "val"
    else:
        return "test"

df["split"] = df["text_clean"].apply(assign_split)

train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()
test_df  = df[df["split"] == "test"].copy()

# ── Validation: no text_clean appears in more than one split ──────────────
overlap_tv = train_set & val_set
overlap_tt = train_set & test_set
overlap_vt = val_set   & test_set
assert len(overlap_tv) == 0, "LEAK: " + str(len(overlap_tv)) + " texts in both train and val!"
assert len(overlap_tt) == 0, "LEAK: " + str(len(overlap_tt)) + " texts in both train and test!"
assert len(overlap_vt) == 0, "LEAK: " + str(len(overlap_vt)) + " texts in both val and test!"
print("  Leakage check: PASSED (no text_clean shared across splits)")

# ── Validation: row counts sum to original ────────────────────────────────
assert len(train_df) + len(val_df) + len(test_df) == n_raw, \
    "FAIL: row counts don't sum to " + str(n_raw)
print("  Row count integrity: PASSED (" + str(len(train_df)) + " + "
      + str(len(val_df)) + " + " + str(len(test_df)) + " = " + str(n_raw) + ")")

# ── Per-split sizes and class balance ────────────────────────────────────
print("\n  Split sizes and per-class balance (sentiment_label):")
for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
    n_s = len(sdf)
    pct = round(100.0 * n_s / n_raw, 1)
    counts = sdf["sentiment_label"].value_counts()
    balance_str = "  ".join(
        str(lbl) + "=" + str(int(counts.get(lbl, 0)))
        for lbl in ["Negative", "Neutral", "Positive"]
    )
    print("    " + name.ljust(6) + ": " + str(n_s).rjust(5) + " rows (" + str(pct) + "%)  |  " + balance_str)

# =============================================================================
# STEP 5 — Save processed CSVs
# =============================================================================
print("\n[STEP 5] Saving processed CSVs ...")

OUTPUT_COLS = [
    "text_id", "post_text", "text_clean",
    "sentiment_label", "sentiment_label_id",
    "topic_category",  "topic_category_id",
]

for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
    out_path = PROC_DIR / (name + ".csv")
    sdf[OUTPUT_COLS].to_csv(out_path, index=False, encoding="utf-8")
    print("  Saved " + str(out_path) + "  (" + str(len(sdf)) + " rows)")

# =============================================================================
# STEP 6 — Write cleaning_log.md
# =============================================================================
print("\n[STEP 6] Writing cleaning_log.md ...")

def ex_table(examples, label_before="Raw `post_text`", label_after="Cleaned `text_clean`"):
    """Build a markdown table of before/after examples."""
    lines = ["| # | " + label_before + " | " + label_after + " |",
             "|---|---|---|"]
    for i, (raw, clean) in enumerate(examples, 1):
        # Escape pipes; truncate to keep table readable
        r = raw.replace("|", "/")[:120]
        c = clean.replace("|", "/")[:120]
        lines.append("| " + str(i) + " | " + r + " | " + c + " |")
    return "\n".join(lines)

# Re-compute split-level stats for the log
def split_stats(sdf, name):
    n_s = len(sdf)
    pct = round(100.0 * n_s / n_raw, 1)
    counts = sdf["sentiment_label"].value_counts()
    rows = [
        "| " + name + " | " + str(n_s) + " | " + str(pct) + "% |"
        + " | ".join(str(int(counts.get(lbl, 0))) for lbl in ["Negative", "Neutral", "Positive"]) + " |"
    ]
    return "\n".join(rows)

log_lines = []
log_lines.append("# Cleaning Log \u2014 DATA VORTEX A\u201926 Round 2")
log_lines.append("**Phase 2: Preprocessing Pipeline**")
log_lines.append("Generated by `src/02_preprocess.py`.")
log_lines.append("")
log_lines.append("---")
log_lines.append("")
log_lines.append("## 1. Source & Row Counts")
log_lines.append("")
log_lines.append("| Item | Value |")
log_lines.append("|---|---|")
log_lines.append("| Raw CSV | `data/raw/Labeled_Social_NLP_Training_Data.csv` |")
log_lines.append("| Raw row count | " + str(n_raw) + " |")
log_lines.append("| Rows dropped | 0 (no rows dropped \u2014 zero nulls, zero bad labels) |")
log_lines.append("| Rows in processed splits | " + str(n_raw) + " |")
log_lines.append("")
log_lines.append("---")
log_lines.append("")
log_lines.append("## 2. Text Cleaning Steps")
log_lines.append("")
log_lines.append("All cleaning is applied in `src/utils.py::clean_text()`. "
                  "The original `post_text` column is preserved unchanged alongside `text_clean`.")
log_lines.append("")
log_lines.append("| Step | Description | Rows Affected |")
log_lines.append("|---|---|---|")
log_lines.append("| 1. Unicode escape decode | Convert literal `\\uXXXX` sequences to real characters | "
                  + str(n_unicode_rows) + " (" + str(round(100.0*n_unicode_rows/n_raw,1)) + "%) |")
log_lines.append("| 2. HTML entity decode | `html.unescape()` for `&amp;`, `&quot;`, `&#39;`, etc. | "
                  + str(n_html_rows) + " (" + str(round(100.0*n_html_rows/n_raw,1)) + "%) |")
log_lines.append("| 3. @mention normalisation | All `@xxx` \u2192 `@user` (placeholder kept, not deleted) | "
                  + str(n_atall_rows) + " (" + str(round(100.0*n_atall_rows/n_raw,1)) + "%) |")
log_lines.append("| 4. URL replacement | URLs \u2192 `<url>` (placeholder kept, not deleted) | "
                  + str(n_url_rows) + " (" + str(round(100.0*n_url_rows/n_raw,1)) + "%) |")
log_lines.append("| 5. Whitespace collapse | Strip leading/trailing; collapse repeated spaces | all rows |")
log_lines.append("| **Total rows changed** | At least one step applied | **"
                  + str(n_changed) + " (" + str(round(100.0*n_changed/n_raw,1)) + "%)** |")
log_lines.append("")
log_lines.append("### Design Decisions")
log_lines.append("")
log_lines.append("- **Lowercasing withheld** \u2014 ALL-CAPS emphasis may carry sentiment signal; "
                  "the vectorizer config in Phase 4 will decide (ablatable without re-running this phase).")
log_lines.append("- **Hashtags NOT stripped** \u2014 hashtag terms carry content; "
                  "the `#` symbol itself can be kept or stripped by the vectorizer tokeniser. "
                  "Removing hashtags here would destroy that choice for Phase 4/5.")
log_lines.append("- **@user normalised, not deleted** \u2014 the presence/count of mentions "
                  "may weakly correlate with conversational tone.")
log_lines.append("- **URLs replaced with `<url>`, not deleted** \u2014 presence of a URL "
                  "may carry weak signal (e.g. news-sharing vs. personal opinion posts).")
log_lines.append("- **Zero rows dropped** \u2014 the dataset has no nulls, no bad labels, "
                  "and no rows that failed cleaning; dropping any row would require explicit justification.")
log_lines.append("")
log_lines.append("---")
log_lines.append("")
log_lines.append("## 3. Before/After Examples")
log_lines.append("")
log_lines.append("### 3.1 Unicode Escape Decoding")
log_lines.append("")
log_lines.append(ex_table(unicode_examples))
log_lines.append("")
log_lines.append("### 3.2 HTML Entity Decoding")
log_lines.append("")
log_lines.append(ex_table(html_examples))
log_lines.append("")
log_lines.append("### 3.3 @mention Normalisation")
log_lines.append("")
log_lines.append(ex_table(mention_examples))
log_lines.append("")
log_lines.append("### 3.4 URL Replacement")
log_lines.append("")
log_lines.append(ex_table(url_examples) if url_examples else "_No URL rows available in first 3 non-`@user` mention rows._")
log_lines.append("")
log_lines.append("---")
log_lines.append("")
log_lines.append("## 4. Post-Cleaning Validation")
log_lines.append("")
log_lines.append("| Check | Result |")
log_lines.append("|---|---|")
log_lines.append("| Remaining literal `\\uXXXX` in `text_clean` | **"
                  + str(remaining_unicode) + "** (must be 0) \u2714 |")
log_lines.append("| Remaining HTML entities in `text_clean` | **"
                  + str(remaining_html) + "** (must be 0) \u2714 |")
log_lines.append("| `text_clean` shared across train/val | **"
                  + str(len(overlap_tv)) + "** (must be 0) \u2714 |")
log_lines.append("| `text_clean` shared across train/test | **"
                  + str(len(overlap_tt)) + "** (must be 0) \u2714 |")
log_lines.append("| `text_clean` shared across val/test | **"
                  + str(len(overlap_vt)) + "** (must be 0) \u2714 |")
log_lines.append("| Row count sum = raw row count | **"
                  + str(len(train_df)+len(val_df)+len(test_df)) + " = " + str(n_raw) + "** \u2714 |")
log_lines.append("")
log_lines.append("---")
log_lines.append("")
log_lines.append("## 5. Label Encoding")
log_lines.append("")
log_lines.append("Mappings saved to `artifacts/label_encoders.json`. "
                  "**Always load from that file at inference time \u2014 do not recompute.**")
log_lines.append("")
log_lines.append("### `sentiment_label` encoding")
log_lines.append("")
log_lines.append("| Label | ID |")
log_lines.append("|---|---|")
for lbl, idx in sorted(SENTIMENT_LABEL_MAP.items(), key=lambda x: x[1]):
    log_lines.append("| `" + lbl + "` | " + str(idx) + " |")
log_lines.append("")
log_lines.append("### `topic_category` encoding")
log_lines.append("")
log_lines.append("| Label | ID |")
log_lines.append("|---|---|")
for lbl, idx in sorted(TOPIC_LABEL_MAP.items(), key=lambda x: x[1]):
    log_lines.append("| `" + lbl + "` | " + str(idx) + " |")
log_lines.append("")
log_lines.append("---")
log_lines.append("")
log_lines.append("## 6. Split Sizes & Class Balance")
log_lines.append("")
log_lines.append("Split strategy: **group-aware stratified split** at the `text_clean` group level.")
log_lines.append("All rows sharing a cleaned text are assigned to the same split partition.")
log_lines.append("Stratification by `sentiment_label` within each group-stratum.")
log_lines.append("Target ratio: 70 / 15 / 15. Random seed: " + str(RANDOM_SEED) + ".")
log_lines.append("")

# Build full balance table
log_lines.append("| Split | N rows | % of total | Negative | Neutral | Positive |")
log_lines.append("|---|---|---|---|---|---|")
for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
    n_s   = len(sdf)
    pct_s = round(100.0 * n_s / n_raw, 1)
    vc    = sdf["sentiment_label"].value_counts()
    log_lines.append(
        "| " + name + " | " + str(n_s) + " | " + str(pct_s) + "% | "
        + str(int(vc.get("Negative", 0))) + " | "
        + str(int(vc.get("Neutral",  0))) + " | "
        + str(int(vc.get("Positive", 0))) + " |"
    )

log_lines.append("")

# Flag imbalance if any class deviates by >5pp from expected (1/3 of split)
flag_lines = []
for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
    n_s = len(sdf)
    vc  = sdf["sentiment_label"].value_counts()
    for lbl in ["Negative", "Neutral", "Positive"]:
        actual_pct = 100.0 * vc.get(lbl, 0) / n_s if n_s > 0 else 0
        if abs(actual_pct - 33.33) > 5.0:
            flag_lines.append("  - " + name + " / " + lbl + ": "
                               + str(round(actual_pct, 1)) + "% (expected ~33.3%)")

if flag_lines:
    log_lines.append("> **WARNING: Class imbalance exceeds 5pp from expected 33.3% in:**")
    log_lines.extend(flag_lines)
else:
    log_lines.append("> All splits maintain per-class balance within 5pp of the expected 33.3%. "
                     "Group-aware splitting preserved class balance successfully.")

log_lines.append("")
log_lines.append("---")
log_lines.append("")
log_lines.append("## 7. Output Files")
log_lines.append("")
log_lines.append("| File | Rows | Columns |")
log_lines.append("|---|---|---|")
for name in ["train", "val", "test"]:
    log_lines.append("| `data/processed/" + name + ".csv` | "
                      + str(len(df[df["split"]==name])) + " | "
                      + str(len(OUTPUT_COLS)) + " |")
log_lines.append("| `artifacts/label_encoders.json` | \u2014 | JSON mapping |")
log_lines.append("")
log_lines.append("---")
log_lines.append("")
log_lines.append("*Report generated by `src/02_preprocess.py` "
                  "\u2014 re-run to regenerate all files deterministically.*")

REPORT_PATH.write_text("\n".join(log_lines), encoding="utf-8")
print("[DONE] cleaning_log.md written to " + str(REPORT_PATH))

# =============================================================================
print("\n=== Phase 2 complete \u2014 Definition of Done ===")
print("  [OK] data/processed/train.csv : " + str(len(train_df)) + " rows")
print("  [OK] data/processed/val.csv   : " + str(len(val_df)) + " rows")
print("  [OK] data/processed/test.csv  : " + str(len(test_df)) + " rows")
print("  [OK] artifacts/label_encoders.json")
print("  [OK] reports/cleaning_log.md")
print()
print("  Validation summary:")
print("    0 Unicode escapes remaining in text_clean")
print("    0 HTML entities remaining in text_clean")
print("    0 text_clean values shared across any two splits")
print("    Row sum " + str(len(train_df)+len(val_df)+len(test_df)) + " = raw " + str(n_raw))
