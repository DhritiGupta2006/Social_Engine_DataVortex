"""
Round3/config.py — DATA VORTEX A'26 Round 3: "Social Engine"
Central configuration for the Round 3 data-collection and analysis pipeline.

Edit this file to change product/data-source targets, collection windows, or
file paths.  All Round 3 scripts import from here — do not hardcode paths
elsewhere.

Created: Phase 1 (Audit & Workspace Setup)
Revised: Phase 1 corrections — Bluesky public API, actual timestamps recorded by Phase 2
"""

from pathlib import Path

# ── Repo root (two levels up from this file: Round3/ → datavortex/) ───────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
ROUND3_DIR = Path(__file__).resolve().parent

# =============================================================================
# PRODUCT / TOPIC TARGET
# =============================================================================

chosen_product = "iPhone 17"          # Product / topic being tracked

# =============================================================================
# DATA SOURCE
# =============================================================================

data_source = "bluesky"               # Bluesky public API (no credentials required)

# Search / query terms used to pull relevant posts.
# Used by the collection script as search keywords / subreddit names.
search_query_terms = [
    "iPhone 17",
    "Apple iPhone",
    "apple",
    "iphone",
    "technology",
]

# =============================================================================
# COLLECTION WINDOW (PLANNED)
# =============================================================================
# These are the PLANNED bounds from the Round 3 rulebook.
# Actual collection timestamps MUST be recorded in collection_log.md by Phase 2.
# Do NOT assume data was collected across this entire window — use the Phase 2
# log (ACTUAL_FIRST_RUN_UTC / ACTUAL_LAST_RUN_UTC) as the ground truth for
# all timeline analysis.

planned_collection_start = "2026-09-20T00:00:00Z"   # Window B start (launch period)
planned_collection_end   = "2026-09-22T23:59:59Z"   # Window B end

# How often the collection script is intended to run (informational only).
# The actual run cadence is what collection_log.md records.
collection_interval = "single-shot"   # "single-shot" | "hourly" | "every-N-hours"

# =============================================================================
# FILE PATHS
# =============================================================================

# --- Raw data ----------------------------------------------------------------
RAW_DIR          = ROUND3_DIR / "data" / "raw"
RAW_POSTS_CSV       = RAW_DIR / "bluesky_raw.csv"           # append target per run
RAW_FINAL_CSV       = RAW_DIR / "bluesky_raw_final.csv"     # deduped, sorted, final
RELEVANT_POSTS_CSV  = RAW_DIR / "bluesky_relevant.csv"      # topic-filtered (Fix 1)
RAW_META_JSON       = RAW_DIR / "collection_metadata.json"

# --- Processed data ----------------------------------------------------------
PROCESSED_DIR       = ROUND3_DIR / "data" / "processed"
PROCESSED_POSTS_CSV = PROCESSED_DIR / "bluesky_enriched.csv" # after inference
TIMELINE_CSV        = PROCESSED_DIR / "hourly_timeline.csv"   # Phase 3 output
FLAGGED_EVENTS_CSV  = PROCESSED_DIR / "flagged_events.csv"    # Phase 4 output
TOPIC_ENTITY_CSV    = PROCESSED_DIR / "topic_entity_by_event.csv"  # Phase 4 output
TRIGGER_CANDS_CSV   = PROCESSED_DIR / "trigger_candidates.csv"     # Phase 5 output

# --- Round 2 reuse -----------------------------------------------------------
R2_REUSE_DIR     = ROUND3_DIR / "src" / "round2_reuse"
R2_MODEL_PKL     = REPO_ROOT / "Round2" / "models"    / "final_sentiment_model.pkl"
R2_CONFIG_JSON   = REPO_ROOT / "Round2" / "artifacts" / "final_model_config.json"
R2_ENCODERS_JSON = REPO_ROOT / "Round2" / "artifacts" / "label_encoders.json"

# --- Reports -----------------------------------------------------------------
REPORTS_DIR        = ROUND3_DIR / "reports"
FIGURES_DIR        = REPORTS_DIR / "figures"
COLLECTION_LOG_MD  = REPORTS_DIR / "collection_log.md"
PHASE1_AUDIT_NOTES = REPORTS_DIR / "phase1_audit_notes.md"
PHASE3_SANITY_MD   = REPORTS_DIR / "phase3_sanity_check.md"
PHASE4_FINDINGS_MD = REPORTS_DIR / "phase4_findings_notes.md"
TRIGGER_TABLE_MD   = REPORTS_DIR / "trigger_table.md"
FINAL_REPORT_MD    = REPORTS_DIR / "Round3_Analytical_Report.md"
FINAL_VALIDATION_MD = REPORTS_DIR / "final_validation.md"

# =============================================================================
# ROUND 2 MODEL — EXACT METRICS (from Round2/artifacts/final_model_config.json)
# =============================================================================
# These are the precise values from the artifact — do NOT use approximations.

R2_VAL_MACRO_F1  = 0.6011   # val_macro_F1  (60.11%)
R2_VAL_ACCURACY  = 0.6015   # val_accuracy  (60.15%)
# Test metrics from Round2/reports/evaluation_report.md (one-shot, 2026-09-19)
R2_TEST_MACRO_F1 = 0.6093   # 60.93%
R2_TEST_ACCURACY = 0.6092   # 60.92%

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

RANDOM_SEED            = 42
SHIFT_THRESHOLD_PP     = 15     # ≥15 pp swing in (pct_pos − pct_neg) = sentiment shift
SHIFT_HARD_FLOOR_PP    = 10     # must NOT lower threshold below this
SPIKE_STD_MULTIPLIER   = 2.0    # post_count ≥ mean + 2σ = engagement spike
SPIKE_HARD_FLOOR_STD   = 1.0    # must NOT lower threshold below mean + 1σ
MIN_POSTS_PER_BUCKET   = 6      # non-zero hourly buckets required before stat detection
TIME_BIN_HOURS         = 1      # primary bucket size (falls back to 3 if sparse)
MIN_ENTITY_MENTIONS    = 3      # entity must appear ≥ 3× to be reported
