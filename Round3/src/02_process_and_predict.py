"""
Round3/src/02_process_and_predict.py
Phase 3: Sentiment Prediction

Loads the raw dataset, applies the exact Round 2 cleaning function,
and predicts sentiment using the frozen Round 2 model.
Output is saved to `bluesky_enriched.csv` with all raw columns preserved.
"""

import sys
import csv
import importlib.util
from pathlib import Path

# Resolve paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROUND3_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _ROUND3_DIR.parent
sys.path.insert(0, str(_ROUND3_DIR))
import config as cfg

# Add Round 2 source directory to path for imports within the inference pipeline
_R2_SRC = _REPO_ROOT / "Round2" / "src"
sys.path.insert(0, str(_R2_SRC))

def load_inference_pipeline():
    """Load the digit-prefixed inference_pipeline module from Round 2."""
    spec = importlib.util.spec_from_file_location(
        "inference_pipeline", _R2_SRC / "08_inference_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    print("=================================================================")
    print("DATA VORTEX A'26 Round 3 — Process & Predict")
    print("=================================================================")

    # 1. Load Round 2 Model
    print("[1/4] Loading frozen Round 2 model...")
    pipeline_mod = load_inference_pipeline()
    try:
        model_bundle = pipeline_mod.load_model(
            model_path=str(cfg.R2_MODEL_PKL),
            config_path=str(cfg.R2_CONFIG_JSON),
            encoders_path=str(cfg.R2_ENCODERS_JSON),
        )
        print("  [OK] Model bundle loaded.")
    except Exception as e:
        print(f"  [ERROR] Failed to load Round 2 model: {e}")
        sys.exit(1)

    # 2. Load Relevance-Filtered Data (Fix 1: uses bluesky_relevant.csv not raw_final)
    raw_csv = cfg.RELEVANT_POSTS_CSV
    print(f"[2/4] Loading filtered dataset: {raw_csv.name}...")
    if not raw_csv.exists():
        print("  [ERROR] Filtered dataset not found. Run 00_filter_relevant.py first.")
        sys.exit(1)

    rows = []
    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            rows.append(row)
    print(f"  [OK] Loaded {len(rows)} posts.")

    # 3. Predict Sentiment
    print("[3/4] Running inference (cleaning + prediction)...")
    texts = [r.get("text", "") for r in rows]
    
    # We use the pipeline's predict function which handles cleaning and returns dicts
    # containing 'raw_text', 'clean_text', 'label', 'confidence'
    results = pipeline_mod.predict(texts, model_bundle)
    
    out_rows = []
    failed_preds = 0
    import dateutil.parser
    from datetime import timezone

    for row, res in zip(rows, results):
        out_row = dict(row)
        out_row["clean_text"] = res.get("clean_text", "")
        # Handle cases where model might return None or fail
        label = res.get("label")
        if not label:
            failed_preds += 1
            label = "Neutral" # Fallback, shouldn't happen with valid TF-IDF
            
        out_row["sentiment_label"] = label
        
        # Check future dated anomaly
        is_future = "False"
        try:
            c_at = dateutil.parser.isoparse(row.get("created_at", "")).astimezone(timezone.utc)
            coll_at = dateutil.parser.isoparse(row.get("collected_at", "")).astimezone(timezone.utc)
            if c_at > coll_at:
                is_future = "True"
        except Exception:
            pass
        out_row["future_dated_anomaly"] = is_future
        
        out_rows.append(out_row)

    print(f"  [OK] Inference complete. Prediction failures: {failed_preds}")

    # 4. Save Enriched Data
    out_csv = cfg.PROCESSED_POSTS_CSV
    print(f"[4/4] Saving enriched dataset: {out_csv.name}...")
    
    out_fields = list(fields) + ["clean_text", "sentiment_label", "future_dated_anomaly"]
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"  [OK] Saved {len(out_rows)} enriched posts.")
    print("=================================================================")

if __name__ == "__main__":
    main()
