"""
08_inference_pipeline.py — DATA VORTEX A'26, Round 2
Phase 8: End-to-End Inference Pipeline

Exposes a clean, reusable interface for sentiment prediction using the
final trained model.  Preprocessing uses the SAME clean_text() function
from src/utils.py that was used during training — no duplication.

Usage (module):
    from src.inference_pipeline import load_model, predict
    model = load_model()
    results = predict(["great day!", "terrible loss"], model)

Usage (script demo):
    python src/08_inference_pipeline.py          # from project root
    python src/08_inference_pipeline.py --demo   # explicit demo flag
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

# ── Path resolution: works whether called from project root or src/ ───────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add src/ to path so utils can be imported from any working directory
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import clean_text   # THE single source-of-truth cleaning function

# ── Artifact paths ─────────────────────────────────────────────────────────────
_MODEL_PKL      = _PROJECT_ROOT / "models"    / "final_sentiment_model.pkl"
_CONFIG_JSON    = _PROJECT_ROOT / "artifacts" / "final_model_config.json"
_ENCODERS_JSON  = _PROJECT_ROOT / "artifacts" / "label_encoders.json"


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def load_model(model_path: Path = _MODEL_PKL,
               config_path: Path = _CONFIG_JSON,
               encoders_path: Path = _ENCODERS_JSON) -> dict:
    """
    Load the serialised final model and associated metadata from disk.

    Returns
    -------
    dict with keys:
        "pipeline"   : fitted sklearn Pipeline (vectorizer + classifier)
        "config"     : dict from final_model_config.json
        "encoders"   : dict from label_encoders.json
        "classes"    : list of class name strings in classifier order
    """
    with open(model_path, "rb") as fh:
        pipeline = pickle.load(fh)

    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    with open(encoders_path, "r", encoding="utf-8") as fh:
        encoders = json.load(fh)

    classes = config.get("sentiment_classes", ["Negative", "Neutral", "Positive"])

    return {
        "pipeline": pipeline,
        "config":   config,
        "encoders": encoders,
        "classes":  classes,
    }


def predict(texts: list, model_bundle: dict) -> list:
    """
    Apply the full inference pipeline to a list of raw text strings.

    Steps (in order, mirroring the training pipeline exactly):
        1. clean_text()  — same function used during preprocessing (Phase 2)
        2. pipeline.predict()  — vectorize + classify

    LinearSVC does not expose predict_proba; confidence is returned as None.

    Parameters
    ----------
    texts       : list[str]  — raw input strings (may contain \\uXXXX escapes,
                  HTML entities, @mentions, URLs)
    model_bundle: dict returned by load_model()

    Returns
    -------
    list[dict]  — one dict per input string:
        {
          "raw_text":   original input,
          "clean_text": text after cleaning,
          "label":      predicted sentiment label (str),
          "confidence": None  (LinearSVC — no probability available),
        }
    """
    if not texts:
        return []

    pipeline = model_bundle["pipeline"]

    results = []
    for raw in texts:
        cleaned = clean_text(str(raw))
        label   = pipeline.predict([cleaned])[0]
        results.append({
            "raw_text":   raw,
            "clean_text": cleaned,
            "label":      label,
            "confidence": None,   # LinearSVC — no predict_proba without calibration
        })
    return results


# =============================================================================
# BATCH PREDICT (convenience wrapper for lists, returns labels only)
# =============================================================================

def predict_labels(texts: list, model_bundle: dict) -> list:
    """Return only the label strings, no metadata. Vectorised batch call."""
    pipeline = model_bundle["pipeline"]
    cleaned  = [clean_text(str(t)) for t in texts]
    return list(pipeline.predict(cleaned))


# =============================================================================
# DEMO / VERIFICATION  (__main__ block)
# =============================================================================

_DEMO_TEXTS = [
    # Clearly Positive
    "This is absolutely amazing! I love every single moment of it :)",
    # Clearly Negative
    "Worst day ever. I hate this so much. :(",
    # Clearly Neutral
    "The event is scheduled for Saturday at 3:30 PM at the downtown venue.",
    # Negation edge case
    "I'm not at all excited about this, not even a little.",
    # Mixed sentiment
    "I love the concept but the execution was pretty disappointing tbh.",
    # Raw text-quality issues: \\u2019 escape, &amp; entity, @mention, URL
    r"Tune into @DhritTV for live coverage of tonight\u2019s finale! "
    r"It\u2019s &amp; always a blast. Check https://example.com for more.",
    # Social-media style with hashtag
    "Just saw the game. #GoTeam what a win! Best night ever!!!",
    # Sarcastic (edge case — model may misjudge)
    "Oh yeah sure, that policy is definitely going to help everyone. Obviously.",
]


def main():
    print("=" * 65)
    print("DATA VORTEX A'26 Round 2 — Inference Pipeline Demo")
    print("=" * 65)
    print()

    print("[1/3] Loading model artifacts ...")
    bundle = load_model()
    print(f"      Model: {bundle['config']['config_description']}")
    print(f"      Trained on: {bundle['config']['trained_on']}")
    print(f"      Val macro-F1: {round(bundle['config']['val_macro_F1']*100, 2)}%")
    print()

    print("[2/3] Running predictions on demo texts ...")
    print()
    results = predict(_DEMO_TEXTS, bundle)
    for i, r in enumerate(results, 1):
        clean_disp = r["clean_text"][:80] + ("…" if len(r["clean_text"]) > 80 else "")
        raw_disp   = r["raw_text"][:70] + ("…" if len(r["raw_text"]) > 70 else "")
        print(f"  [{i}] Label: {r['label']:<10}")
        print(f"       Raw  : {raw_disp}")
        print(f"       Clean: {clean_disp}")
        print()

    print("[3/3] Verification checks ...")
    # Check cleaning ran correctly on the raw-issues text (item 6, index 5)
    r6 = results[5]
    assert r"\u2019" not in r6["clean_text"],  "FAIL: \\u2019 not decoded at inference time"
    assert "&amp;"   not in r6["clean_text"],  "FAIL: &amp; not decoded at inference time"
    assert "@DhritTV" not in r6["clean_text"], "FAIL: @mention not normalised at inference time"
    assert "@user"    in r6["clean_text"],      "FAIL: @mention should become @user"
    assert "<url>"    in r6["clean_text"],      "FAIL: URL not replaced with <url>"
    print("  [OK] Unicode escape decoded at inference time")
    print("  [OK] HTML entity decoded at inference time")
    print("  [OK] @mention normalised to @user at inference time")
    print("  [OK] URL replaced with <url> at inference time")
    print()

    # Check all predictions are valid class names
    valid_classes = set(bundle["classes"])
    for r in results:
        assert r["label"] in valid_classes, f"Unknown label: {r['label']}"
    print("  [OK] All predicted labels are valid class names")
    print()

    print("=" * 65)
    print("  INFERENCE PIPELINE: ALL CHECKS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
