# Phase 1 — Audit Round 2 & Prepare Round 3 Workspace

## Objective
Confirm exactly what Round 2 already gives us (model, cleaning code, inference interface) and set up a clean `Round3/` workspace that reuses it — without touching or retraining anything in `Round2/`.

## Context / Input
- Existing repo contains `Round2/` with a trained sentiment model and supporting code.
- Round 3 rulebook requires: live-collected data, reuse of the Round 2 NLP model, ≥2 sentiment shifts, ≥1 engagement spike, topic/entity analysis, trigger explanation, PDF report.
- Nothing from Round 3 exists yet.

## Exact Tasks
1. Open and read (do not edit):
   - `Round2/README.md`
   - `Round2/src/utils.py` (focus on `clean_text()`)
   - `Round2/src/08_inference_pipeline.py` (`load_model()`, `predict()`, `predict_labels()`)
   - `Round2/artifacts/final_model_config.json`
   - `Round2/artifacts/label_encoders.json`
   - `Round2/reports/evaluation_report.md`
   - `Round2/reports/error_analysis.md`
2. Verify the model actually loads and predicts, from a fresh script:
   ```python
   import sys; sys.path.insert(0, "Round2/src")
   from inference_pipeline import load_model, predict
   bundle = load_model(
       model_path="Round2/models/final_sentiment_model.pkl",
       config_path="Round2/artifacts/final_model_config.json",
       encoders_path="Round2/artifacts/label_encoders.json",
   )
   print(predict(["I love this!", "This is terrible.", "It launches Tuesday."], bundle))
   ```
3. Record in a short audit note (`Round3/reports/phase1_audit_notes.md`):
   - Model type, classes (Negative/Neutral/Positive), val/test macro-F1 (~60%).
   - Confirmed: NO topic/entity classifier exists in Round 2 — topic/entity work in Round 3 is new.
   - Confirmed known weakness: negation is the top error driver; Neutral is the weakest class. Note this so later phases sanity-check flagged results by hand.
4. Create the Round 3 folder structure:
   ```
   Round3/
     data/raw/
     data/processed/
     src/
     notebooks/
     reports/figures/
     reports/
   ```
5. Copy (do not move/modify) `Round2/src/utils.py` and `Round2/src/08_inference_pipeline.py` into `Round3/src/round2_reuse/` so Round 3 has its own frozen copy to import from, decoupled from any future Round 2 changes.
6. Create `Round3/config.py` with: chosen product/launch name, subreddit(s) to monitor, collection window start/end (Day 8–10, 20–22 Sep 2026), and file paths.

## Files to Inspect
`Round2/README.md`, `Round2/src/utils.py`, `Round2/src/08_inference_pipeline.py`, `Round2/artifacts/*.json`, `Round2/reports/evaluation_report.md`, `Round2/reports/error_analysis.md`

## Files to Create
`Round3/reports/phase1_audit_notes.md`, `Round3/config.py`, `Round3/src/round2_reuse/utils.py`, `Round3/src/round2_reuse/inference_pipeline.py`, empty `Round3/` folder tree above.

## Expected Outputs
- Working proof that the Round 2 model loads and predicts inside `Round3/`.
- `phase1_audit_notes.md` summarizing model facts + known weaknesses.
- Full `Round3/` folder skeleton ready for Phase 2.

## Validation / Checks
- [ ] The test prediction script runs without errors and returns 3 valid labels.
- [ ] `phase1_audit_notes.md` explicitly states there is no existing topic/entity model.
- [ ] `Round3/` folder structure exists exactly as specified.
- [ ] Nothing under `Round2/` was modified (diff check).

## Completion Criteria
Round 2 model verified working from inside `Round3/`, audit notes written, workspace skeleton exists.

## Must NOT Do
- Do not retrain, fine-tune, or edit the Round 2 model or its training scripts.
- Do not modify any file inside `Round2/`.
- Do not build a topic/entity classifier yet — that's Phase 4.
- Do not start data collection yet.
