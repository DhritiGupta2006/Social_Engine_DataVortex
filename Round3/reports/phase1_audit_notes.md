# Phase 1 Audit Notes — DATA VORTEX A'26 Round 3
Generated: 2026-09-22

---

## 1. Round 2 Model Facts

| Property | Value |
|---|---|
| **Model type** | `char(3,5)+LinearSVC C=1.0` — TF-IDF character (3–5)-gram vectorizer + LinearSVC (sklearn Pipeline) |
| **Serialised artifact** | `Round2/models/final_sentiment_model.pkl` |
| **Config artifact** | `Round2/artifacts/final_model_config.json` |
| **Label encoders** | `Round2/artifacts/label_encoders.json` |
| **Sentiment classes** | `Negative` (0), `Neutral` (1), `Positive` (2) |
| **Trained on** | train + val combined (7,654 rows) |
| **Val macro-F1** | 60.11% (exact: `val_macro_F1: 0.6011` from `final_model_config.json`) |
| **Test macro-F1** | **60.93%** (one-shot, 2026-09-19T01:05:10Z) |
| **Test accuracy** | 60.92% |
| **Prediction speed** | 0.413 s on 1,346 rows (CPU) |
| **Random seed** | 42 (fully reproducible) |

### Per-class test F1
| Class | F1 |
|---|---|
| Negative | 65.48% |
| Neutral | **56.61%** (weakest) |
| Positive | 60.70% |

---

## 2. Confirmed: No Topic/Entity Classifier in Round 2

**There is NO existing topic/entity classifier in Round 2.**

The `label_encoders.json` contains a `topic_category` key with four labels
(`Account_Security`, `Community_Discussion`, `Feature_Feedback`,
`Technical_Issues`) that were used to *stratify* the dataset split in Phase 2,
but **no classifier was trained to predict topic**.  The Round 2 model predicts
sentiment only.

Therefore, **topic and entity analysis in Round 3 is entirely new work**.  No
Round 2 artefact can be reused for this task.

---

## 3. Known Weaknesses — Sanity-Check Flags for Later Phases

### 3.1 Negation is the #1 error driver

- Posts containing a negation word have a **45.7%** error rate vs 37.5%
  overall (+8.2 pp).
- 121 of 526 test errors (23.0%) involve negation words.
- Root cause: char(3–5)-gram bag-of-n-grams has no mechanism to represent
  negation scope.  "not excited" shares `excit`, `cite`, `ited` n-grams with
  genuinely positive posts.
- **Action for Round 3**: When interpreting sentiment predictions on collected
  posts, hand-check any post containing negation words (`not`, `no`, `never`,
  `won't`, `don't`, etc.) before drawing conclusions.

### 3.2 Neutral is the weakest class (F1 = 56.61%)

- Neutral posts are most often confused with Negative (21.9%) and Positive
  (21.4%).
- Top failure sub-mode: **Positive→Neutral** is the single largest confusion
  pair (118 errors = 22.4% of all errors), typically mild/hedged positive posts
  lacking strong exclamatory char-grams.
- **Action for Round 3**: Treat model-labelled Neutral boundaries with extra
  caution.  When a post is labelled Neutral but contains weak positive language,
  consider it a borderline case and note it in the final report.

### 3.3 Sarcasm/irony and factual-context failures
- Sarcastic posts use positive surface vocabulary with reversed intent;
  the model picks up positive char-grams and misclassifies (~15% of reviewed
  errors).
- Factual-context posts (news-style, historical references) can trigger the
  wrong class if they mention sentiment-bearing vocabulary in an objective frame.
- **Action for Round 3**: During Phase 5 trigger investigation, if a detected
  "sentiment shift" appears driven by news/factual content or sarcasm, note this
  explicitly and sanity-check labels by hand.

---

## 4. Inference Interface Summary

Public functions exposed by `Round2/src/08_inference_pipeline.py`:

| Function | Signature | Returns |
|---|---|---|
| `load_model()` | `(model_path, config_path, encoders_path) → dict` | bundle: pipeline, config, encoders, classes |
| `predict()` | `(texts: list, model_bundle: dict) → list[dict]` | per-text: raw_text, clean_text, label, confidence (None) |
| `predict_labels()` | `(texts: list, model_bundle: dict) → list[str]` | label strings only, vectorised batch |

Cleaning applied at inference time (via `utils.py::clean_text()`):
1. Decode literal `\uXXXX` Unicode escape sequences
2. Decode HTML entities (`&amp;`, `&quot;`, `&#39;`, `&lt;`, `&gt;`)
3. Normalise all `@mentions` → `@user`
4. Replace URLs → `<url>`
5. Collapse repeated whitespace

---

## 5. Proof of Working Inference (Task 2 Verification)

Script: `Round3/scratch/verify_model.py`
Command run from repo root: `python Round3/scratch/verify_model.py`
Exit code: **0** (no errors)

Output:
```
Positive | I love this!
Negative | This is terrible.
Neutral | It launches Tuesday.
```

All 3 inputs received a valid label from {Negative, Neutral, Positive}. ✔

---

## 6. Files Audited (Read-Only)

| File | Status |
|---|---|
| `Round2/README.md` | Read ✔ |
| `Round2/src/utils.py` | Read ✔ |
| `Round2/src/08_inference_pipeline.py` | Read ✔ |
| `Round2/artifacts/final_model_config.json` | Read ✔ |
| `Round2/artifacts/label_encoders.json` | Read ✔ |
| `Round2/reports/evaluation_report.md` | Read ✔ |
| `Round2/reports/error_analysis.md` | Read ✔ |

**None of the above files were modified.**
