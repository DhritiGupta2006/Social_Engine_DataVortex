# DATA VORTEX A'26 — Round 2 Technical Report
## "Rebuilding the Social Engine" — NLP Layer

**Team / Author:** Dhrit Gupta  
**Event:** AARUUSH '26, DATA VORTEX A'26, Round 2  
**Submission date:** 2026-09-19  
**Random seed:** 42 | **Python:** 3.10.11 | **scikit-learn:** 1.4.2

---

## 1. Problem Definition

Round 2 of DATA VORTEX A'26 frames the task as rebuilding the *semantic understanding layer* of a Social Engine whose structured-data layer was restored in Round 1. The dataset — `Labeled_Social_NLP_Training_Data.csv` — contains 9,000 short, tweet-like social media posts, each labelled with a `sentiment_label` (Negative / Neutral / Positive) and a `topic_category` (four classes).

**Primary task selected: 3-class sentiment classification** (`sentiment_label`). This task was chosen because:

- The 9,000-row dataset is **perfectly balanced** (3,000 rows per class by construction), eliminating class-imbalance complications.
- Labels are content-correlated (spot-checked and confirmed in Phase 1).
- Sentiment classification maps directly to the rulebook's first suggested task and to the "Model Performance & Evaluation" judging criterion.
- A completed, well-evaluated sentiment pipeline is more valuable than an incomplete attempt at a harder task.

**Secondary task (topic classification): Not attempted.** The `topic_category` label is severely imbalanced (Community_Discussion 86.1% majority) and shows weak content correlation on manual inspection (posts about Boko Haram labelled `Account_Security`; golf posts labelled `Technical_Issues`). A baseline classifier that always predicts the majority class already achieves 86.1% accuracy — topic classification provides no honest additional signal within the time budget.

**Named Entity Recognition: Out of scope.** No entity span annotations exist in the dataset; doing NER would require either an off-the-shelf tagger with no ground truth to evaluate against (unverifiable claims) or fabricated labels, both prohibited.

**Dataset facts (verified by Phase 1 audit script `src/01_audit.py`):**

| Fact | Value |
|---|---|
| Total rows | 9,000 |
| Columns | `text_id`, `post_text`, `sentiment_label`, `topic_category` |
| Null values | 0 |
| Exact duplicate `post_text` rows | 1,100 (12.2%), forming 987 duplicate groups |
| All duplicate groups label-consistent | Yes |
| Rows with literal `\uXXXX` Unicode escapes | 1,092 (12.1%) |
| Rows with HTML entities (`&amp;` etc.) | 360 (4.0%) |
| Rows with `@mentions` | 2,659 (29.5%) |
| Sentiment class balance | 3,000 Negative / 3,000 Neutral / 3,000 Positive |

---

## 2. Preprocessing Pipeline

Full log: `reports/cleaning_log.md`. Script: `src/02_preprocess.py`. Cleaning function (single source of truth, reused at inference): `src/utils.py::clean_text()`.

### 2.1 Cleaning Steps

| Step | Description | Rows Affected |
|---|---|---|
| 1. Unicode escape decode | Convert literal `\uXXXX` → real character (e.g. `\u2019` → `'`) | 1,092 (12.1%) |
| 2. HTML entity decode | `html.unescape()` for `&amp;`, `&quot;`, `&#39;`, etc. | 360 (4.0%) |
| 3. @mention normalisation | All `@xxx` → `@user` (placeholder kept, not deleted) | 2,659 (29.5%) |
| 4. URL replacement | URLs → `<url>` (placeholder kept, not deleted) | 16 (0.2%) |
| 5. Whitespace collapse | Strip leading/trailing; collapse repeated spaces | All rows |
| **Total rows changed** | At least one step applied | **2,162 (24.0%)** |

**Design decisions:**
- Lowercasing **withheld** — ALL-CAPS emphasis may carry sentiment signal; left as a vectorizer config decision.
- Hashtags **not stripped** — hashtag terms carry content.
- Zero rows dropped — the dataset has no nulls, no bad labels, no unrecoverable rows.

### 2.2 Before/After Example

| Raw `post_text` | Cleaned `text_clean` |
|---|---|
| `Lakers vs Heat on Jan. 17th! It\u2019s D Wade\u2019s b day... I feel bad he\u2019ll lose` | `Lakers vs Heat on Jan. 17th! It's D Wade's b day... I feel bad he'll lose` |
| `@user Hi! KAI stole my heart &amp; it's not changing` | `@user Hi! KAI stole my heart & it's not changing` |

### 2.3 Split Strategy

**Group-aware stratified split** — all rows sharing a `text_clean` value are assigned to the same split partition, preventing the 1,100 duplicate posts from leaking across train/val/test. Target ratio: 70/15/15. Seed: 42.

| Split | Rows | % Total | Negative | Neutral | Positive |
|---|---|---|---|---|---|
| train | 6,279 | 69.8% | 2,095 | 2,099 | 2,085 |
| val | 1,375 | 15.3% | 466 | 448 | 461 |
| test | 1,346 | 15.0% | 439 | 453 | 454 |

Post-split validation: zero `text_clean` overlap across any pair of splits (verified by set intersection).

---

## 3. Model Selection

### 3.1 Phase 4 — Baseline Models

Evaluated on `val.csv`. TF-IDF config: `min_df=2`, `max_features=50,000`, `sublinear_tf=True`.

| Model | Val Accuracy | Val Macro-F1 | F1-Neg | F1-Neu | F1-Pos |
|---|---|---|---|---|---|
| **TF-IDF char(3,5) + LR C=1.0** | **58.91%** | **58.81%** | 62.96% | 49.50% | 63.98% |
| TF-IDF word(1,2) + LinearSVC | 58.18% | 58.18% | 63.29% | 50.60% | 60.65% |
| TF-IDF word(1,2) + LR C=1.0 | 57.31% | 57.25% | 60.92% | 50.22% | 60.60% |
| TF-IDF word(1,2) + MNB | 57.38% | 56.92% | 61.66% | 47.23% | 61.87% |
| Stratified-random | 35.56% | 35.57% | — | — | — |
| Majority-class | 32.58% | 16.38% | — | — | — |

Phase 4 incumbent: **TF-IDF char(3,5) + LR C=1.0**, val macro-F1 = **58.81%**.

### 3.2 Phase 5 — Targeted Experiments (Fast Path)

Five single-fit experiments (no GridSearchCV, no CV folds) on the Phase 4 incumbent's char(3,5) feature family. Trained on `train.csv`, evaluated on `val.csv`.

| Run | Configuration | Val macro-F1 | Δ vs incumbent | F1-Neu |
|---|---|---|---|---|
| **Exp-D** | char(3,5) + LinearSVC C=1.0 | **60.11%** | **+1.30 pp** | 53.30% |
| Exp-B | char(3,5) + LR C=5.0 | 59.74% | +0.93 pp | 53.35% |
| Exp-C | FeatureUnion(char(3,5)+word(1,2)) + LR C=1.0 | 59.41% | +0.60 pp | 51.86% |
| Exp-E | char(3,5) + LR C=1.0 class_weight=balanced | 58.81% | 0.00 pp | 49.55% |
| Exp-A | char(3,5) + LR C=0.5 | 58.74% | −0.07 pp | 49.94% |

### 3.3 Selected Final Model

**`char(3,5) + LinearSVC C=1.0`** (Exp-D) — beats the Phase 4 incumbent by +1.30 pp on the validation split.

**Justification:** LinearSVC with character (3,5)-grams learns morphological and punctuation patterns that word n-grams miss (emoticons like `:( `, partial-word cues like `uck`, `sad`). LinearSVC's hinge loss is known to be more aggressive than logistic regression's log loss on sparse high-dimensional feature spaces, producing better margin separation when features are inherently noisy. The model trains in under 4 seconds on CPU, is fully interpretable via feature weights, and is deterministic given seed 42.

**Transformer track:** Skipped — deadline constraint. No GPU available (CPU-only PyTorch 2.12.1); a CPU-only DistilBERT fine-tune would require 30–90 minutes per epoch and would not be reproducible in a verification run. The classical pipeline is fully competitive on this short-text, balanced, 9,000-row dataset and provides interpretability for error analysis.

---

## 4. Training Methodology

| Parameter | Value |
|---|---|
| Feature type | Character (3,5)-grams, `analyzer="char_wb"` (pads word boundaries) |
| Vectorizer | `TfidfVectorizer`, `min_df=2`, `max_features=50,000`, `sublinear_tf=True` |
| Classifier | `LinearSVC`, `C=1.0`, `max_iter=2000`, `dual=True` |
| Random seed | 42 |
| Validation strategy | Phase 5: single train→val fit (no CV). Phase 4 baseline selection used train→val. |
| Final refit | On train + val combined (7,654 rows), after model selection locked on val |
| Final train time | 3.67 seconds on CPU |
| Saved artifact | `models/final_sentiment_model.pkl` (sklearn Pipeline: tfidf → clf) |

**No vectorizer or classifier was ever fit on validation or test data.**

---

## 5. Evaluation Metrics

Evaluated on `data/processed/test.csv` — opened **once**, at `2026-09-19T01:05:10Z`, after all model selection decisions were locked. Not re-evaluated after seeing results.

### 5.1 Full Test Metrics

| Metric | Value |
|---|---|
| **Test Accuracy** | **60.92%** |
| **Test Macro-F1** | **60.93%** |
| Test Macro-Precision | 60.94% |
| Test Macro-Recall | 60.99% |
| Test Weighted-F1 | 60.88% |
| F1 — Negative | 65.48% |
| F1 — Neutral | 56.61% |
| F1 — Positive | 60.70% |
| Val macro-F1 (reference) | 60.11% |
| Val / test delta | +0.82 pp |

The val/test gap of +0.82 pp confirms there is no significant overfitting or selection bias from the Phase 5 experiment sweep. Neutral remains the hardest class (56.61% F1), consistent across all phases.

### 5.2 Per-Class Classification Report

```
              precision    recall  f1-score   support

    Negative     0.6385    0.6720    0.6548       439
     Neutral     0.5648    0.5673    0.5661       453
    Positive     0.6247    0.5903    0.6070       454

    accuracy                         0.6092      1346
   macro avg     0.6094    0.6099    0.6093      1346
weighted avg     0.6091    0.6092    0.6088      1346
```

### 5.3 Baseline → Incumbent → Final Comparison (Test Split)

| Model | Test Accuracy | Test Macro-F1 | F1-Neg | F1-Neu | F1-Pos |
|---|---|---|---|---|---|
| Majority-class (trivial) | 32.62% | 16.40% | 49.19% | 0.00% | 0.00% |
| Stratified-random (trivial) | 33.73% | 33.74% | 32.47% | 35.77% | 32.97% |
| char(3,5)+LR C=1.0 (Phase 4 incumbent) | 60.33% | 60.35% | 62.47% | 57.73% | 60.83% |
| **char(3,5)+LinearSVC C=1.0 (final)** | **60.92%** | **60.93%** | **65.48%** | **56.61%** | **60.70%** |

The final model outperforms the trivial majority-class baseline by **+44.53 pp** macro-F1 and the Phase 4 incumbent by **+0.58 pp** on the test split.

---

## 6. Confusion Matrix

### Raw Counts

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 295 | 80 | 64 |
| **Neutral** | 99 | 257 | 97 |
| **Positive** | 68 | 118 | 268 |

*(Figure: `reports/figures/confusion_matrix.png`)*

### Row-Normalised (Recall per Class)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.672 | 0.182 | 0.146 |
| **Neutral** | 0.219 | 0.567 | 0.214 |
| **Positive** | 0.150 | 0.260 | 0.590 |

*(Figure: `reports/figures/confusion_matrix_normalized.png`)*

**Interpretation:** Neutral is most often confused with both Negative (21.9% of Neutral rows predicted Negative) and Positive (21.4% predicted Positive), confirming the expected Neutral boundary fuzziness. Negative and Positive are rarely confused with each other directly (14.6% and 15.0% cross-confusion), confirming the model distinguishes sentiment polarity well even when absolute intensity is ambiguous.

---

## 7. Error Analysis

Full report: `reports/error_analysis.md`. Script: `src/07_error_analysis.py`.

**526 errors on 1,346 test rows (39.1% overall error rate).**

### 7.1 Error Rate per Class

| True Class | Total | Errors | Error Rate |
|---|---|---|---|
| Negative | 439 | 144 | 32.8% |
| Neutral | 453 | 196 | **43.3%** |
| Positive | 454 | 186 | 41.0% |

**Most common confusion pair: Positive → Neutral** (118 errors = 22.4% of all errors).

### 7.2 Named Failure Categories (20 reviewed examples)

| Category | Count | % | Description |
|---|---|---|---|
| NEGATION | 4 | 20% | Negation scope not captured by char n-grams |
| MIXED_SENTIMENT | 4 | 20% | Post contains both positive and negative cues |
| FACTUAL_CONTEXT | 4 | 20% | Sentiment-bearing words in neutral/factual reporting |
| SARCASM_IRONY | 3 | 15% | Ironic phrasing misread at surface level |
| AMBIGUOUS_LABEL | 3 | 15% | Ground-truth label is genuinely debatable |
| MILD_POSITIVE | 2 | 10% | Low-intensity positive posts lack exclamatory markers |

All 20 examples are real rows from `reports/test_predictions.csv`; none are fabricated.

### 7.3 Key Quantitative Finding — Negation

Posts containing negation words have a **45.7% error rate** vs. **37.5%** for posts without (+8.2 pp lift). Of 526 errors, **121 (23.0%) involve negation words**, vs. 17.6% of correct predictions. This is the single strongest text-level predictor of error and reflects a fundamental limitation of bag-of-character-n-grams: "not bad" shares `bad` grams with genuinely negative posts, while "not " is a much weaker counterweight.

### 7.4 Model Interpretability — Top Features (LinearSVC)

| Class | Top positive features | Top negative features |
|---|---|---|
| **Negative** | `:( ` (2.29), ` no ` (1.98), `sad` (1.79), `uck` (1.77), `hate` (1.74) | `love ` (−1.11), `lov` (−0.94), `bless` (−0.92), `good ` (−0.87) |
| **Neutral** | ` or` (1.19), `an ` (1.11), ` qu` (1.06), `ring ` (1.06) | ` a ` (−1.82), `:)` (−1.35), `!! ` (−1.25), `lov` (−1.25) |
| **Positive** | `lov` (1.88), `:) ` (1.77), `e! ` (1.63), `y! ` (1.60), `fun` (1.45) | ` los` (−1.35), `:( ` (−1.32), `uck` (−1.29), `ad ` (−1.23) |

The `@user` token does not appear in top-15 features for any class, confirming the model has not over-indexed on the anonymisation placeholder. The `:30` gram (Neutral weight +1.02) is an incidental artefact — time references in scheduling posts are typically neutral.

### 7.5 Improvement Proposals

1. **Negation handling** (NEGATION, 4/20) — Prepend `NOT_` to tokens within a negation window, or switch to a contextual embedding model (DistilBERT). Negation posts have +8.2 pp error rate; fixing this could recover ~2 pp macro-F1.
2. **Positive→Neutral boundary** (MILD_POSITIVE, 118-count pair) — Apply threshold tuning on the LinearSVC decision function score rather than argmax. Highest-leverage single intervention.
3. **Sarcasm detection** (SARCASM_IRONY, 3/20) — Add a binary sarcasm feature from a dedicated sarcasm lexicon or SBERT sentence embeddings.
4. **Factual context** (FACTUAL_CONTEXT, 4/20) — Add VADER compound score as an auxiliary feature to disambiguate topic-mentioning from sentiment-expressing.
5. **Label noise** (AMBIGUOUS_LABEL, 3/20) — Re-annotate a stratified sample with 3 independent annotators; exclude or re-label instances with κ < 0.4.

---

## 8. Reproducibility

### Environment

| Package | Version |
|---|---|
| Python | 3.10.11 |
| scikit-learn | 1.4.2 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| matplotlib | 3.8.2 |

Install: `pip install -r requirements.txt`

### Commands to reproduce the full pipeline end-to-end

```bash
cd Round2/
python src/01_audit.py          # Phase 1: Data audit
python src/02_preprocess.py     # Phase 2: Cleaning + splits
python src/03_eda.py            # Phase 3: EDA figures
python src/04_baselines.py      # Phase 4: Baseline models
python src/05_experiments.py    # Phase 5: Model experiments + selection
python src/06_evaluate.py       # Phase 6: Test evaluation (opens test.csv ONCE)
python src/07_error_analysis.py # Phase 7: Error analysis
python src/08_inference_pipeline.py  # Phase 8: Inference demo + verification
```

All scripts are deterministic given `RANDOM_SEED = 42`. Re-running produces identical outputs.

### Key seeds

- `RANDOM_SEED = 42` used in all phases.
- `sklearn.utils.check_random_state(42)` seeded in all stochastic steps (DummyClassifier, LogisticRegression, LinearSVC).
- Train/val/test split derived by GroupShuffleSplit with `random_state=42`.

---

*Report compiled from generated phase reports. Every number traces to a script-generated artifact in `reports/`. No metrics are typed from memory.*
