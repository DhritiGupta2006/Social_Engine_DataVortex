"""
04_baselines.py — DATA VORTEX A'26, Round 2
Phase 4: Baseline Models

Inputs  : data/processed/train.csv, data/processed/val.csv
          artifacts/label_encoders.json
          reports/eda_report.md (informs min_df / max_features choices)

Outputs : reports/baseline_results.md
          reports/baseline_results.csv

NOTE: test.csv is never loaded or referenced in this script.

Run from the Round2/ project root:
    python src/04_baselines.py
"""

import sys
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TRAIN_CSV    = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV      = PROJECT_ROOT / "data" / "processed" / "val.csv"
# test.csv intentionally never referenced
ENCODERS_JSON = PROJECT_ROOT / "artifacts" / "label_encoders.json"
REPORT_MD    = PROJECT_ROOT / "reports" / "baseline_results.md"
REPORT_CSV   = PROJECT_ROOT / "reports" / "baseline_results.csv"

RANDOM_SEED  = 42
SENT_CLASSES = ["Negative", "Neutral", "Positive"]

# ── EDA-informed TF-IDF config (from Phase 3 vocab-size table) ────────────────
# min_df=2 -> vocab=6,452 tokens  (recommended start)
# max_features=50_000 is a non-binding upper ceiling — vocab is ~6k so it has no effect
# @user and <url> tokens: kept (Phase 3 showed they add negligible signal but don't hurt)
TFIDF_MIN_DF     = 2
TFIDF_MAX_FEAT   = 50_000
TFIDF_SUBLINEAR  = True   # sublinear_tf=True consistently helps on sparse social text

# =============================================================================
# LOAD DATA
# =============================================================================
print("[INFO] Loading train and val splits ...")
train_df = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
val_df   = pd.read_csv(VAL_CSV,   dtype=str, keep_default_na=False)

X_train = train_df["text_clean"].tolist()
y_train = train_df["sentiment_label"].tolist()
X_val   = val_df["text_clean"].tolist()
y_val   = val_df["sentiment_label"].tolist()

print("  Train: " + str(len(X_train)) + " rows")
print("  Val  : " + str(len(X_val))   + " rows")

# Verify class balance in val (should be ~1:1:1)
val_counts = pd.Series(y_val).value_counts()
print("  Val class counts: " + str(val_counts.to_dict()))
for cls in SENT_CLASSES:
    pct = 100.0 * val_counts.get(cls, 0) / len(y_val)
    assert abs(pct - 33.33) < 6.0, \
        "Class balance broken in val split! " + cls + " = " + str(round(pct, 1)) + "%"
print("  Val balance check: OK (all classes within 6pp of 33.3%)")

# =============================================================================
# EVALUATION HELPER
# =============================================================================
def evaluate(name, vec_desc, y_true, y_pred, train_secs):
    """Return a metrics dict for one model."""
    acc     = accuracy_score(y_true, y_pred)
    mac_p   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mac_r   = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mac_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_cls = {cls: f1_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)
               for cls in SENT_CLASSES}
    return {
        "model":        name,
        "vectorizer":   vec_desc,
        "accuracy":     round(acc,   4),
        "macro_P":      round(mac_p, 4),
        "macro_R":      round(mac_r, 4),
        "macro_F1":     round(mac_f1, 4),
        "F1_Negative":  round(per_cls["Negative"], 4),
        "F1_Neutral":   round(per_cls["Neutral"],  4),
        "F1_Positive":  round(per_cls["Positive"], 4),
        "train_secs":   round(train_secs, 2),
    }

results = []

# =============================================================================
# SECTION A — TRIVIAL BASELINES
# =============================================================================
print("\n=== TRIVIAL BASELINES ===")

# A1: Majority-class classifier
print("\n[A1] Majority-class classifier ...")
t0 = time.perf_counter()
majority_cls = pd.Series(y_train).value_counts().index[0]
y_pred_maj   = [majority_cls] * len(y_val)
t1 = time.perf_counter()
r = evaluate("Majority-class", "n/a", y_val, y_pred_maj, t1 - t0)
results.append(r)
print("  Accuracy=" + str(r["accuracy"]) + "  macro-F1=" + str(r["macro_F1"]))
print("  Majority class: '" + majority_cls + "'")

# Sanity check: ~33% accuracy on balanced 3-class problem
assert abs(r["accuracy"] - 1/3) < 0.06, \
    "SANITY FAIL: majority accuracy should be ~33% for balanced split, got " + str(r["accuracy"])
print("  Sanity check: majority accuracy ~33% -> OK")

# A2: Stratified random classifier (uses training class distribution)
print("\n[A2] Stratified random classifier ...")
t0 = time.perf_counter()
strat_clf = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
strat_clf.fit(X_train, y_train)
y_pred_strat = strat_clf.predict(X_val)
t1 = time.perf_counter()
r = evaluate("Stratified-random", "n/a", y_val, y_pred_strat, t1 - t0)
results.append(r)
print("  Accuracy=" + str(r["accuracy"]) + "  macro-F1=" + str(r["macro_F1"]))

# =============================================================================
# SECTION B — CLASSICAL BASELINES
# =============================================================================
print("\n=== CLASSICAL BASELINES (train->val, Pipeline, no test data used) ===")

# B1: TF-IDF word (1,2)-grams + Multinomial Naive Bayes
print("\n[B1] TF-IDF word (1,2)-grams + Multinomial Naive Bayes ...")
pipe_mnb = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEAT,
        sublinear_tf=TFIDF_SUBLINEAR,
    )),
    ("clf", MultinomialNB()),
])
t0 = time.perf_counter()
pipe_mnb.fit(X_train, y_train)
t1 = time.perf_counter()
y_pred_mnb = pipe_mnb.predict(X_val)
r = evaluate("TF-IDF(1,2) + MNB", "word, min_df=2, sublinear_tf", y_val, y_pred_mnb, t1 - t0)
results.append(r)
print("  Accuracy=" + str(r["accuracy"]) + "  macro-F1=" + str(r["macro_F1"]))
print(classification_report(y_val, y_pred_mnb, target_names=SENT_CLASSES, zero_division=0))

# B2: TF-IDF word (1,2)-grams + Logistic Regression
print("\n[B2] TF-IDF word (1,2)-grams + Logistic Regression ...")
pipe_lr = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEAT,
        sublinear_tf=TFIDF_SUBLINEAR,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
    )),
])
t0 = time.perf_counter()
pipe_lr.fit(X_train, y_train)
t1 = time.perf_counter()
y_pred_lr = pipe_lr.predict(X_val)
r = evaluate("TF-IDF(1,2) + LR", "word, min_df=2, sublinear_tf", y_val, y_pred_lr, t1 - t0)
results.append(r)
print("  Accuracy=" + str(r["accuracy"]) + "  macro-F1=" + str(r["macro_F1"]))
print(classification_report(y_val, y_pred_lr, target_names=SENT_CLASSES, zero_division=0))

# B3: TF-IDF word (1,2)-grams + Linear SVM
print("\n[B3] TF-IDF word (1,2)-grams + LinearSVC ...")
pipe_svm = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEAT,
        sublinear_tf=TFIDF_SUBLINEAR,
    )),
    ("clf", LinearSVC(
        max_iter=2000,
        random_state=RANDOM_SEED,
        C=1.0,
    )),
])
t0 = time.perf_counter()
pipe_svm.fit(X_train, y_train)
t1 = time.perf_counter()
y_pred_svm = pipe_svm.predict(X_val)
r = evaluate("TF-IDF(1,2) + SVM", "word, min_df=2, sublinear_tf", y_val, y_pred_svm, t1 - t0)
results.append(r)
print("  Accuracy=" + str(r["accuracy"]) + "  macro-F1=" + str(r["macro_F1"]))
print(classification_report(y_val, y_pred_svm, target_names=SENT_CLASSES, zero_division=0))

# B4: TF-IDF CHARACTER (3,5)-grams + Logistic Regression
print("\n[B4] TF-IDF char (3,5)-grams + Logistic Regression ...")
pipe_char_lr = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",          # char_wb pads word boundaries — better for social text
        ngram_range=(3, 5),
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEAT,
        sublinear_tf=TFIDF_SUBLINEAR,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
    )),
])
t0 = time.perf_counter()
pipe_char_lr.fit(X_train, y_train)
t1 = time.perf_counter()
y_pred_char = pipe_char_lr.predict(X_val)
r = evaluate("TF-IDF char(3,5) + LR", "char_wb, min_df=2, sublinear_tf", y_val, y_pred_char, t1 - t0)
results.append(r)
print("  Accuracy=" + str(r["accuracy"]) + "  macro-F1=" + str(r["macro_F1"]))
print(classification_report(y_val, y_pred_char, target_names=SENT_CLASSES, zero_division=0))

# =============================================================================
# VALIDATE: every classical baseline beats both trivial baselines
# =============================================================================
print("\n=== VALIDATION ===")
trivial_best_f1 = max(results[0]["macro_F1"], results[1]["macro_F1"])
for r in results[2:]:
    assert r["macro_F1"] > trivial_best_f1, \
        r["model"] + " macro-F1 (" + str(r["macro_F1"]) + ") does NOT beat trivial (" + str(trivial_best_f1) + ")!"
    print("  " + r["model"] + " macro-F1=" + str(r["macro_F1"]) + " > trivial " + str(trivial_best_f1) + " [OK]")

# =============================================================================
# SORT BY MACRO-F1 AND IDENTIFY INCUMBENT
# =============================================================================
results_sorted = sorted(results, key=lambda x: -x["macro_F1"])
incumbent = results_sorted[0]
print("\n=== INCUMBENT MODEL ===")
print("  Name     : " + incumbent["model"])
print("  Vec      : " + incumbent["vectorizer"])
print("  Accuracy : " + str(incumbent["accuracy"]))
print("  macro-F1 : " + str(incumbent["macro_F1"]))
print("  Train sec: " + str(incumbent["train_secs"]))

# =============================================================================
# SAVE RESULTS CSV
# =============================================================================
results_df = pd.DataFrame(results_sorted)
results_df.to_csv(REPORT_CSV, index=False)
print("\n[INFO] Saved " + str(REPORT_CSV))

# =============================================================================
# WRITE baseline_results.md
# =============================================================================
print("[INFO] Writing baseline_results.md ...")

def pct(v):
    return str(round(v * 100, 2)) + "%"

lines = []
lines.append("# Baseline Results \u2014 DATA VORTEX A\u201926 Round 2")
lines.append("**Phase 4: Baseline Models**")
lines.append("Generated by `src/04_baselines.py`. Evaluated on `val.csv`. "
             "`test.csv` not loaded or referenced.")
lines.append("")
lines.append("Random seed: " + str(RANDOM_SEED) + ". "
             "TF-IDF config: `min_df=" + str(TFIDF_MIN_DF) + "`, "
             "`max_features=" + str(TFIDF_MAX_FEAT) + "`, "
             "`sublinear_tf=" + str(TFIDF_SUBLINEAR) + "` "
             "(informed by Phase 3 vocab-size table: min_df=2 -> 6,452 tokens).")
lines.append("")
lines.append("---")
lines.append("")

# Full comparison table
lines.append("## Comparison Table (sorted by macro-F1, descending)")
lines.append("")
lines.append("| Model | Vectorizer | Accuracy | Macro-P | Macro-R | **Macro-F1** | F1-Neg | F1-Neu | F1-Pos | Train (s) |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
for r in results_sorted:
    lines.append(
        "| " + r["model"] +
        " | " + r["vectorizer"] +
        " | " + pct(r["accuracy"]) +
        " | " + pct(r["macro_P"]) +
        " | " + pct(r["macro_R"]) +
        " | **" + pct(r["macro_F1"]) + "**" +
        " | " + pct(r["F1_Negative"]) +
        " | " + pct(r["F1_Neutral"]) +
        " | " + pct(r["F1_Positive"]) +
        " | " + str(r["train_secs"]) + "s |"
    )
lines.append("")

# Trivial vs classical comparison block
lines.append("### Trivial baseline reference")
lines.append("")
for r in [results[0], results[1]]:
    lines.append("- **" + r["model"] + "**: accuracy=" + pct(r["accuracy"])
                 + ", macro-F1=" + pct(r["macro_F1"]))
lines.append("")
lines.append("> For a perfectly balanced 3-class problem the majority-class accuracy is exactly "
             "33.33% and macro-F1 is ~0% (it scores 0 on the two minority classes). "
             "Every real model must substantially beat both. "
             "The stratified random classifier scores ~33% accuracy and ~33% macro-F1 by construction.")
lines.append("")
lines.append("---")
lines.append("")

# Per-model narrative
lines.append("## Per-Model Notes")
lines.append("")
for r in results_sorted:
    lines.append("### " + r["model"])
    lines.append("")
    lines.append("- Vectorizer: " + r["vectorizer"])
    lines.append("- Accuracy: **" + pct(r["accuracy"]) + "**  |  Macro-F1: **" + pct(r["macro_F1"]) + "**")
    lines.append("- Per-class F1: Negative=" + pct(r["F1_Negative"])
                 + ", Neutral=" + pct(r["F1_Neutral"])
                 + ", Positive=" + pct(r["F1_Positive"]))
    lines.append("- Training time: " + str(r["train_secs"]) + "s")
    lines.append("")

lines.append("---")
lines.append("")

# Incumbent statement
lines.append("## Incumbent Model")
lines.append("")
lines.append("The best-performing classical configuration, which Phase 5 experiments must beat, is:")
lines.append("")
lines.append("> **" + incumbent["model"] + "** ("
             + incumbent["vectorizer"] + ")  ")
lines.append("> Validation accuracy = **" + pct(incumbent["accuracy"]) + "**,  "
             + "validation macro-F1 = **" + pct(incumbent["macro_F1"]) + "**.")
lines.append("")
lines.append("This is the performance floor that all Phase 5 model variants are measured against. "
             "Any Phase 5 model that does not beat this macro-F1 on the same val split is not "
             "selected as the final model regardless of its accuracy, since accuracy alone is "
             "insufficient per Phase 1/3's reasoning.")
lines.append("")
lines.append("Training time: **" + str(incumbent["train_secs"]) + "s** — extremely fast; "
             "the time budget for Phase 5 transformer experiments is therefore not constrained "
             "by this baseline.")
lines.append("")
lines.append("---")
lines.append("")

# Validation checks
lines.append("## Validation Checks")
lines.append("")
lines.append("| Check | Result |")
lines.append("|---|---|")
lines.append("| Majority baseline accuracy ~33% | **" + pct(results[0]["accuracy"]) + "** \u2714 |")
classical_beat = all(r["macro_F1"] > trivial_best_f1 for r in results[2:])
lines.append("| All classical baselines beat trivial | **" + str(classical_beat) + "** \u2714 |")
lines.append("| `test.csv` referenced in script | **No** \u2714 |")
lines.append("| macro-F1 reported for every model | **Yes** \u2714 |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("*Report generated by `src/04_baselines.py` \u2014 re-run to regenerate.*")

REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
print("[DONE] baseline_results.md written.")

# =============================================================================
print("\n=== Phase 4 complete \u2014 Definition of Done ===")
print("  [OK] reports/baseline_results.md")
print("  [OK] reports/baseline_results.csv")
print()
print("  Trivial baselines:")
print("    Majority-class  accuracy=" + pct(results[0]["accuracy"])
      + "  macro-F1=" + pct(results[0]["macro_F1"]))
print("    Stratified-rand accuracy=" + pct(results[1]["accuracy"])
      + "  macro-F1=" + pct(results[1]["macro_F1"]))
print()
print("  Classical baselines (val macro-F1):")
for r in results_sorted[2:] if results_sorted[0]["model"] in ["Majority-class","Stratified-random"] else results_sorted:
    if r["model"] not in ["Majority-class", "Stratified-random"]:
        print("    " + r["model"].ljust(28) + "  " + pct(r["macro_F1"]))
print()
print("  INCUMBENT: " + incumbent["model"] + "  val macro-F1=" + pct(incumbent["macro_F1"]))
print("  Phase 5 must beat macro-F1=" + pct(incumbent["macro_F1"]) + " to replace incumbent.")
