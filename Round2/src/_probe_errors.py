"""Scratch probe — error pattern analysis to inform 07_error_analysis.py"""
import pandas as pd
import numpy as np
import pickle
import re

df  = pd.read_csv("reports/test_predictions.csv")
err = df[df["correct"] == False].copy()
ok  = df[df["correct"] == True].copy()
print(f"Total rows: {len(df)}  Errors: {len(err)}  Correct: {len(ok)}")

# ── Text feature helpers
NEG_PAT = re.compile(
    r"\b(not|no|never|can't|cannot|won't|isn't|aren't|wasn't|weren't|"
    r"don't|doesn't|didn't|nobody|nothing|neither|nor|hardly|barely|scarcely)\b",
    re.I,
)
SARC_PAT = re.compile(
    r"\b(yeah right|sure|totally|obviously|clearly|great job|well done|"
    r"nice one|as if|whatever)\b|:-\)|;\)|:P|\blol\b|\blmao\b",
    re.I,
)

def featurize(df_):
    d = df_.copy()
    d["word_count"]   = d["text_clean"].apply(lambda t: len(str(t).split()))
    d["has_user"]     = d["text_clean"].apply(lambda t: "@user" in str(t))
    d["has_hashtag"]  = d["text_clean"].apply(lambda t: "#" in str(t))
    d["has_negation"] = d["text_clean"].apply(lambda t: bool(NEG_PAT.search(str(t))))
    d["has_sarcasm"]  = d["text_clean"].apply(lambda t: bool(SARC_PAT.search(str(t))))
    return d

df_feat  = featurize(df)
err_feat = featurize(err)
ok_feat  = featurize(ok)

print("\n=== TEXT FEATURES: correct vs incorrect ===")
for feat in ["word_count", "has_user", "has_hashtag", "has_negation", "has_sarcasm"]:
    if feat == "word_count":
        print(f"  {feat}: correct={ok_feat[feat].mean():.2f}  incorrect={err_feat[feat].mean():.2f}")
    else:
        print(f"  {feat}: correct={ok_feat[feat].mean()*100:.1f}%  incorrect={err_feat[feat].mean()*100:.1f}%")

print("\n=== ERROR RATE PER TRUE CLASS ===")
for cls in ["Negative", "Neutral", "Positive"]:
    sub   = df[df["true_label"] == cls]
    n_err = (sub["correct"] == False).sum()
    print(f"  {cls}: {n_err}/{len(sub)} errors = {n_err/len(sub)*100:.1f}%")

print("\n=== CONFUSION PAIRS ===")
pairs = (err.groupby(["true_label", "predicted_label"])
           .size().reset_index(name="count")
           .sort_values("count", ascending=False))
print(pairs.to_string(index=False))
top = pairs.iloc[0]
print(f"\n  Most common pair: {top['true_label']} -> {top['predicted_label']}  count={top['count']}  ({top['count']/len(err)*100:.1f}% of all errors)")

print("\n=== NEGATION ERRORS ===")
neg_errors = err_feat[err_feat["has_negation"]]
print(f"  Errors with negation: {len(neg_errors)} / {len(err)} ({len(neg_errors)/len(err)*100:.1f}%)")
print(f"  Correct with negation: {ok_feat['has_negation'].sum()} / {len(ok)} ({ok_feat['has_negation'].mean()*100:.1f}%)")

print("\n=== SARCASM ERRORS ===")
sarc_errors = err_feat[err_feat["has_sarcasm"]]
print(f"  Errors with sarcasm markers: {len(sarc_errors)} / {len(err)} ({len(sarc_errors)/len(err)*100:.1f}%)")

print("\n=== SAMPLE ERRORS BY PAIR ===")
for (true_lbl, pred_lbl), group in err.groupby(["true_label", "predicted_label"]):
    print(f"\n  {true_lbl} -> {pred_lbl}  (n={len(group)})")
    for _, row in group.head(5).iterrows():
        print(f"    [{row['text_id']}] {row['text_clean'][:100]}")

print("\n=== FEATURE WEIGHTS (LinearSVC) ===")
with open("models/final_sentiment_model.pkl", "rb") as fh:
    model = pickle.load(fh)

tfidf = model.named_steps["tfidf"]
clf   = model.named_steps["clf"]
feats = tfidf.get_feature_names_out()
classes = clf.classes_
coefs = clf.coef_   # shape: (n_classes, n_features) for multiclass

TOP_N = 20
for i, cls in enumerate(classes):
    w = coefs[i]
    top_pos_idx = np.argsort(w)[-TOP_N:][::-1]
    top_neg_idx = np.argsort(w)[:TOP_N]
    print(f"\n  Class: {cls}")
    print(f"  Top positive features: {[(feats[j], round(w[j],4)) for j in top_pos_idx]}")
    print(f"  Top negative features: {[(feats[j], round(w[j],4)) for j in top_neg_idx]}")

print("\nDone.")
