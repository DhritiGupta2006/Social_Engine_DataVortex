# Phase 3: Sentiment Sanity Check

**Date:** 2026-09-22
**Dataset:** `bluesky_enriched.csv`
**Total Rows Inspected:** 20 random samples

## Review of Predictions (Round 2 Frozen Model)

The Round 2 model (TF-IDF char n-grams + LinearSVC) was applied to the raw Bluesky data without any retraining or modification. Overall, it captures explicit English sentiment reasonably well but struggles with cross-language data, mixed sentiments, and complex context (which is expected for a frozen, non-contextual model).

### 1. Negation Handling
* **Example:** *"Catchlight's own search reads every word you've written. iOS search never gets any of them."* 
* **Prediction:** Negative
* **Assessment:** The model successfully parsed the negation ("never gets any") and correctly identified the complaint.

* **Example:** *"Creo que no ha salido nunca un teléfono más bonito que el iPhone 4S"* (Translation: I think a prettier phone than the iPhone 4S has never come out)
* **Prediction:** Negative
* **Assessment:** Misclassification. The model likely keyed in on "no" and "nunca" without understanding the Spanish positive context. The model is English-centric.

### 2. Mixed / Ambiguous Sentiment
* **Example:** *"#xiaomi destroys #apple ... again! The new Xiaomi 18 Fold is better than the #iPhone Duo, and it is cheaper."*
* **Prediction:** Positive
* **Assessment:** This is a mixed-entity post. It is positive toward Xiaomi but negative toward Apple. The model output "Positive", which fails to capture the competitive nuance regarding the target product.

* **Example:** *"I hate my stupid job, but I also love my stupid job... Of course, it’s an iPhone launch week, so could come back on the 30th completely fucked"*
* **Prediction:** Negative
* **Assessment:** Highly mixed/ambiguous sentiment. The frustration dominates the text, so "Negative" is an acceptable prediction, though the post isn't strictly negative about the iPhone itself (just the retail stress).

### 3. Neutral / Factual
* **Example:** *"Preorders open on September 21, with devices shipping on October 4..."*
* **Prediction:** Neutral
* **Assessment:** Correctly identified as a factual, non-opinionated statement.

### Conclusion
The frozen model behaves exactly as a static TF-IDF + SVC classifier should. It captures obvious English sentiment (frustration, praise) but misses entity-level sentiment (praising a competitor) and struggles with non-English negations. **No retraining was performed**, and the model was preserved exactly as it was in Round 2.

21 posts have source `created_at` later than their `collected_at`; the cause is unknown.
