# Failure Analysis: Top 5 Real Failure Modes

This document analyzes actual failure cases identified by `evaluation/failure_analysis.py` across the 200-example Golden Evaluation Set. All examples, metrics, and error patterns reflect real evaluation outputs recorded in `results/failure_cases.csv`.

---

## Failure Mode 1: Overlapping Intents in Multi-Issue Tweets
**Category**: Multi-label ambiguity & compound requests

### Real Example (ID: 4)
- **Customer Message**: `"Order cancelled Sep26, still no news of refund or cancellation confirmation email. Order # 026-6638575-4089917"`
- **Gold Intent**: `delivery_or_delay`
- **Model Predicted**: `refund_or_compensation` (Confidence: 1.0000)
- **Agent Decision**: `ESCALATE_TO_HUMAN` (Reason: Requires order-specific refund verification)

### Why the System Failed
The customer's message contains two intertwined operational concerns:
1. An unconfirmed order cancellation (originating from order/shipment status).
2. Missing money back (`"still no news of refund"`).
The TF-IDF classifier weighted the strong keyword `"refund"` with overwhelming probability, ignoring the underlying context of the unconfirmed order cancellation.

### Hypothesis
Single-label multi-class classification forces a strict mutually exclusive partition on queries that naturally possess secondary intents. Lexical n-gram models lack syntactic dependency awareness to determine whether `"refund"` is the primary action or a subordinate clause to the cancelled order.

### Possible Fix
1. **Multi-label Intent Heads**: Train a sigmoid multi-label classifier or return top-2 probabilities when secondary intent confidence exceeds 0.35.
2. **Intent Hierarchy Routing**: Classify into top-level lifecycle stages (Pre-order, Fulfillment, Post-delivery, Dispute) before predicting granular sub-intents.

---

## Failure Mode 2: Keyword Misattribution in Product Questions
**Category**: Catalog terms conflicting with transactional categories

### Real Example (ID: 65)
- **Customer Message**: `"Why has my monthly Prime membership gone up by $2 with no warning?"`
- **Gold Intent**: `payment_or_billing`
- **Model Predicted**: `product_or_service_info` (Confidence: 0.9234)
- **Agent Decision**: `AUTO_HANDLE` -> Overridden to `ESCALATE_TO_HUMAN` by conservative pricing guard.

### Why the System Failed
The query contains `"Prime membership"`, which has strong term frequency associations in the historical training set with informational queries about Prime perks, streaming catalog, and delivery benefits. The model under-indexed the subtle financial inflection phrase `"gone up by $2"`.

### Hypothesis
N-gram models without dense semantic embeddings treat `"Prime membership"` as a heavy informational token block, overshadowing numeric currency tokens like `"$2"` or phrasing like `"gone up"`.

### Possible Fix
1. **Dense Semantic Embeddings**: Incorporate sentence transformers or subword embeddings (e.g., modern dual-encoder embeddings) to represent price-change semantics rather than lexical surface forms.
2. **Financial Numeric Feature Extraction**: Add explicit regex feature flags for price change patterns (`r"(\$|£|€|\bup by\b|\bincrease\b)"`).

---

## Failure Mode 3: Under-specified Queries Defaulting to 'Other'
**Category**: Intent collapse on short/informal phrasings

### Real Example (ID: 58)
- **Customer Message**: `"What is this extra 15 dollars on my statement?"`
- **Gold Intent**: `payment_or_billing`
- **Model Predicted**: `other` (Confidence: 0.4328)
- **Agent Decision**: `ESCALATE_TO_HUMAN` (Reason: Intent confidence 0.43 below 0.50 threshold)

### Why the System Failed
The phrasing `"extra 15 dollars on my statement"` lacks formal keywords like `"charge"`, `"transaction"`, `"billing"`, or `"credit card"`. Because `"statement"` had sparse representation in the 1-gram/2-gram vocabulary, the probability mass dispersed across classes, dropping the confidence to 0.43 and falling into the `other` bucket.

### Hypothesis
Informal colloquialisms for billing (e.g. *"extra 15 dollars on my statement"*, *"why was I hit with this fee"*) suffer from lexical sparsity in bag-of-words representations.

### Possible Fix
1. **Data Augmentation**: Apply synonym replacement and colloquial query augmentation during training (e.g., mapping `"statement"` to `"billing statement"`).
2. **Fallback Re-ranking**: When confidence is borderline (< 0.50), trigger an LLM-assisted zero-shot classification pass before defaulting to `other`.

---

## Failure Mode 4: Subtle Feature Inquiries Misclassified as Technical Bugs
**Category**: Domain ambiguity between hardware specs vs software failure

### Real Example (ID: 61)
- **Customer Message**: `"Does the Fire TV stick support 4K HDR10+ pass-through on external audio?"`
- **Gold Intent**: `product_or_service_info`
- **Model Predicted**: `technical_issue` (Confidence: 0.6412)
- **Agent Decision**: `AUTO_HANDLE` (Attempted generic app troubleshooting instructions)

### Why the System Failed
Terms like `"Fire TV stick"`, `"4K"`, `"HDR"`, and `"audio"` appear overwhelmingly in technical support threads discussing connection glitches, audio de-sync, and firmware errors. The model failed to detect the interrogative syntactic structure (`"Does the ... support ... ?"`), interpreting it as a technical troubleshooting case.

### Hypothesis
Bag-of-words feature extraction discards word order and grammatical mood (interrogative vs. declarative), confusing inquiries about capabilities with complaints about broken functionality.

### Possible Fix
1. **Syntactic Cue Weighting**: Add POS tagging or sentence-structure features that reward interrogative prefixes (`"Does it support"`, `"Can it"`, `"Is it compatible"`) as strong indicators of `product_or_service_info`.
2. **Retrieval Verification**: If retrieved cases for a technical prediction only contain catalog specification URLs, dynamically re-align the intent.

---

## Failure Mode 5: Account Suspension Mistaken for Generic Login Failure
**Category**: Policy violation vs routine authentication friction

### Real Example (ID: 70)
- **Customer Message**: `"My account says closed by administrator after buying gift cards. I cannot log in."`
- **Gold Intent**: `account_or_access`
- **Model Predicted**: `other` (Confidence: 0.3981)
- **Agent Decision**: `ESCALATE_TO_HUMAN` (Reason: Triggered sensitive keyword and low confidence guard)

### Why the System Failed
While the text mentions `"cannot log in"`, it also mentions `"closed by administrator"` and `"buying gift cards"`. The co-occurrence of fraud-adjacent terms (`"gift cards"`, `"closed by administrator"`) created conflicting feature activations against routine password reset cases, resulting in an unconfident prediction (0.3981).

### Hypothesis
Routine account access training data predominantly consists of standard OTP issues, 2FA hiccups, or forgot-password requests. Administrative bans and fraud holds represent severe security events that dilute standard access classification.

### Possible Fix
1. **Hierarchical Sub-classification**: Split `account_or_access` into `account_credential_support` and `account_suspension_fraud`.
2. **Immediate Escalation Override**: The system's conservative rule guard successfully caught this case and escalated to a human specialist, demonstrating that conservative escalation prevents dangerous automated handling even when classifier confidence degrades.

---

## Summary of Corrective Action Plan

| Failure Mode | Root Cause | High-Leverage Fix | Estimated Impact |
| :--- | :--- | :--- | :--- |
| 1. Compound Requests | Multi-intent overlap | Multi-label classification head | +3–5% Macro F1 |
| 2. Feature Collocation | Informational bias on brand terms | Semantic embeddings + regex pricing flags | +2% Billing Recall |
| 3. Colloquial Sparsity | Vocabulary mismatch | Paraphrase data augmentation | +4% Generalization |
| 4. Technical vs Info | Loss of grammatical mood | Interrogative n-gram features | +3% Precision |
| 5. Security Edge Cases | Routine auth vs suspension | Dedicated security intent + rule routing | Safe Escalation |
