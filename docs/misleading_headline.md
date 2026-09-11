# What Is Misleading About My Headline Number?

In machine learning and customer support AI evaluations, headline numbers can be dangerously deceptive when reported without rigorous context. This document presents an honest, transparent deconstruction of our evaluation numbers, the original notebook's metrics, and the real-world operational realities of deploying an AI support agent.

---

## 1. The Original Notebook's 80% Number Was Not Intent Classification
The initial exploration notebook (`Rp_task_2.ipynb`, preserved under `notebooks/existing_model.ipynb`) reported an impressive **~80.01% Cross-Validation Accuracy** with Logistic Regression.

### Why That Number Was Fundamentally Misleading:
- **It solved the wrong problem**: The notebook extracted company `@mentions` via Named Entity Recognition (NER) and trained classifiers to predict the **Organization/Brand name** (AmazonHelp, AppleSupport, Uber_Support, Delta, SpotifyCares) from the customer's text.
- **Trivial Lexical Cues**: Distinguishing between an airline query (*"flight"*, *"boarding pass"*), a music app (*"playlist"*, *"offline songs"*), and an e-commerce platform (*"package"*, *"delivery"*) is a brand identification task heavily assisted by topical keywords.
- **Zero Applicability to Brand Support**: When building an AI support agent for a single brand (`AmazonHelp`), predicting that a customer is talking to Amazon provides zero operational utility. The real challenge is classifying the granular **intent** of an Amazon customer (e.g. distinguishing a delivery delay from a refund demand, payment dispute, or technical app crash).

> [!WARNING]
> Claiming "80% intent classification accuracy" based on the brand prediction notebook would be completely inaccurate. Our true intent classification accuracy on the independent AmazonHelp Golden Evaluation Set is **84.00%** (Macro F1: **83.76%**).

---

## 2. Deconstructing Our 84.0% Golden-Set Accuracy

While our tuned intent classifier achieved 84.00% accuracy and 0.8376 macro F1 across 9 classes on the 200-example golden set, calling this a single "headline metric" hides several critical nuances:

### A. Golden Set Size (N = 200)
- While 200 examples is standard for detailed qualitative audits, each class contains between 14 and 28 examples.
- **Statistical Variance**: A swing of just 2 misclassified examples in the `booking_or_travel` class (N = 14) shifts that class's recall by over **14%**.
- A 95% confidence interval on an 84% accuracy score over 200 samples spans approximately `[78.4%, 88.6%]`. Reporting 84.0% as a definitive fixed number overstates precision.

### B. Class Stratification vs. Production Class Imbalance
- The Golden Evaluation Set was purposefully stratified to evaluate all 9 classes fairly (roughly 10–14% per class).
- In live production Twitter traffic, class distributions are drastically skewed:
  - `delivery_or_delay` and generic inquiries (`other`) account for over **75%** of real-world volume.
  - Rare intents like `booking_or_travel` constitute less than **0.5%** of AmazonHelp volume.
- A model that performs well on an artificially balanced evaluation set will experience different aggregate error rates under live production distribution shifts.

### C. Semi-Automated Curation vs. Exhaustive Multi-Annotator Consensus
- Initial candidate pools were surfaced using lexical indicators before undergoing systematic auditing against the 9-class definitions.
- While audited to prevent mislabelling, true gold-standard benchmarking in research settings requires **inter-annotator agreement (e.g., Fleiss' kappa across 3+ independent human raters)**. Without multiple raters, boundary ambiguities (such as whether a delivery cancellation with an unpaid refund is primarily `delivery_or_delay` or `refund_or_compensation`) reflect single-annotator policy decisions.

### D. Single-Label Reduction of Multi-Intent Realities
- Twitter customers frequently write compound complaints:
  > *"My package was 5 days late, your driver was rude, and now I want my money back!"*
- Forcing a single label (`refund_or_compensation`) treats the correct identification of `delivery_or_delay` or `complaint_or_feedback` as a complete prediction error (0.0). Single-label accuracy undercounts functional model capability on compound requests.

---

## 3. Retrieval Success Metric Limitations (98.0% Top-5 Utility)
Our retrieval harness achieved a **98.0% Top-5 Useful Rate** on 50 sample queries. Why this must be interpreted carefully:
1. **Shallow Relevance vs. Solution Completeness**: A historical support reply might say *"Please DM us your order number so we can help."* This is technically "useful" (it matches standard social support protocol), but it does not provide an autonomous factual resolution.
2. **Privacy Redaction in Public Tweets**: Historical support tweets rarely reveal the final private resolution due to Twitter's 280-character limit and privacy policies directing users to private links (`https://t.co/...`). Measuring lexical similarity on public tweets measures adherence to public triaging scripts, not end-to-end case resolution.

---

## 4. LLM-as-a-Judge Limitations (4.67 / 5.0 Average)
Our LLM-as-a-judge scored generated replies at an average of **4.67 / 5.0**:
1. **Self-Consistency and LLM Sycophancy**: LLM judges inherently prefer fluent, polite, grammatically perfect language—qualities that LLM generators naturally produce. High scores for Tone (5.0) and Groundedness (5.0) reflect fluency and safety adherence, not necessarily that the customer felt their issue was resolved.
2. **Circular Validation Risk**: Using an LLM to evaluate an LLM's output risks shared blind spots. If the generator makes a subtle domain-specific policy assumption that sounds convincing, the LLM judge is likely to accept it as grounded unless checked against strict operational rules or human expert ratings.

---

## 5. Why One Metric Never Represents Support Quality
An AI customer support system cannot be judged by accuracy alone. A system with **95% intent accuracy** that hallucinates refund policies on the remaining 5% can cause severe financial liability and brand reputation damage.

Conversely, our system pairs **84.0% intent accuracy** with a **Conservative Escalation Policy** that routes 62.0% of sensitive, ambiguous, or financial requests to human specialists. In customer support:
$$\text{System Value} = (\text{Accurate Auto-Resolution} \times \text{Volume}) - (\text{Hallucinated Commitments} \times \text{Cost})$$

Safety, groundedness, and conservative escalation are vastly more critical than maximizing an isolated headline classification accuracy.
