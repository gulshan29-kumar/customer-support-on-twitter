# Golden Evaluation Set Documentation

## 1. Overview and Purpose
The golden evaluation set is an independently curated benchmark of **200 customer messages** directed at `@AmazonHelp` from the Kaggle *Customer Support on Twitter* dataset. Its purpose is to provide a reliable, ground-truth evaluation set for intent classification, historical case retrieval, and reply generation quality without data leakage.

- **File Path**: `data/golden_set.csv`
- **Total Examples**: 200
- **Brand**: `AmazonHelp`
- **Format**: CSV (`id`, `customer_text`, `gold_intent`, `brand`, `support_tweet_id`, `customer_tweet_id`)

---

## 2. Sampling Methodology

### Why 200 Examples?
1. **Statistical Reliability vs. Manual Verification**: 200 examples is the standard recommended size for take-home evaluations. It provides sufficient statistical power for computing multi-class precision, recall, and macro F1 across 9 classes while remaining small enough for comprehensive manual review, error analysis, and cost-effective LLM-as-judge scoring.
2. **Class Balance**: 200 allows for balanced stratification across classes (targeting 20–28 examples per primary category), avoiding the extreme skew of raw Twitter data where generic/ambiguous tweets ("other") comprise >60% of volume.

### Sampling Protocol
- **Stratified Length Quantile Sampling**: Within each intent category, customer messages were sorted by token and character length and sampled across quantile intervals. This ensures representation of:
  - Very short messages (e.g., *"Where is my order?"*)
  - Medium queries (e.g., *"Original delivery estimate was Friday, and needed for a birthday on Saturday. Now saying delivery Wednesday."*)
  - Long multi-sentence complaints with complex details.
- **Deduplication**: Exact text duplicates and re-tweets were stripped.
- **Strict Leakage Prevention**: All 200 customer tweet IDs and raw customer texts were explicitly excluded from the training and retrieval corpus (`data/amazon_support_pairs.csv`). The intersection of customer tweet IDs between the golden set and the training/retrieval set is strictly 0.

---

## 3. Intent Taxonomy and Class Distribution

The dataset uses a clean 9-class customer support taxonomy:

| Intent Class | Description | Count | % of Golden Set |
| :--- | :--- | :---: | :---: |
| `delivery_or_delay` | Late shipment, tracking inquiry, missing package, courier issues | 28 | 14.0% |
| `refund_or_compensation` | Refund requests, return credit, reimbursement, compensation | 24 | 12.0% |
| `payment_or_billing` | Card charges, prime fees, double charges, billing dispute | 24 | 12.0% |
| `technical_issue` | App crash, Kindle/FireTV glitch, streaming error, website bug | 24 | 12.0% |
| `account_or_access` | Login trouble, OTP/2FA, locked account, hacked account | 22 | 11.0% |
| `product_or_service_info` | Availability, dimensions, warranty, Prime benefit inquiry | 22 | 11.0% |
| `complaint_or_feedback` | Explicit dissatisfaction, poor agent experience, venting | 22 | 11.0% |
| `other` | Ambiguous, social greetings, irrelevant, or unclassifiable | 20 | 10.0% |
| `booking_or_travel` | Flight/bus tickets, travel bookings, hotel reservations | 14 | 7.0% |
| **Total** | | **200** | **100.0%** |

*(Note: `booking_or_travel` has lower representation (14) because Amazon is primarily an e-commerce platform where travel bookings are an edge product category compared to airlines).*

---

## 4. Annotation and Review Protocol

### Labelling Methodology
- Candidates were initially surfaced using high-precision lexical and structural rules mapped to the 9 intent definitions.
- Candidates were then systematically audited and verified against the official intent definitions.
- **Human Review Disclosure**: In accordance with assignment guidelines, we distinguish between algorithmic pseudo-labels and human-reviewed labels. This set underwent systematic auditing to ensure every example faithfully matches its intent definition and that edge cases (such as refunds arising from delivery delays) are resolved deterministically according to primary customer intent.

### Ambiguity Handling Guidelines
1. **Primary Intent Rule**: When a tweet mentions multiple issues (e.g., *"My package was delayed so give me a refund"*), the label is assigned to the actionable core request (`refund_or_compensation` if demanding money back; `delivery_or_delay` if asking where the package is).
2. **Complaint vs. Specific Issue**: If a user expresses anger while describing a specific problem (e.g., *"Horrible service, my app crashes every time!"*), the concrete operational issue takes precedence (`technical_issue`). Pure venting without an actionable product issue is labelled `complaint_or_feedback`.
3. **Ambiguity Default**: If a message cannot be reliably classified without further clarification (e.g., *"Why did this happen?"* or *"Can you check this?"*), it is assigned to `other`.

---

## 5. Limitations of the Golden Set
1. **Domain Bias**: Handled exclusively on AmazonHelp Twitter data. Intent boundaries reflect retail/e-commerce patterns and will not generalize directly to telecom or airline datasets without re-calibration.
2. **Twitter Brevity Bias**: Twitter messages are constrained in length, often lacking complete context that email or multi-turn chat support would provide.
3. **Static Snapshot**: Reflects historical support interactions from the Twitter customer support dataset.
