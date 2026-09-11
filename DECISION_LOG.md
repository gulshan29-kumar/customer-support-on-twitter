# Decision Log: Engineering & Design Trade-offs

This document logs 15 non-obvious design, architectural, and evaluation decisions made while engineering the Hiver AI Customer Support Agent.

---

### Decision 1: Why Focus Exclusively on AmazonHelp
- **Context**: The raw Kaggle dataset contains over 100 brands across airlines, telecom, tech hardware, and streaming services.
- **Decision**: Restrict the operational domain strictly to `AmazonHelp`.
- **Rationale**: Multi-brand support creates conflicting support policies (e.g., flight cancellation rules vs. e-commerce package return windows). Focusing on one high-volume brand ensures that historical retrieval contains coherent, domain-consistent grounding.

---

### Decision 2: Why Reject Multi-Brand "One-Model-Fits-All" Architectures
- **Context**: The original notebook attempted to predict across 5 disparate brands.
- **Decision**: Discard cross-brand prediction in favor of deep, single-brand resolution.
- **Rationale**: In production, a support desk like Hiver serves enterprise clients who require brand-specific domain adaptation. Multi-brand classification in customer support is an artificial problem; real support agents operate within a single company's policy bounds.

---

### Decision 3: Why a 9-Class Intent Taxonomy Was Chosen
- **Context**: Support requests range from granular sub-intents (e.g., *"where is my OTP"* vs *"password reset"*) to coarse binary triage (e.g., *"urgent"* vs *"routine"*).
- **Decision**: Adopt a structured 9-class taxonomy covering the complete retail customer lifecycle.
- **Rationale**: 9 classes strikes the optimal balance between explainability in an interview, statistical support in a 200-example golden set, and actionable downstream routing (e.g., routing billing disputes to finance and app crashes to engineering).

---

### Decision 4: Why Retain an Explicit 'Other' Class
- **Context**: Classifiers often force every input into a functional category, leading to high-confidence errors on noisy input.
- **Decision**: Keep an explicit `other` class for ambiguous, purely social, or uninterpretable messages.
- **Rationale**: Twitter traffic is notoriously noisy. Forcing ambiguous queries like *"hey what happened"* or *"thanks for nothing"* into functional classes causes dangerous automation errors. An explicit `other` category provides a safe catchment basin for conservative escalation.

---

### Decision 5: Why Customer-Support Pairs Were Formally Reconstructed
- **Context**: The raw dataset contains isolated tweets with parent-pointer tweet IDs (`in_response_to_tweet_id`).
- **Decision**: Reconstruct true conversational dyads `(customer_inbound, support_reply)` using pointer chains.
- **Rationale**: Intent classification requires customer text, while grounded generation requires seeing how official brand agents actually resolved that specific problem. Grounded reply generation is impossible without linked pairs.

---

### Decision 6: Why Mentions and URLs Were Cleaned Prior to Modeling
- **Context**: Raw tweets contain `@AmazonHelp`, customer handles (`@115712`), and short links (`https://t.co/...`).
- **Decision**: Strip all `@mentions` and URLs during preprocessing.
- **Rationale**:
  1. Mentions leak brand identity into classifiers and confuse tokenizers with arbitrary user IDs.
  2. Short URLs are dead links that risk hallucination if an LLM regurgitates them to a user.
  3. Cleaning ensures that models learn semantic representations rather than memorizing Twitter handles.

---

### Decision 7: Why TF-IDF with Word N-Grams (1,2) Was Chosen for Retrieval
- **Context**: Modern retrieval often uses dense neural vector embeddings (e.g., bi-encoders).
- **Decision**: Use sublinear TF-IDF with word 1-gram and 2-gram representations and cosine similarity.
- **Rationale**:
  1. For short domain-specific Twitter queries, lexical keywords (*"Prime video"*, *"double charged"*, *"out for delivery"*) are strong, precise signals.
  2. TF-IDF requires zero external API latency, has minimal compute footprint, and runs in milliseconds without neural inference bottlenecks.
  3. It is transparent and easily explainable in a live technical interview.

---

### Decision 8: Why Top-5 Historical Cases Were Retrieved
- **Context**: Prompting an LLM with 1 case risks brittle grounding; prompting with 20 cases risks context dilution and latency.
- **Decision**: Provide exactly the top-5 retrieved historical interactions.
- **Rationale**: 5 cases provide enough diversity to detect consensus support behavior (e.g., directing users to 'Your Orders' tracking) while fitting cleanly within the prompt token budget.

---

### Decision 9: Why Golden Set Examples Are Strictly Excluded from Retrieval
- **Context**: A common evaluation trap is indexing the entire dataset, allowing the retrieval engine to retrieve the ground-truth historical answer for evaluation queries.
- **Decision**: Completely remove all 200 golden set tweet IDs and duplicate texts from `amazon_support_pairs.csv`.
- **Rationale**: Zero-leakage retrieval benchmarking. If the model retrieves its own evaluation target, retrieval metrics and groundedness scores are completely invalid.

---

### Decision 10: Why Cosine Similarity Was NOT Used as the Sole Escalation Rule
- **Context**: A simplistic approach escalates whenever `similarity < 0.20`.
- **Decision**: Combine multi-tiered guards: similarity thresholds, intent confidence, sensitive keyword triggers, and security intent checks.
- **Rationale**: A query like *"Someone stole my credit card and made $500 of charges"* might have a high 0.45 similarity to a historical unauthorized charge tweet, but automated handling would be disastrous. High similarity does not imply safe auto-handling.

---

### Decision 11: Why Gemini with Strict JSON Schema Was Used for Generation
- **Context**: Unconstrained generative models frequently produce verbose conversational disclaimers (*"As an AI..."*) or hallucinate refund guarantees.
- **Decision**: Use Gemini (`gemini-2.5-flash`) with structured JSON schema output (`reply`, `decision`, `reason`).
- **Rationale**: Structured JSON guarantees programmatic parseability in backend services and eliminates conversational filler. Temperature 0.1 maximizes deterministic adherence to retrieved evidence.

---

### Decision 12: Why Conservative Escalation Is Preferred Over Aggressive Automation
- **Context**: Support organizations often focus on maximizing the "deflection rate" (percentage of auto-handled tickets).
- **Decision**: Adopt an intentionally conservative escalation policy (62% escalation rate on evaluation data).
- **Rationale**: In customer support, an unnecessary escalation costs $2 in agent time, while a single hallucinated financial promise or mishandled security breach costs thousands in customer churn, compliance penalties, and brand erosion.

---

### Decision 13: Why a Majority-Class Baseline Is Mandatory
- **Context**: Many machine learning reports only present the final model's numbers.
- **Decision**: Implement and evaluate a `DummyClassifier(strategy='most_frequent')` on the golden set.
- **Rationale**: The majority baseline provides the true empirical floor (10.0% accuracy on our 9-class golden set). Comparing against it proves that the model's 84.0% accuracy represents genuine learning rather than class skew exploitation.

---

### Decision 14: Why a Simple TF-IDF Logistic Regression Baseline Was Kept Separate
- **Context**: The final model uses tuned n-grams, sublinear scaling, and class-weight balancing.
- **Decision**: Include a vanilla unigram TF-IDF + standard Logistic Regression baseline (Baseline 2: 58.5% accuracy).
- **Rationale**: Separating Baseline 2 proves exactly where performance gains originated (ngram collocation, sublinear tf scaling, and class balancing increased accuracy from 58.5% to 84.0% and macro F1 from 0.5828 to 0.8376).

---

### Decision 15: Why LLM-as-a-Judge Requires Human Agreement Evidence
- **Context**: Generative evaluations often rely entirely on LLM-judge scores without human verification.
- **Decision**: Establish an explicit human evaluation harness (`human_agreement.py`) with Spearman correlation and Cohen's kappa, reporting "Human ratings pending" when un-annotated.
- **Rationale**: LLM judges suffer from self-preference bias. Claiming that an agent is effective based solely on an LLM's opinion is circular reasoning. Real evaluation integrity requires measuring whether human support leads agree with the LLM's judgments.
