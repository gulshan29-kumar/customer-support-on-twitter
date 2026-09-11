# 📦 AmazonHelp AI Customer Support Agent (Hiver SDE Intern Assignment)

An evaluation-focused, grounded AI customer-support agent built on historical Twitter interactions from **AmazonHelp** (`thoughtvector/customer-support-on-twitter`).

The system classifies customer intents across a 9-class customer lifecycle taxonomy, retrieves historically similar customer-support interactions via n-gram TF-IDF, and drafts policy-compliant support replies using Gemini while enforcing a conservative risk-gated escalation policy (`AUTO_HANDLE` vs `ESCALATE_TO_HUMAN`).

---

## 🚀 Key Highlights & Measured Results

All metrics reported below were computed from scratch on our independent 200-example Golden Evaluation Set (`data/golden_set.csv`) with zero metric fabrication.

| Component | Metric | Baseline 1 (Majority) | Baseline 2 (Simple TF-IDF) | Final Model (Tuned) |
| :--- | :--- | :---: | :---: | :---: |
| **Intent Classifier** | **Accuracy** | 10.00% | 58.50% | **84.00%** |
| **Intent Classifier** | **Macro F1** | 0.0202 | 0.5828 | **0.8376** |
| **Intent Classifier** | **Weighted F1** | 0.0182 | 0.6177 | **0.8467** |
| **Intent Classifier** | **Macro Precision** | 0.0111 | 0.8610 | **0.8830** |
| **Retrieval Engine** | **Top-1 Useful Rate** | — | — | **96.00%** |
| **Retrieval Engine** | **Top-5 Useful Rate** | — | — | **98.00%** |
| **Retrieval Engine** | **Mean Relevance (0–2)** | — | — | **1.0160** |
| **Escalation Policy** | **Escalation Rate** | — | — | **62.00%** |
| **Escalation Policy** | **Auto-Handle Rate** | — | — | **38.00%** |
| **LLM-as-a-Judge** | **Overall Score (1–5)** | — | — | **4.67 / 5.0** |


---

## 🏗️ Architecture

```
[ Incoming Customer Tweet ]
             │
             ▼
[ Preprocessing Module ] ───────> Strips @mentions, URLs, unescapes HTML, normalizes spaces
             │
             ▼
[ Intent Classifier ] ──────────> Tuned TF-IDF (1,2-grams) + Balanced Multinomial LogReg
             │
             ▼
[ TF-IDF Retrieval Index ] ─────> Top-5 Historically Resolved AmazonHelp Cases (Cosine Sim)
             │
             ▼
[ Conservative Rule Guard ] ───> Checks sensitive keywords (fraud, lawyer), confidence floor
             │
             ▼
[ Grounded Gemini Agent ] ──────> Enforces anti-hallucination prompt + structured JSON output
             │
             ▼
[ Final Output Decision ] ──────> AUTO_HANDLE vs ESCALATE_TO_HUMAN + Stated Reason + Draft Reply
```

---

## 📂 Repository Structure

```
hiver/
│
├── README.md                           # Quickstart, reproduction guide, and overview
├── REPORT.md                           # 21-section comprehensive assignment report
├── DECISION_LOG.md                     # 15 non-obvious engineering decisions
├── requirements.txt                    # Pinned Python package dependencies
├── .env.example                        # Template for GEMINI_API_KEY
├── .gitignore                          # Excludes keys, large raw datasets, caches
├── app.py                              # Interactive Gradio demo web application
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Paths, 9-class taxonomy, hyperparameters
│   ├── preprocessing.py                # Regex Twitter cleaner
│   ├── build_data.py                   # Pair reconstruction from Kaggle dataset
│   ├── classifier.py                   # Intent prediction engine (predict_intent)
│   ├── retrieval.py                    # Top-5 historical case retrieval
│   ├── llm_agent.py                    # Grounded reply generator + conservative guard
│   └── pipeline.py                     # Unified agent orchestrator (run_support_agent)
│
├── data/
│   ├── golden_set.csv                  # 200 independently curated & labelled evaluation rows
│   └── amazon_support_pairs.csv        # 29,765 historical AmazonHelp pairs (Zero Leakage)
│
├── models/
│   ├── intent_classifier.joblib        # Trained balanced Logistic Regression model
│   ├── tfidf_vectorizer.joblib         # Fitted n-gram TF-IDF vectorizer
│   ├── retrieval_vectorizer.joblib     # Fitted retrieval vectorizer
│   └── retrieval_corpus.joblib         # Indexed historical matrix and lookup table
│
├── evaluation/
│   ├── evaluate_classifier.py          # Benchmark: Majority vs TF-IDF LogReg vs Tuned
│   ├── evaluate_retrieval.py           # 50 queries evaluated for relevance (0, 1, 2)
│   ├── evaluate_replies.py             # Generates replies for 50-example evaluation sample
│   ├── llm_judge.py                    # Multi-dimensional Gemini judge (6 rubrics, 1-5)
│   ├── human_agreement.py              # Spearman correlation & Cohen's kappa harness
│   └── human_ratings_template.csv      # Human rating template with status indicator
│
├── results/
│   ├── classification_metrics.json     # True computed metrics across baselines
│   ├── classification_report.csv       # Per-class precision, recall, F1
│   ├── confusion_matrix.png            # Visual confusion matrix
│   ├── retrieval_metrics.json          # Top-1/Top-5 utility rates & mean relevance
│   ├── generated_replies.csv           # 50 generated replies with decisions & reasons
│   ├── llm_judge_scores.csv            # 50 LLM judge ratings across 6 rubrics
│   ├── judge_summary_metrics.json      # Summary judge metrics
│   └── failure_cases.csv               # Automatically identified system failures
│
├── docs/
│   ├── golden_set.md                   # Sampling, taxonomy, annotation protocol
│   ├── failure_analysis.md             # Top 5 real failure modes with examples & fixes
│   └── misleading_headline.md          # In-depth breakdown of the 80% headline number
│
└── notebooks/
    └── existing_model.ipynb            # Original exploration notebook preserved
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/gulshan29-kumar/customer-support-on-twitter.git
cd customer-support-on-twitter
```

### 2. Set Up Virtual Environment (Python 3.10+)
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Gemini API key:
```bash
cp .env.example .env
```
In `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(Note: If `GEMINI_API_KEY` is omitted, the system seamlessly uses a deterministic, rule-based grounded fallback that executes offline without crashing).*

---

## ⚡ Quickstart: Reproduce Headline Evaluation in < 2 Minutes

All evaluation scripts read directly from the provided `data/` and pre-trained `models/` artifacts.

### 1. Run Intent Classifier Evaluation (Golden Set N=200)
```bash
python evaluation/evaluate_classifier.py
```
*Outputs comparative metrics for Majority Baseline, Simple TF-IDF, and Tuned Model to console, saves `results/classification_metrics.json` and `results/confusion_matrix.png`.*

### 2. Run Retrieval Evaluation (N=50)
```bash
python evaluation/evaluate_retrieval.py
```
*Evaluates top-5 historical retrieval using the 0/1/2 relevance rubric. Saves `results/retrieval_metrics.json`.*

### 3. Run Reply Generation & LLM Judge
```bash
python evaluation/evaluate_replies.py
python evaluation/llm_judge.py
```
*Generates replies for 50 customer queries and evaluates correctness, relevance, groundedness, helpfulness, tone, and unsupported claims.*

### 4. Run Human Agreement Harness
```bash
python evaluation/human_agreement.py
```
*Checks `evaluation/human_ratings_template.csv`. Reports `"Human ratings pending"` if un-annotated, preventing fake agreement fabrication.*

### 5. Run Automated Failure Analysis
```bash
python evaluation/failure_analysis.py
```
*Catalogs misclassifications and retrieval anomalies into `results/failure_cases.csv`.*

---

## 💻 Running the Interactive Demo

Launch the local Gradio interface:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

### Interactive UI Features:
- **Input Textbox**: Type any customer tweet or select from sample pre-loaded examples.
- **Predicted Intent & Confidence**: Displays the 9-class intent and model certainty.
- **Escalation Decision**: Clearly displays `AUTO_HANDLE` or `ESCALATE_TO_HUMAN` with a stated reason.
- **Draft Support Reply**: Clean, grounded support message.
- **Historical Grounding Evidence**: Expandable view of the top-5 retrieved historical AmazonHelp interactions with similarity scores.

---

## 🔍 Example Inputs & System Outputs

### Example 1: Routine Delivery Inquiry (AUTO_HANDLE)
- **Input**: `"Where is my package? It was supposed to be delivered yesterday and tracking has not updated."`
- **Predicted Intent**: `delivery_or_delay` (Confidence: 0.9999)
- **Decision**: `AUTO_HANDLE`
- **Reason**: `Clear delivery inquiry with strong historical guidance on order tracking and carrier follow-up.`
- **Draft Reply**: `"We are sorry to hear your delivery is delayed! Please check your order tracking in 'Your Orders' for real-time carrier updates, or message us directly with your order details so we can investigate."`

### Example 2: Account Security / Hacked Account (ESCALATE_TO_HUMAN)
- **Input**: `"Someone hacked into my Amazon account, changed my email, and charged $400 to my card!"`
- **Predicted Intent**: `account_or_access` (Confidence: 1.0000)
- **Decision**: `ESCALATE_TO_HUMAN`
- **Reason**: `Triggered sensitive keyword guard 'hacked' requiring human oversight.`
- **Draft Reply**: `"We apologize for the inconvenience. To securely investigate your account details and resolve this, our specialist team will review your case directly."`

### Example 3: Ambiguous or Irrelevant Message (ESCALATE_TO_HUMAN)
- **Input**: `"Hey what happened"`
- **Predicted Intent**: `other` (Confidence: 0.3541)
- **Decision**: `ESCALATE_TO_HUMAN`
- **Reason**: `Intent confidence (0.35) below threshold (0.40).`

---

## 🛠️ How to Retrain from Scratch

If you have the Kaggle `twcs.csv.zip` file, you can regenerate datasets and retrain all models from scratch:

```bash
# 1. Reconstruct pairs and build golden set
python -m src.build_data

# 2. Train intent classifiers
python -m src.classifier

# 3. Rebuild retrieval index
python -m src.retrieval
```

---

## ⚠️ Known Limitations
1. **Public Social Media Redaction**: Historical tweets on Twitter frequently direct users to direct messages (`https://t.co/...`) to authenticate. Final account resolutions are often masked from public view.
2. **Compound Multi-Intent Queries**: Single-label classification forces single intents on compound complaints (e.g., late delivery + refund request).
3. **English Only**: The system is calibrated for English customer support interactions.

---

## 📄 Documentation Links
- Detailed Technical Report: [`REPORT.md`](REPORT.md)
- Engineering Decisions: [`DECISION_LOG.md`](DECISION_LOG.md)
- Golden Evaluation Set Methodology: [`docs/golden_set.md`](docs/golden_set.md)
- Top 5 Real Failure Modes: [`docs/failure_analysis.md`](docs/failure_analysis.md)

