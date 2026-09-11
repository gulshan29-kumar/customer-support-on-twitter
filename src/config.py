"""Configuration constants, taxonomy definitions, and file paths."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DOCS_DIR = BASE_DIR / "docs"

# Ensure runtime directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Target Brand
TARGET_BRAND = "AmazonHelp"

# 9-class Intent Taxonomy
INTENT_CLASSES = [
    "technical_issue",
    "delivery_or_delay",
    "booking_or_travel",
    "payment_or_billing",
    "refund_or_compensation",
    "account_or_access",
    "product_or_service_info",
    "complaint_or_feedback",
    "other",
]

INTENT_DESCRIPTIONS = {
    "technical_issue": "App, website, device, software, network, loading, crash, error, connection problems.",
    "delivery_or_delay": "Late delivery, missing order, shipment, tracking, delivery delay, courier questions.",
    "booking_or_travel": "Booking, ticket, reservation, flight, train, check-in, boarding, seat, travel.",
    "payment_or_billing": "Payment, charges, billing, transaction, card, fee, pricing, unauthorized charge.",
    "refund_or_compensation": "Refunds, reimbursement, compensation, money back, credit request.",
    "account_or_access": "Login, account access, password, hacked account, OTP, verification, security, suspended.",
    "product_or_service_info": "General product/service questions, availability, offers, policies, discounts, information.",
    "complaint_or_feedback": "Explicit dissatisfaction, complaints, praise, feedback, suggestions, poor service.",
    "other": "Messages that are ambiguous, social, extremely short, irrelevant, or cannot be reliably classified.",
}

# Data File Paths
GOLDEN_SET_PATH = DATA_DIR / "golden_set.csv"
HISTORICAL_PAIRS_PATH = DATA_DIR / "amazon_support_pairs.csv"

# Model Artifact Paths
CLASSIFIER_PATH = MODELS_DIR / "intent_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
RETRIEVAL_VECTORIZER_PATH = MODELS_DIR / "retrieval_vectorizer.joblib"
RETRIEVAL_CORPUS_PATH = MODELS_DIR / "retrieval_corpus.joblib"

# Escalation Decisions
DECISION_AUTO_HANDLE = "AUTO_HANDLE"
DECISION_ESCALATE = "ESCALATE_TO_HUMAN"

# Escalation policy risk triggers
ESCALATION_KEYWORDS = [
    "lawyer", "attorney", "sue", "legal", "police", "fraud", "scam", 
    "stolen", "hacked", "identity theft", "unauthorized", "chargeback"
]
MIN_CONFIDENCE_THRESHOLD = 0.40
MIN_RETRIEVAL_SIMILARITY = 0.15

# Gemini API settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
