"""Intent classification module for Amazon customer support tweets.

Provides training, evaluation, and inference for the 9-class intent taxonomy.
Compares Majority-Class baseline, Simple TF-IDF + Logistic Regression, and Tuned Classifier.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.pipeline import Pipeline

from src.config import (
    HISTORICAL_PAIRS_PATH,
    GOLDEN_SET_PATH,
    CLASSIFIER_PATH,
    VECTORIZER_PATH,
    INTENT_CLASSES,
)
from src.preprocessing import clean_text

# Global cached inference artifacts
_MODEL = None
_VECTORIZER = None


def load_inference_artifacts():
    """Load the trained model and vectorizer into global memory."""
    global _MODEL, _VECTORIZER
    if _MODEL is None and CLASSIFIER_PATH.exists() and VECTORIZER_PATH.exists():
        _MODEL = joblib.load(CLASSIFIER_PATH)
        _VECTORIZER = joblib.load(VECTORIZER_PATH)


def train_models():
    """Train baseline and tuned intent classifiers on the historical corpus.
    
    Returns:
        Tuple of (majority_model, simple_logreg_model, tuned_logreg_model, vectorizer)
    """
    print(f"Loading training data from {HISTORICAL_PAIRS_PATH}...")
    df_train = pd.read_csv(HISTORICAL_PAIRS_PATH)
    
    # Filter valid rows
    df_train = df_train[df_train["customer_text"].notna() & df_train["intent"].notna()]
    X_train = df_train["customer_text"].astype(str)
    y_train = df_train["intent"].astype(str)

    print(f"Training set size: {len(X_train)} rows across {len(y_train.unique())} classes.")

    # 1. Baseline 1: Majority Class Classifier
    print("Training Baseline 1: Majority Class...")
    majority_model = DummyClassifier(strategy="most_frequent")
    majority_model.fit(X_train, y_train)

    # 2. Baseline 2: Simple TF-IDF + Logistic Regression (unigram)
    print("Training Baseline 2: Simple TF-IDF (1-gram) + Logistic Regression...")
    simple_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=5000)
    X_train_simple = simple_vectorizer.fit_transform(X_train)
    simple_logreg = LogisticRegression(max_iter=500, random_state=42)
    simple_logreg.fit(X_train_simple, y_train)

    # 3. Tuned Model: Tuned TF-IDF (1,2-grams, sublinear tf) + Balanced Logistic Regression
    print("Training Final Model: Tuned TF-IDF (1,2-grams) + Multinomial Logistic Regression...")
    tuned_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=15000,
    )
    X_train_tuned = tuned_vectorizer.fit_transform(X_train)
    tuned_logreg = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    tuned_logreg.fit(X_train_tuned, y_train)

    # Save final chosen model and vectorizer
    joblib.dump(tuned_logreg, CLASSIFIER_PATH)
    joblib.dump(tuned_vectorizer, VECTORIZER_PATH)
    print(f"Saved final classifier -> {CLASSIFIER_PATH}")
    print(f"Saved final vectorizer -> {VECTORIZER_PATH}")

    return majority_model, (simple_vectorizer, simple_logreg), (tuned_vectorizer, tuned_logreg)


def predict_intent(text: str) -> Dict[str, Any]:
    """Predict customer intent and confidence for a given input text.
    
    Args:
        text: Raw or partially cleaned customer message.
        
    Returns:
        dict: {"intent": str, "confidence": float}
    """
    load_inference_artifacts()
    cleaned = clean_text(text)
    if not cleaned:
        return {"intent": "other", "confidence": 0.0}

    if _MODEL is None or _VECTORIZER is None:
        # Fallback if model not yet trained
        return {"intent": "other", "confidence": 0.0}

    X_vec = _VECTORIZER.transform([cleaned])
    probs = _MODEL.predict_proba(X_vec)[0]
    max_idx = np.argmax(probs)
    predicted_intent = _MODEL.classes_[max_idx]
    confidence = float(probs[max_idx])

    return {
        "intent": predicted_intent,
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":
    train_models()
