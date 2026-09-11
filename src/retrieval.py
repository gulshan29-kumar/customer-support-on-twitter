"""Historical case retrieval module for AmazonHelp support interactions.

Indexes historical (customer_tweet, support_tweet) pairs using TF-IDF and n-grams.
Finds top-k most similar historical cases via cosine similarity to ground LLM replies.
"""

from pathlib import Path
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    HISTORICAL_PAIRS_PATH,
    RETRIEVAL_VECTORIZER_PATH,
    RETRIEVAL_CORPUS_PATH,
    TARGET_BRAND,
)
from src.preprocessing import clean_text

# Cached in-memory objects
_RETRIEVAL_VECTORIZER = None
_CORPUS_MATRIX = None
_CORPUS_DF = None


def build_retrieval_index():
    """Build and persist the TF-IDF retrieval index from historical AmazonHelp pairs."""
    print(f"Building retrieval index from {HISTORICAL_PAIRS_PATH}...")
    df = pd.read_csv(HISTORICAL_PAIRS_PATH)
    df = df[df["customer_text"].notna() & df["support_text"].notna()].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=25000,
        min_df=2,
    )
    corpus_matrix = vectorizer.fit_transform(df["customer_text"])

    # Persist artifacts
    joblib.dump(vectorizer, RETRIEVAL_VECTORIZER_PATH)
    joblib.dump({"matrix": corpus_matrix, "df": df[["customer_tweet_id", "customer_text", "support_tweet_id", "support_text", "intent"]]}, RETRIEVAL_CORPUS_PATH)
    print(f"Saved retrieval vectorizer -> {RETRIEVAL_VECTORIZER_PATH}")
    print(f"Saved retrieval corpus matrix ({corpus_matrix.shape}) -> {RETRIEVAL_CORPUS_PATH}")


def load_retrieval_index():
    """Load the retrieval index and corpus into global memory."""
    global _RETRIEVAL_VECTORIZER, _CORPUS_MATRIX, _CORPUS_DF
    if _RETRIEVAL_VECTORIZER is None or _CORPUS_MATRIX is None or _CORPUS_DF is None:
        if not (RETRIEVAL_VECTORIZER_PATH.exists() and RETRIEVAL_CORPUS_PATH.exists()):
            build_retrieval_index()
        _RETRIEVAL_VECTORIZER = joblib.load(RETRIEVAL_VECTORIZER_PATH)
        bundle = joblib.load(RETRIEVAL_CORPUS_PATH)
        _CORPUS_MATRIX = bundle["matrix"]
        _CORPUS_DF = bundle["df"]


def retrieve_similar_cases(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve the top-k historically similar AmazonHelp customer support cases.
    
    Args:
        query: Customer query string.
        top_k: Number of historical cases to return.
        
    Returns:
        List of dicts with customer message, historical support reply, and similarity score.
    """
    load_retrieval_index()
    cleaned_query = clean_text(query)
    if not cleaned_query:
        return []

    q_vec = _RETRIEVAL_VECTORIZER.transform([cleaned_query])
    similarities = cosine_similarity(q_vec, _CORPUS_MATRIX)[0]

    # Get top-k indices with highest similarity
    if len(similarities) == 0:
        return []

    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        sim_score = float(similarities[idx])
        row = _CORPUS_DF.iloc[idx]
        results.append({
            "customer_message": row["customer_text"],
            "historical_support_reply": row["support_text"],
            "intent": row.get("intent", "unknown"),
            "similarity_score": round(sim_score, 4),
            "customer_tweet_id": str(row["customer_tweet_id"]),
            "support_tweet_id": str(row["support_tweet_id"]),
        })

    return results


if __name__ == "__main__":
    build_retrieval_index()
    sample = "My package has not arrived and tracking is not updating."
    print(f"\nTesting retrieval for: '{sample}'")
    hits = retrieve_similar_cases(sample, top_k=3)
    for i, h in enumerate(hits, 1):
        print(f"\n[Case {i}] Sim: {h['similarity_score']} | Intent: {h['intent']}")
        print(f"Customer: {h['customer_message']}")
        print(f"Support:  {h['historical_support_reply']}")
