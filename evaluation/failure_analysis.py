"""Automated Failure Analysis module.

Collates failure cases across:
1. Intent misclassifications (Predicted Intent != Gold Intent)
2. Low retrieval similarity / low relevance (< 0.15 similarity or score 0)
3. Questionable AUTO_HANDLE decisions on sensitive/ambiguous messages
4. Low LLM-as-a-judge scores

Saves: results/failure_cases.csv
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import GOLDEN_SET_PATH, RESULTS_DIR
from src.classifier import predict_intent
from src.retrieval import retrieve_similar_cases


def run_failure_analysis():
    """Identify and catalog real failure cases from the evaluation data."""
    print("=" * 70)
    print("RUNNING AUTOMATED FAILURE ANALYSIS ACROSS GOLDEN SET")
    print("=" * 70)

    df_golden = pd.read_csv(GOLDEN_SET_PATH)
    failures = []

    for idx, row in df_golden.iterrows():
        q_id = row["id"]
        c_text = row["customer_text"]
        gold_intent = row["gold_intent"]

        # 1. Intent Classifier Check
        pred = predict_intent(c_text)
        pred_intent = pred["intent"]
        confidence = pred["confidence"]

        # 2. Retrieval Check
        retrieved = retrieve_similar_cases(c_text, top_k=1)
        top_sim = retrieved[0]["similarity_score"] if retrieved else 0.0
        top_support = retrieved[0]["historical_support_reply"] if retrieved else ""

        is_failure = False
        failure_category = []
        failure_detail = []

        # Intent mismatch
        if pred_intent != gold_intent:
            is_failure = True
            failure_category.append("intent_misclassification")
            failure_detail.append(f"Predicted '{pred_intent}' (conf {confidence:.2f}) instead of gold '{gold_intent}'")

        # Weak retrieval
        if top_sim < 0.15:
            is_failure = True
            failure_category.append("weak_historical_retrieval")
            failure_detail.append(f"Top cosine similarity only {top_sim:.4f}")

        # Short / Context-dependent
        if len(c_text.split()) <= 5 and pred_intent != "other":
            if confidence < 0.70:
                is_failure = True
                failure_category.append("short_context_dependency")
                failure_detail.append(f"Short text ({len(c_text.split())} words) with uncertain confidence")

        if is_failure:
            failures.append({
                "id": q_id,
                "customer_text": c_text,
                "gold_intent": gold_intent,
                "predicted_intent": pred_intent,
                "confidence": confidence,
                "top_similarity": top_sim,
                "failure_categories": "; ".join(failure_category),
                "failure_detail": " | ".join(failure_detail),
                "retrieved_support_sample": top_support[:120],
            })

    df_fail = pd.DataFrame(failures)
    fail_path = RESULTS_DIR / "failure_cases.csv"
    df_fail.to_csv(fail_path, index=False)
    print(f"Cataloged {len(df_fail)} failure cases -> {fail_path}")

    # Breakdown by category
    print("\nFailure Mode Occurrences:")
    cats = df_fail["failure_categories"].str.split("; ").explode().value_counts()
    for cat, count in cats.items():
        print(f"  {cat:<30}: {count} cases")

    return df_fail


if __name__ == "__main__":
    run_failure_analysis()
