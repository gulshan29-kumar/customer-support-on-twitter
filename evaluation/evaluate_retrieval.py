"""Evaluation harness for Historical Case Retrieval.

Evaluates top-5 retrieved historical AmazonHelp cases for 50 representative queries.
Scores relevance using a 0/1/2 rubric:
  0 = Irrelevant (different product/unrelated context)
  1 = Somewhat useful (same domain, generic guidance)
  2 = Clearly useful (direct match, specific actionable resolution)

Calculates:
  - Top-1 useful rate (% where rank-1 case score >= 1)
  - Top-5 useful rate (% where at least one case in top-5 has score >= 1)
  - Mean relevance (average score across all retrieved cases, range 0.0 - 2.0)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np
from src.config import GOLDEN_SET_PATH, RESULTS_DIR
from src.retrieval import retrieve_similar_cases
from src.preprocessing import clean_text


def score_case_relevance(query: str, retrieved_customer: str, retrieved_support: str, sim_score: float) -> int:
    """Deterministic relevance scorer based on semantic and keyword overlap with support actionability."""
    q_words = set(clean_text(query).lower().split())
    c_words = set(clean_text(retrieved_customer).lower().split())
    s_words = set(clean_text(retrieved_support).lower().split())

    overlap = len(q_words & c_words)
    overlap_ratio = overlap / max(len(q_words), 1)

    # Clearly useful: strong token overlap AND support response provides direct instruction/link
    if sim_score >= 0.25 and overlap_ratio >= 0.35 and len(s_words) >= 6:
        return 2
    # Somewhat useful: moderate overlap or relevant domain guidance
    elif sim_score >= 0.12 and overlap >= 2:
        return 1
    # Irrelevant
    else:
        return 0


def run_retrieval_evaluation(sample_size: int = 50):
    """Run retrieval evaluation on 50 sample queries from the golden set."""
    print("=" * 70)
    print(f"EVALUATING HISTORICAL CASE RETRIEVAL (Sample: {sample_size} queries)")
    print("=" * 70)

    df_golden = pd.read_csv(GOLDEN_SET_PATH)
    # Sample 50 queries with fixed seed
    sample_df = df_golden.sample(n=sample_size, random_state=42).reset_index(drop=True)

    top_1_useful_count = 0
    top_5_useful_count = 0
    all_scores = []
    case_records = []

    for idx, row in sample_df.iterrows():
        query = row["customer_text"]
        retrieved = retrieve_similar_cases(query, top_k=5)
        
        scores_for_query = []
        for rank, item in enumerate(retrieved, 1):
            score = score_case_relevance(
                query=query,
                retrieved_customer=item["customer_message"],
                retrieved_support=item["historical_support_reply"],
                sim_score=item["similarity_score"],
            )
            scores_for_query.append(score)
            all_scores.append(score)

            case_records.append({
                "query_id": row["id"],
                "query_text": query,
                "gold_intent": row["gold_intent"],
                "rank": rank,
                "similarity_score": item["similarity_score"],
                "relevance_score": score,
                "retrieved_customer": item["customer_message"],
                "retrieved_support": item["historical_support_reply"],
            })

        # Check Top-1 utility
        if scores_for_query and scores_for_query[0] >= 1:
            top_1_useful_count += 1

        # Check Top-5 utility (at least one useful case in top 5)
        if any(s >= 1 for s in scores_for_query):
            top_5_useful_count += 1

    top_1_useful_rate = round(top_1_useful_count / sample_size, 4)
    top_5_useful_rate = round(top_5_useful_count / sample_size, 4)
    mean_relevance = round(float(np.mean(all_scores)), 4) if all_scores else 0.0

    retrieval_metrics = {
        "evaluation_queries": sample_size,
        "total_cases_evaluated": len(all_scores),
        "top_1_useful_rate": top_1_useful_rate,
        "top_5_useful_rate": top_5_useful_rate,
        "mean_relevance_score_0_to_2": mean_relevance,
        "score_distribution": {
            "clearly_useful_2": int(sum(1 for s in all_scores if s == 2)),
            "somewhat_useful_1": int(sum(1 for s in all_scores if s == 1)),
            "irrelevant_0": int(sum(1 for s in all_scores if s == 0)),
        },
        "rubric_definition": {
            "0": "Irrelevant context or dissimilar issue",
            "1": "Somewhat useful / same problem space with general instructions",
            "2": "Clearly useful / actionable solution matching customer query directly",
        },
        "limitations": [
            "Lexical TF-IDF n-grams can miss paraphrases without shared vocabulary.",
            "Short queries have lower overlap scores due to sparse n-grams.",
            "Historical responses often link to private DMs/forms, limiting full ground-truth text visibility.",
        ]
    }

    # Save metrics JSON
    metrics_path = RESULTS_DIR / "retrieval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_metrics, f, indent=2)
    print(f"Saved retrieval metrics -> {metrics_path}")

    # Save detailed per-case scores
    cases_path = RESULTS_DIR / "retrieval_case_evaluations.csv"
    pd.DataFrame(case_records).to_csv(cases_path, index=False)
    print(f"Saved detailed case evaluations -> {cases_path}")

    print("\n" + "=" * 50)
    print(f"Top-1 Useful Rate:           {top_1_useful_rate * 100:.2f}%")
    print(f"Top-5 Useful Rate:           {top_5_useful_rate * 100:.2f}%")
    print(f"Mean Relevance Score (0-2):  {mean_relevance:.4f}")
    print("=" * 50)

    return retrieval_metrics


if __name__ == "__main__":
    run_retrieval_evaluation()
