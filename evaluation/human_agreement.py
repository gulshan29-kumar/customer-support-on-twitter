"""Evaluate Human-LLM agreement on response quality judgments.

Compares human evaluation ratings against LLM-as-a-judge scores.
Calculates Spearman rank correlation and quadratic-weighted Cohen's kappa.
Strict Rule: If human ratings are not yet filled, reports 'Human ratings pending'
without fabricating synthetic agreement.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from src.config import RESULTS_DIR

HUMAN_TEMPLATE_PATH = Path(__file__).resolve().parent / "human_ratings_template.csv"
LLM_SCORES_PATH = RESULTS_DIR / "llm_judge_scores.csv"
AGREEMENT_OUTPUT_PATH = RESULTS_DIR / "human_llm_agreement.json"

DIMENSIONS = ["correctness", "relevance", "groundedness", "helpfulness", "tone", "unsupported_claims"]


def evaluate_agreement():
    """Compute human-vs-LLM agreement metrics if human ratings exist."""
    print("=" * 70)
    print("EVALUATING HUMAN vs. LLM-AS-A-JUDGE AGREEMENT")
    print("=" * 70)

    if not HUMAN_TEMPLATE_PATH.exists():
        print(f"Error: Template not found at {HUMAN_TEMPLATE_PATH}")
        return

    if not LLM_SCORES_PATH.exists():
        print(f"Error: LLM judge scores not found at {LLM_SCORES_PATH}. Run llm_judge.py first.")
        return

    df_human = pd.read_csv(HUMAN_TEMPLATE_PATH)
    df_llm = pd.read_csv(LLM_SCORES_PATH)

    # Check whether ratings columns are populated
    has_ratings = False
    for dim in DIMENSIONS:
        if dim in df_human.columns:
            valid_numeric = pd.to_numeric(df_human[dim], errors="coerce").dropna()
            if len(valid_numeric) >= 10:  # Require at least 10 human annotations to calculate
                has_ratings = True
                break

    if not has_ratings:
        print("\n[STATUS]: Human ratings pending")
        print("-" * 70)
        print("Human ratings have not yet been completed in:")
        print(f"  {HUMAN_TEMPLATE_PATH}")
        print("To compute real agreement:")
        print("  1. Open evaluation/human_ratings_template.csv")
        print("  2. Enter integer scores (1-5) for correctness, relevance, groundedness,")
        print("     helpfulness, tone, and unsupported_claims.")
        print("  3. Re-run: python evaluation/human_agreement.py")
        print("\n*Note: Synthetic agreement metrics are strictly disabled to prevent metric fabrication.*")
        print("=" * 70)

        # Write pending status JSON
        status_payload = {
            "status": "Human ratings pending",
            "message": "Template provided at evaluation/human_ratings_template.csv. Awaiting manual human annotation.",
            "metrics": None,
        }
        with open(AGREEMENT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(status_payload, f, indent=2)
        return status_payload

    # Merge on query id
    merged = pd.merge(df_human, df_llm, on="id", suffixes=("_human", "_llm"))

    agreement_results = {
        "status": "Completed",
        "sample_size": len(merged),
        "dimensions": {},
    }

    print("\nCalculated Agreement Metrics:")
    print(f"{'Dimension':<20} | {'Spearman Rho':<14} | {'Cohen Kappa (Weighted)':<22}")
    print("-" * 62)

    for dim in DIMENSIONS:
        h_col = f"{dim}_human"
        l_col = f"{dim}_llm"

        if h_col in merged.columns and l_col in merged.columns:
            sub = merged[[h_col, l_col]].dropna()
            sub[h_col] = pd.to_numeric(sub[h_col], errors="coerce")
            sub[l_col] = pd.to_numeric(sub[l_col], errors="coerce")
            sub = sub.dropna()

            if len(sub) >= 5:
                rho, p_val = spearmanr(sub[h_col], sub[l_col])
                # Quadratic weighted kappa for ordinal 1-5 scale
                kappa = cohen_kappa_score(
                    sub[h_col].astype(int),
                    sub[l_col].astype(int),
                    weights="quadratic",
                )
                agreement_results["dimensions"][dim] = {
                    "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else 0.0,
                    "spearman_p_value": round(float(p_val), 4) if not np.isnan(p_val) else 1.0,
                    "weighted_cohen_kappa": round(float(kappa), 4) if not np.isnan(kappa) else 0.0,
                    "annotated_count": len(sub),
                }
                print(f"{dim.capitalize():<20} | {rho:<14.4f} | {kappa:<22.4f}")

    with open(AGREEMENT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(agreement_results, f, indent=2)
    print(f"\nSaved agreement results -> {AGREEMENT_OUTPUT_PATH}")

    return agreement_results


if __name__ == "__main__":
    evaluate_agreement()
