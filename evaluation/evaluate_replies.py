"""Generate agent support replies for 50 representative evaluation queries.

Outputs results/generated_replies.csv with:
  id, customer_text, intent, reply, decision, reason
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm
from src.config import GOLDEN_SET_PATH, RESULTS_DIR
from src.pipeline import run_support_agent


def generate_evaluation_replies(sample_size: int = 50):
    """Run full agent pipeline across 50 sample customer messages from the golden set."""
    print("=" * 70)
    print(f"GENERATING AGENT REPLIES FOR EVALUATION (Sample size: {sample_size})")
    print("=" * 70)

    df_golden = pd.read_csv(GOLDEN_SET_PATH)
    sample_df = df_golden.sample(n=sample_size, random_state=42).reset_index(drop=True)

    results = []
    print("Processing customer queries through pipeline...")
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        q_id = row["id"]
        customer_text = row["customer_text"]
        
        # Execute unified agent pipeline
        out = run_support_agent(customer_text)

        results.append({
            "id": q_id,
            "customer_text": customer_text,
            "gold_intent": row["gold_intent"],
            "intent": out["intent"],
            "confidence": out["confidence"],
            "reply": out["reply"],
            "decision": out["decision"],
            "reason": out["reason"],
            "top_similarity": out["retrieved_cases"][0]["similarity_score"] if out["retrieved_cases"] else 0.0,
        })

    df_out = pd.DataFrame(results)
    out_path = RESULTS_DIR / "generated_replies.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved generated replies -> {out_path}")

    # Summary statistics
    decisions = df_out["decision"].value_counts()
    print("\nDecision Breakdown:")
    for dec, count in decisions.items():
        print(f"  {dec}: {count} ({count / len(df_out) * 100:.1f}%)")

    return df_out


if __name__ == "__main__":
    generate_evaluation_replies()
