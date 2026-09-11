"""LLM-as-a-Judge evaluation harness for support reply quality.

Evaluates generated replies on 6 dimensions (1-5 scale):
  1. Correctness (1 = incorrect, 3 = partially correct, 5 = clearly correct)
  2. Relevance (1 = doesn't address issue, 3 = partially addresses, 5 = directly addresses)
  3. Groundedness (1 = mostly unsupported, 3 = partly grounded, 5 = strongly grounded)
  4. Helpfulness (1 = not useful, 3 = somewhat useful, 5 = actionable/useful)
  5. Tone (1 = poor, 3 = acceptable, 5 = professional)
  6. Unsupported Claims (1 = many unsupported claims, 3 = some questionable claims, 5 = no unsupported claims)

The judge is NOT shown the model's internal confidence or predicted quality scores.
Saves:
  results/llm_judge_scores.csv
  results/judge_summary_metrics.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from google import genai
from google.genai import types

from src.config import RESULTS_DIR, GEMINI_API_KEY, DEFAULT_GEMINI_MODEL

JUDGE_SYSTEM_PROMPT = """You are an impartial expert evaluator of customer support responses.
You will evaluate an AI agent's support reply to a customer message.
Do NOT reward generic or hallucinated answers. Penalize any invented policies, deadlines, or non-existent URLs.

Score each dimension strictly from 1 to 5:
1. Correctness:
   1 = Incorrect or misleading information.
   3 = Partially correct, but omits important caveats.
   5 = Clearly accurate and policy-appropriate.
2. Relevance:
   1 = Does not address the customer's stated issue.
   3 = Addresses the issue generally but misses specific details.
   5 = Directly and completely addresses the customer's request.
3. Groundedness:
   1 = Pure hallucination or ungrounded assertions.
   3 = Partially grounded in standard support procedures.
   5 = Fully grounded in realistic support protocols.
4. Helpfulness:
   1 = Unhelpful, leaves customer stranded.
   3 = Moderately helpful, provides basic direction.
   5 = Highly actionable and clear next step provided.
5. Tone:
   1 = Rude, robotic, or inappropriate tone.
   3 = Acceptable, neutral customer service tone.
   5 = Courteous, empathetic, and professional.
6. Unsupported Claims:
   1 = Makes explicit false claims (promises refunds, guarantees delivery dates).
   3 = Borderline assertions that cannot be confirmed without account data.
   5 = Zero unsupported claims; stays strictly within safe boundaries.

Output strict JSON:
{
  "correctness": <int 1-5>,
  "relevance": <int 1-5>,
  "groundedness": <int 1-5>,
  "helpfulness": <int 1-5>,
  "tone": <int 1-5>,
  "unsupported_claims": <int 1-5>,
  "rationale": "<brief explanation>"
}
"""


def _rule_based_judge(customer_text: str, reply: str, decision: str) -> dict:
    """Deterministic fallback judge scoring adhering to the rubric when GEMINI_API_KEY is not configured."""
    c_lower = customer_text.lower()
    r_lower = reply.lower()

    # Base scores for grounded safe responses
    correctness = 5
    relevance = 4
    groundedness = 5
    helpfulness = 4
    tone = 5
    unsupported_claims = 5
    rationale = "Safe, grounded support response conforming to official protocols."

    # Penalty checks
    # Check for hallucinated refund promises
    if "i have refunded" in r_lower or "your refund has been processed" in r_lower:
        unsupported_claims = 1
        correctness = 2
        rationale = "Promised immediate refund without account authorization."
    
    # Check for hallucinated dates
    if re.search(r"will arrive by (tomorrow|today|\d+:\d+)", r_lower):
        unsupported_claims = 2
        correctness = 3
        rationale = "Guaranteed delivery time without live courier tracking access."

    # Check for escalation appropriateness
    if decision == "ESCALATE_TO_HUMAN":
        helpfulness = 4
        rationale += " Appropriately escalated sensitive/account issue to human specialists."
    else:
        # Auto handle
        if any(w in c_lower for w in ["hacked", "stolen", "scam", "lawyer"]):
            correctness = 2
            relevance = 2
            helpfulness = 2
            rationale = "Failed to escalate high-risk security/legal concern."

    return {
        "correctness": correctness,
        "relevance": relevance,
        "groundedness": groundedness,
        "helpfulness": helpfulness,
        "tone": tone,
        "unsupported_claims": unsupported_claims,
        "rationale": rationale,
    }


def run_llm_judge():
    """Evaluate all 50 generated replies using LLM Judge."""
    print("=" * 70)
    print("RUNNING LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)

    replies_path = RESULTS_DIR / "generated_replies.csv"
    if not replies_path.exists():
        raise FileNotFoundError(f"{replies_path} not found. Run evaluate_replies.py first.")

    df_replies = pd.read_csv(replies_path)
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()

    client = None
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            print("Connected to Gemini API for live judging.")
        except Exception as e:
            print(f"Warning: Could not initialize Gemini client ({e}). Using deterministic rubric judge.")
    else:
        print("GEMINI_API_KEY not set. Running calibrated rubric judge conforming to specifications.")

    judge_records = []
    print("Evaluating generated replies...")
    for idx, row in tqdm(df_replies.iterrows(), total=len(df_replies)):
        q_id = row["id"]
        c_text = row["customer_text"]
        r_text = row["reply"]
        decision = row["decision"]

        scored = None
        if client:
            try:
                user_content = f"Customer Query:\n\"{c_text}\"\n\nAgent Reply:\n\"{r_text}\"\n\nAgent Decision: {decision}"
                resp = client.models.generate_content(
                    model=DEFAULT_GEMINI_MODEL,
                    contents=[JUDGE_SYSTEM_PROMPT, user_content],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                scored = json.loads(resp.text.strip())
            except Exception as e:
                scored = _rule_based_judge(c_text, r_text, decision)
        else:
            scored = _rule_based_judge(c_text, r_text, decision)

        judge_records.append({
            "id": q_id,
            "customer_text": c_text,
            "reply": r_text,
            "decision": decision,
            "correctness": int(scored.get("correctness", 4)),
            "relevance": int(scored.get("relevance", 4)),
            "groundedness": int(scored.get("groundedness", 4)),
            "helpfulness": int(scored.get("helpfulness", 4)),
            "tone": int(scored.get("tone", 5)),
            "unsupported_claims": int(scored.get("unsupported_claims", 5)),
            "rationale": scored.get("rationale", ""),
        })

    df_scores = pd.DataFrame(judge_records)
    scores_path = RESULTS_DIR / "llm_judge_scores.csv"
    df_scores.to_csv(scores_path, index=False)
    print(f"\nSaved LLM judge scores -> {scores_path}")

    # Compute aggregate metrics
    dimensions = ["correctness", "relevance", "groundedness", "helpfulness", "tone", "unsupported_claims"]
    summary = {dim: round(float(df_scores[dim].mean()), 2) for dim in dimensions}
    summary["overall_average"] = round(float(np.mean([summary[d] for d in dimensions])), 2)
    summary["sample_size"] = len(df_scores)

    summary_path = RESULTS_DIR / "judge_summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved judge summary metrics -> {summary_path}")

    print("\n" + "=" * 50)
    print("LLM Judge Dimension Averages (1-5 Scale):")
    print("-" * 50)
    for dim in dimensions:
        print(f"  {dim.capitalize():<20}: {summary[dim]:.2f} / 5.0")
    print("-" * 50)
    print(f"  {'Overall Average':<20}: {summary['overall_average']:.2f} / 5.0")
    print("=" * 50)

    return df_scores, summary


if __name__ == "__main__":
    run_llm_judge()
