"""Unified AI Customer Support Agent pipeline for AmazonHelp."""

from typing import Dict, Any
from src.preprocessing import clean_text
from src.classifier import predict_intent
from src.retrieval import retrieve_similar_cases
from src.llm_agent import generate_grounded_reply


def run_support_agent(customer_message: str) -> Dict[str, Any]:
    """Execute the full end-to-end customer support agent pipeline.
    
    1. Clean input message
    2. Classify intent and confidence
    3. Retrieve top-5 historically similar AmazonHelp customer support pairs
    4. Generate grounded reply with conservative escalation decision (Gemini + guard)
    
    Args:
        customer_message: Raw customer message text.
        
    Returns:
        dict: Full agent response dictionary.
    """
    cleaned = clean_text(customer_message)
    if not cleaned:
        return {
            "intent": "other",
            "confidence": 0.0,
            "retrieved_cases": [],
            "reply": "Please provide more details regarding the issue you are experiencing so we can assist you.",
            "decision": "ESCALATE_TO_HUMAN",
            "reason": "Empty or uninterpretable message.",
        }

    # Step 1: Predict Intent
    pred = predict_intent(cleaned)
    intent = pred["intent"]
    confidence = pred["confidence"]

    # Step 2: Retrieve Top-5 Historical Cases
    retrieved_cases = retrieve_similar_cases(cleaned, top_k=5)

    # Step 3: Grounded Generation & Escalation Decision
    gen_result = generate_grounded_reply(
        customer_message=cleaned,
        intent=intent,
        confidence=confidence,
        retrieved_cases=retrieved_cases,
    )

    return {
        "intent": intent,
        "confidence": confidence,
        "retrieved_cases": retrieved_cases,
        "reply": gen_result.get("reply", ""),
        "decision": gen_result.get("decision", "ESCALATE_TO_HUMAN"),
        "reason": gen_result.get("reason", "Standard review required."),
    }


if __name__ == "__main__":
    queries = [
        "Where is my package? The tracking has been stuck on out for delivery since yesterday!",
        "Someone hacked into my Amazon account and changed my password! Help!",
        "Can you tell me if the Kindle Paperwhite is waterproof?",
    ]
    for q in queries:
        print(f"\n" + "=" * 80)
        print(f"Customer: {q}")
        res = run_support_agent(q)
        print(f"Intent:   {res['intent']} (confidence: {res['confidence']})")
        print(f"Decision: {res['decision']}")
        print(f"Reason:   {res['reason']}")
        print(f"Reply:    {res['reply']}")
        print(f"Retrieved {len(res['retrieved_cases'])} historical cases.")
