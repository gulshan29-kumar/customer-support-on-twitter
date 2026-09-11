"""LLM grounded reply generation and conservative escalation engine using Gemini."""

import json
import os
import re
from typing import List, Dict, Any, Tuple
from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    DEFAULT_GEMINI_MODEL,
    DECISION_AUTO_HANDLE,
    DECISION_ESCALATE,
    ESCALATION_KEYWORDS,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_RETRIEVAL_SIMILARITY,
)
from src.preprocessing import clean_text


SYSTEM_PROMPT = """You are an official, professional customer support agent for AmazonHelp.
Your job is to draft a grounded response to an incoming customer message and decide whether to AUTO_HANDLE or ESCALATE_TO_HUMAN.

Rules and Constraints:
1. Grounding: Rely strictly on how AmazonHelp historically resolved similar issues in the provided historical cases.
2. Anti-Hallucination: DO NOT invent specific company policies, refund amounts, discounts, promotional offers, deadlines, or non-existent URLs.
3. Privacy: DO NOT include raw user IDs, order numbers, tracking numbers, or personal details.
4. Voice & Tone: Be concise, empathetic, and professional. NEVER state or mention that you are an AI, language model, or virtual assistant.
5. Conservative Escalation Policy:
   - Choose ESCALATE_TO_HUMAN if:
     * The customer's message is ambiguous, incomplete, or requires missing context.
     * The issue involves account security (hacked, suspended, unauthorized charges, fraud).
     * The issue demands an immediate refund, compensation, financial exception, or legal escalation.
     * The retrieved historical cases do not clearly support an answer or conflict with each other.
   - Choose AUTO_HANDLE ONLY when:
     * The customer's issue and intent are clear.
     * The historical cases provide consistent, clear, policy-compliant guidance (e.g. how to track a package, app restart steps, return window guidelines, reaching account support securely).
     * The reply requires no unverified claims or private account modifications.

You must respond in valid JSON format with three exact keys:
{
  "reply": "<concise support response>",
  "decision": "AUTO_HANDLE" or "ESCALATE_TO_HUMAN",
  "reason": "<clear explanation of why auto_handled or escalated>"
}
"""


def _check_rule_based_escalation(
    customer_message: str,
    intent: str,
    confidence: float,
    retrieved_cases: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Conservative pre-guard rules for escalation."""
    msg_lower = customer_message.lower()
    
    # 1. High risk or legal keywords
    for kw in ESCALATION_KEYWORDS:
        if kw in msg_lower:
            return True, f"Triggered sensitive keyword guard '{kw}' requiring human oversight."

    # 2. Extremely low intent confidence
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        return True, f"Intent confidence ({confidence:.2f}) below threshold ({MIN_CONFIDENCE_THRESHOLD})."

    # 3. Low retrieval similarity
    if not retrieved_cases or retrieved_cases[0]["similarity_score"] < MIN_RETRIEVAL_SIMILARITY:
        top_sim = retrieved_cases[0]["similarity_score"] if retrieved_cases else 0.0
        return True, f"Insufficient historical retrieval evidence (top similarity {top_sim:.2f} < {MIN_RETRIEVAL_SIMILARITY})."

    # 4. Account security & unauthorized charge requests require human auth
    if intent in ["account_or_access"]:
        return True, "Account security and authentication issues require direct human specialist verification."

    return False, ""


def _deterministic_fallback(
    customer_message: str,
    intent: str,
    confidence: float,
    retrieved_cases: List[Dict[str, Any]],
    rule_reason: str = "",
) -> Dict[str, Any]:
    """Grounded deterministic generator used when GEMINI_API_KEY is unset or API is unreachable."""
    top_case = retrieved_cases[0] if retrieved_cases else None
    
    if rule_reason:
        # Escalated by rule guard
        return {
            "reply": "We apologize for the inconvenience. To securely investigate your account details and resolve this, our specialist team will review your case directly.",
            "decision": DECISION_ESCALATE,
            "reason": rule_reason,
        }

    # If evidence is sufficient and intent is standard
    if intent == "delivery_or_delay":
        return {
            "reply": "We are sorry to hear your delivery is delayed! Please check your order tracking in 'Your Orders' for real-time carrier updates, or message us directly with your order details so we can investigate.",
            "decision": DECISION_AUTO_HANDLE,
            "reason": "Clear delivery inquiry with strong historical guidance on order tracking and carrier follow-up.",
        }
    elif intent == "technical_issue":
        return {
            "reply": "We're sorry for the technical trouble. Please try clearing the app cache, ensuring your device is updated, and restarting the application. If the issue persists, our technical team is here to help.",
            "decision": DECISION_AUTO_HANDLE,
            "reason": "Standard technical issue supported by historical troubleshooting guidance.",
        }
    elif intent == "product_or_service_info":
        return {
            "reply": "Thanks for reaching out! You can view full product specifications, compatibility details, and availability directly on the product detail page under 'Product Information'.",
            "decision": DECISION_AUTO_HANDLE,
            "reason": "General product inquiry grounded in historical catalog self-service guidance.",
        }
    else:
        return {
            "reply": "Thank you for reaching out to Amazon Help. We'd like to look into this for you; please connect with our team directly so we can review the specifics of your request.",
            "decision": DECISION_ESCALATE,
            "reason": f"Customer intent '{intent}' requires individual account inspection.",
        }


def generate_grounded_reply(
    customer_message: str,
    intent: str,
    confidence: float,
    retrieved_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a grounded reply and conservative escalation decision using Gemini.
    
    Args:
        customer_message: Cleaned incoming customer query.
        intent: Classified intent.
        confidence: Intent classifier confidence.
        retrieved_cases: Top historical cases with similarity scores.
        
    Returns:
        dict: {"reply": str, "decision": str, "reason": str}
    """
    # 1. Conservative pre-guard check
    must_escalate, rule_reason = _check_rule_based_escalation(
        customer_message, intent, confidence, retrieved_cases
    )

    # 2. If no Gemini API key configured, use deterministic grounded fallback
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    if not api_key:
        return _deterministic_fallback(
            customer_message, intent, confidence, retrieved_cases, rule_reason
        )

    # 3. Prepare evidence prompt for Gemini
    cases_text = ""
    for i, c in enumerate(retrieved_cases[:5], 1):
        cases_text += (
            f"Case {i} (Similarity: {c.get('similarity_score', 0.0):.3f}):\n"
            f"Customer: {c.get('customer_message', '')}\n"
            f"Support: {c.get('historical_support_reply', '')}\n\n"
        )

    user_prompt = f"""Incoming Customer Message:
\"{customer_message}\"

Predicted Intent: {intent} (Confidence: {confidence:.2f})

Top 5 Historical AmazonHelp Cases:
{cases_text}

Pre-guard suggestion: {"ESCALATE due to: " + rule_reason if must_escalate else "Eligible for AUTO_HANDLE if evidence supports it."}

Generate the JSON response conforming to the rules:"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=[SYSTEM_PROMPT, user_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        
        content = response.text.strip()
        parsed = json.loads(content)
        
        reply = parsed.get("reply", "").strip()
        decision = parsed.get("decision", "").strip()
        reason = parsed.get("reason", "").strip()

        # Enforce conservative pre-guard override if rule triggered
        if must_escalate and decision == DECISION_AUTO_HANDLE:
            decision = DECISION_ESCALATE
            reason = f"Conservative guard override: {rule_reason} (LLM reason: {reason})"

        return {
            "reply": reply,
            "decision": decision,
            "reason": reason,
        }
    except Exception as e:
        print(f"Warning: Gemini generation error: {e}. Falling back to deterministic guard.")
        return _deterministic_fallback(
            customer_message, intent, confidence, retrieved_cases, rule_reason
        )
