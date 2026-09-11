"""Interactive Gradio Web Application for Hiver AI Support Agent.

Displays:
- Customer Message input
- Predicted Intent and Confidence
- Grounded Support Reply
- Conservative Escalation Decision (AUTO_HANDLE vs ESCALATE_TO_HUMAN) with Stated Reason
- Top-5 Retrieved Historical Cases with Similarity Scores
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from src.pipeline import run_support_agent


def handle_support_query(customer_message: str):
    """Gradio callback to run the customer support agent pipeline."""
    if not customer_message.strip():
        return (
            "Please enter a customer message.",
            "N/A",
            0.0,
            "N/A",
            "No message provided.",
            "No historical cases retrieved.",
        )

    res = run_support_agent(customer_message)

    intent_text = res["intent"]
    conf_val = float(res["confidence"])
    decision_text = res["decision"]
    reason_text = res["reason"]
    reply_text = res["reply"]

    # Format retrieved cases nicely
    cases_display = ""
    for i, c in enumerate(res.get("retrieved_cases", []), 1):
        cases_display += (
            f"### Case {i} | Similarity: {c.get('similarity_score', 0.0):.3f} | Intent: `{c.get('intent', 'unknown')}`\n"
            f"**Customer:** {c.get('customer_message', '')}\n\n"
            f"**Historical AmazonHelp Reply:** {c.get('historical_support_reply', '')}\n\n"
            f"---\n"
        )
    if not cases_display:
        cases_display = "No relevant historical cases found."

    return (
        reply_text,
        intent_text,
        conf_val,
        decision_text,
        reason_text,
        cases_display,
    )


def create_demo():
    """Build and configure the Gradio UI."""
    custom_css = """
    .decision-auto { background-color: #d1e7dd; color: #0f5132; padding: 6px 12px; border-radius: 4px; font-weight: bold; }
    .decision-escalate { background-color: #f8d7da; color: #842029; padding: 6px 12px; border-radius: 4px; font-weight: bold; }
    """

    with gr.Blocks(title="AmazonHelp AI Support Agent | Hiver Take-Home") as demo:
        gr.Markdown(
            """
            # 📦 AmazonHelp AI Customer Support Agent
            ### Grounded Support Generation & Conservative Escalation System
            *Built for Hiver SDE Intern Take-Home Assignment*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Customer Input")
                input_text = gr.Textbox(
                    label="Customer Tweet / Message",
                    placeholder="e.g., Where is my package? The tracking has not updated since yesterday.",
                    lines=4,
                )
                
                gr.Examples(
                    examples=[
                        ["Where is my package? It was supposed to be delivered yesterday and tracking has not updated."],
                        ["Someone hacked into my Amazon account, changed my email, and charged $400 to my card!"],
                        ["Can you please cancel my order #104-987654 and process an immediate refund?"],
                        ["Does the Amazon Fire TV Stick support 4K HDR streaming on external soundbars?"],
                        ["Worst customer service ever! Waited 45 minutes on chat and agent hung up on me."],
                        ["What time do your customer support phone lines open on Sunday morning?"],
                    ],
                    inputs=input_text,
                    label="Sample Customer Queries",
                )

                submit_btn = gr.Button("Generate Support Response", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 2. Agent Decision & Reply")
                
                with gr.Row():
                    out_intent = gr.Textbox(label="Predicted Intent", interactive=False)
                    out_confidence = gr.Number(label="Intent Confidence", interactive=False)
                
                with gr.Row():
                    out_decision = gr.Textbox(label="Escalation Decision", interactive=False)
                    out_reason = gr.Textbox(label="Decision Reason", interactive=False)

                out_reply = gr.Textbox(
                    label="Draft Support Reply (Grounded)",
                    lines=4,
                    interactive=False,
                )

        gr.Markdown("---")
        gr.Markdown("### 3. Historical AmazonHelp Grounding Evidence (Top 5 Retrieved Cases)")
        out_cases = gr.Markdown("Retrieved historical cases will appear here after clicking submit.")

        submit_btn.click(
            fn=handle_support_query,
            inputs=[input_text],
            outputs=[
                out_reply,
                out_intent,
                out_confidence,
                out_decision,
                out_reason,
                out_cases,
            ],
        )

    return demo


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    server_name = os.environ.get("HOST", "0.0.0.0")
    demo = create_demo()
    print(f"Launching Gradio demo server on {server_name}:{port}...")
    demo.launch(server_name=server_name, server_port=port, share=False, inbrowser=False)
