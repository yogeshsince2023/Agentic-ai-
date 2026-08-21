"""
Web UI for Agentic AI — wraps the same agent logic in a Gradio chat interface.

Run: python ui.py
Opens at: http://localhost:7860
"""

import io
import sys
import gradio as gr
from agent import load_env, create_client, agent_loop


load_env()
client = create_client()


def respond(message: str, history: list, gemini_history: list):
    """Handle a user message: run the agent loop and return the response."""
    if gemini_history is None:
        gemini_history = []

    # Capture tool-call prints so we can show them in the UI
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        response = agent_loop(client, message, gemini_history)
    except Exception as e:
        response = f"Error: {e}"
    finally:
        sys.stdout = old_stdout

    tool_log = captured.getvalue().strip()

    # Show tool activity above the final answer
    full_response = ""
    if tool_log:
        full_response += f"**Tool Activity:**\n```\n{tool_log}\n```\n\n"
    full_response += response

    return full_response, gemini_history


# --- UI ---

DESCRIPTION = """
# 🤖 Agentic AI
**Powered by Google Gemini** — I can search the web, do math, read/write files, and more.
"""

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)

    gemini_state = gr.State([])

    chatbot = gr.Chatbot(
        label="Chat",
        height=500,
        buttons=["copy_all"],
        placeholder="Ask me anything! Try: *What is 2^10 + sqrt(144)?*",
        examples=[
            {"text": "What is 2^10 + sqrt(144)?", "display_text": "🔢 Math: 2^10 + sqrt(144)"},
            {"text": "Search the web for latest AI news", "display_text": "🔍 Web search: AI news"},
            {"text": "What time is it right now?", "display_text": "🕐 Current time"},
            {"text": "List all files in the current directory", "display_text": "📁 List files"},
        ],
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask me anything...",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat", size="sm")

    # Wire up events
    def user_submit(message, chat_history, gemini_history):
        if not message.strip():
            return "", chat_history, gemini_history
        chat_history = chat_history + [
            {"role": "user", "content": message},
        ]
        response, gemini_history = respond(message, chat_history, gemini_history)
        chat_history = chat_history + [
            {"role": "assistant", "content": response},
        ]
        return "", chat_history, gemini_history

    def clear_chat():
        return [], []

    msg.submit(user_submit, [msg, chatbot, gemini_state], [msg, chatbot, gemini_state])
    send_btn.click(user_submit, [msg, chatbot, gemini_state], [msg, chatbot, gemini_state])
    clear_btn.click(clear_chat, outputs=[chatbot, gemini_state])


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
            .gradio-container { max-width: 850px !important; }
            footer { display: none !important; }
        """,
    )
