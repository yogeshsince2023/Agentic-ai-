"""
FastAPI backend for the Agentic AI futuristic web UI.
Serves the HTML frontend and exposes the agent as an API.

Run: python app.py
Opens at: http://localhost:8000
"""

import io
import sys
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from agent import load_env, create_client, agent_loop

load_env()
client = create_client()

app = FastAPI()

# In-memory conversation history (single user for simplicity)
# ponytail: global state is fine for a local demo tool. Upgrade path: session-based state.
conversation_history = []


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Process a chat message through the agent and return the response."""
    # Capture tool-call prints
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        response = agent_loop(client, req.message, conversation_history)
    except Exception as e:
        response = f"Error: {e}"
    finally:
        sys.stdout = old_stdout

    tool_log = captured.getvalue().strip()

    return JSONResponse({
        "response": response,
        "tool_log": tool_log,
    })


@app.post("/api/clear")
async def clear():
    """Clear conversation history."""
    conversation_history.clear()
    return JSONResponse({"status": "cleared"})


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
