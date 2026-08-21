# Agentic AI

A tool-using AI agent powered by Google Gemini with a futuristic web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Gemini](https://img.shields.io/badge/LLM-Gemini%203.6-purple)

## What It Does

A ReAct-style AI agent that can **think, use tools, observe results, and respond** — the same core loop behind ChatGPT, Gemini, and other AI agents.

### Built-in Tools

| Tool | Description |
|------|-------------|
| `calculator` | Evaluate math expressions (sqrt, sin, pi, etc.) |
| `web_search` | Search the web via DuckDuckGo (free, no API key) |
| `read_file` | Read file contents |
| `write_file` | Create/write files |
| `list_directory` | List files in a directory |
| `get_current_time` | Get current date & time |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free Gemini API key
#    https://aistudio.google.com/apikey

# 3. Create a .env file
echo GEMINI_API_KEY=your-key-here > .env
```

## Run

### Web UI (Futuristic)

```bash
python app.py
# Opens at http://localhost:8000
```

### Terminal CLI

```bash
python agent.py
```

## Project Structure

```
├── agent.py          # Core agent logic — tools + agent loop
├── app.py            # FastAPI backend for web UI
├── index.html        # Futuristic frontend (particles, glassmorphism, ambient audio)
├── ui.py             # Gradio UI (alternative, simpler)
├── requirements.txt  # Dependencies
└── .env              # API key (not committed)
```

## How the Agent Loop Works

```
User Message → Gemini Thinks → Calls Tool → Observes Result → Thinks Again → Responds
```

The core logic is in `agent.py` → `agent_loop()` — ~40 lines that implement the full agentic pattern.

## License

MIT
