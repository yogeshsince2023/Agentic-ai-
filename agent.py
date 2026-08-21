"""
Agentic AI — A tool-using AI agent powered by Google Gemini.

This single script demonstrates the core agent loop that powers systems
like ChatGPT, Gemini, and other AI agents:

    User Message → LLM Thinks → Calls Tool → Observes Result → Thinks Again → Responds

Tools available:
  - calculator     : Evaluate math expressions
  - web_search     : Search the web via DuckDuckGo (free, no API key)
  - read_file      : Read file contents
  - write_file     : Write content to files
  - list_directory : List files in a directory
  - get_current_time : Get current date/time

Setup:
  1. pip install -r requirements.txt
  2. Get a free Gemini API key from https://aistudio.google.com/apikey
  3. python agent.py

You can set GEMINI_API_KEY as an environment variable or enter it at the prompt.
"""

import os
import json
import math
from google import genai
from google.genai import types


# ╔══════════════════════════════════════════════════════════════╗
# ║  TOOLS — Plain Python functions the agent can call          ║
# ╚══════════════════════════════════════════════════════════════╝

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic and math
    functions like sqrt, sin, cos, log, pow, pi, e, etc.

    Args:
        expression: A math expression to evaluate, e.g. '2 + 2' or 'sqrt(144) * pi'
    """
    # ponytail: eval with restricted builtins — safe enough since the LLM generates
    # the expression, not raw user input. Upgrade path: use ast.literal_eval or a
    # proper math parser if this ever faces untrusted input directly.
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def read_file(filepath: str) -> str:
    """Read the contents of a text file.

    Args:
        filepath: Path to the file to read.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 10_000:
            return content[:10_000] + "\n... (truncated at 10,000 chars)"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filepath: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed.

    Args:
        filepath: Path to the file to write.
        content: The text content to write.
    """
    try:
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ Wrote {len(content)} characters to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_directory(path: str = ".") -> str:
    """List files and folders in a directory.

    Args:
        path: Directory path to list. Defaults to current directory.
    """
    try:
        entries = os.listdir(path)
        if not entries:
            return "(empty directory)"
        lines = []
        for name in sorted(entries):
            full = os.path.join(path, name)
            icon = "📁" if os.path.isdir(full) else "📄"
            lines.append(f"  {icon} {name}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str) -> str:
    """Search the web and return top results with titles, snippets, and URLs.

    Args:
        query: The search query.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"{i}. **{r['title']}**\n   {r['body']}\n   🔗 {r['href']}")
        return "\n\n".join(output)
    except ImportError:
        return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Search error: {e}"


def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  AGENT CORE — The loop that makes it "agentic"             ║
# ╚══════════════════════════════════════════════════════════════╝

# Tool registry: name → callable
TOOLS = {
    "calculator": calculator,
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "web_search": web_search,
    "get_current_time": get_current_time,
}

SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools.

Guidelines:
- Use tools when you need real data (calculations, files, web info, time).
- Briefly explain your reasoning before calling a tool.
- If a task needs multiple steps, do them one at a time.
- Be concise and direct in your final answers.
- If you can answer from your own knowledge without tools, just answer directly.
"""


def load_env():
    """Load KEY=VALUE pairs from .env file if it exists (no dependency needed)."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def create_client() -> genai.Client:
    """Create the Gemini API client, prompting for key if needed."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("🔑 Get a free API key at: https://aistudio.google.com/apikey")
        api_key = input("   Enter your Gemini API key: ").strip()
        if not api_key:
            print("❌ No API key provided. Exiting.")
            exit(1)
        os.environ["GEMINI_API_KEY"] = api_key  # Cache for session
    return genai.Client(api_key=api_key)


def agent_loop(client: genai.Client, user_message: str, history: list) -> str:
    """
    The core agent loop — this is where the "agentic" behavior happens.

    How it works:
    ┌─────────────────────────────────────────────────────┐
    │  1. User message is added to conversation history   │
    │  2. Send history + tools to Gemini                  │
    │  3. Gemini responds with either:                    │
    │     a) Text → return it (done!)                     │
    │     b) Function call → execute tool → go to step 2  │
    │  4. Repeat until text response or max iterations    │
    └─────────────────────────────────────────────────────┘
    """
    # Step 1: Add user message to history
    history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
    )

    max_turns = 10  # Safety valve: prevent infinite tool-calling loops

    for turn in range(max_turns):
        # Step 2: Call Gemini with full history + tool definitions
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=list(TOOLS.values()),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        # Step 3: Process the response
        parts = response.candidates[0].content.parts
        if not parts:
            return response.text or "(empty response)"

        function_calls = [p for p in parts if p.function_call]
        text_parts = [p.text for p in parts if p.text]

        # 3a: If no function calls → it's a final text answer
        if not function_calls:
            history.append(response.candidates[0].content)
            return "\n".join(text_parts)

        # 3b: Function calls found → execute each tool
        # Print any reasoning text the model included
        if text_parts:
            print(f"  💭 {' '.join(text_parts)}")

        # Add model's response (with function calls) to history
        history.append(response.candidates[0].content)

        # Execute each tool and collect results
        fn_response_parts = []
        for part in function_calls:
            fc = part.function_call
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}

            # Show what the agent is doing
            args_str = ", ".join(f'{k}="{v}"' for k, v in fn_args.items())
            print(f"  🔧 Calling: {fn_name}({args_str})")

            # Execute
            if fn_name in TOOLS:
                result = TOOLS[fn_name](**fn_args)
            else:
                result = f"Unknown tool: {fn_name}"

            # Show truncated result
            preview = result[:300] + ("..." if len(result) > 300 else "")
            print(f"  📋 Result: {preview}\n")

            fn_response_parts.append(
                types.Part.from_function_response(
                    name=fn_name, response={"result": result}
                )
            )

        # Feed tool results back to Gemini (step 2 repeats)
        history.append(types.Content(role="user", parts=fn_response_parts))

    return "⚠️ Agent hit the max tool-use limit (10 turns). Try a simpler question."


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN — Interactive CLI chat loop                           ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    import sys, io
    # Fix Windows console encoding for emoji/unicode
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    load_env()  # Load API key from .env file

    print()
    print("=" * 54)
    print("  Agentic AI - Powered by Google Gemini")
    print()
    print("  Tools: calculator, web_search, file read/write,")
    print("         list_directory, get_current_time")
    print()
    print("  Commands: 'quit' to exit, 'clear' to reset chat")
    print("=" * 54)
    print()

    client = create_client()
    history = []  # Conversation memory

    print("\n✅ Agent ready! Ask me anything.\n")

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("🔄 Conversation cleared.\n")
            continue

        print()  # Spacing before agent output
        try:
            response = agent_loop(client, user_input, history)
            print(f"🤖 Agent: {response}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
            # ponytail: bare except-and-print is fine for a CLI learning tool.
            # The user sees the error and can retry. No data loss risk.


if __name__ == "__main__":
    main()
