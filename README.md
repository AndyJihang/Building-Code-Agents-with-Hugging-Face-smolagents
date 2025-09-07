# Building-Code-Agents-with-Hugging-Face-smolagents

A hands-on notebook suite showing how to build, secure, trace, and evaluate code-executing LLM agents with smolagents. You’ll learn how to: (1) write tools, (2) execute Python safely, (3) monitor with OpenTelemetry + Phoenix, and (4) assemble a deep-research agent that searches, reads, and synthesizes sources.

📁 What’s in this repo
Notebook	Purpose
Introduction_to_Code_Agents.ipynb	Core concepts: CodeAgent, models, tools, prompts, max_steps.
Secure_Code_Execution.ipynb	Safe Python with LocalPythonExecutor and E2B sandbox; allow-lists, timeouts, resource limits.
Monitoring_and_Evaluating_your_Agent.ipynb	OpenTelemetry tracing via Phoenix; collect spans, analyze tool calls, score behavior.
Build_a_deep_research_agent.ipynb	A multi-step research agent: plan → search → read → extract → synthesize → critique.
All notebooks run in Colab or local Jupyter.

🚀 Quickstart
1) Environment
# Python 3.10+ recommended
pip install -U smolagents arize-phoenix opentelemetry-exporter-otlp-proto-http \
  requests markdownify python-dotenv
# Optional (E2B remote sandbox execution)
pip install -U e2b-code-interpreter
Create .env (edit as needed):
# One of the two model providers is enough ─ choose your stack
HF_API_KEY=hf_xxx                     # for Hugging Face Inference / Together (via smolagents HfApiModel)
OPENAI_API_KEY=sk-xxx                 # for OpenAI-compatible endpoints (via OpenAIServerModel)

# Tracing UI (Phoenix)
DLAI_LOCAL_URL=http://127.0.0.1:{port}/

# Optional: E2B sandbox for remote code execution
E2B_API_KEY=e2b_xxx
2) Launch Phoenix (optional but recommended)
import phoenix as px
px.launch_app()  # prints a local URL like http://127.0.0.1:6006/
If you don’t launch Phoenix, tracing calls to http://127.0.0.1:6006/v1/traces will fail with “Connection refused” (that’s expected).
3) Open a notebook and run cells
Start with Introduction_to_Code_Agents.ipynb.

🧠 Model backends
You can swap providers with minimal changes.
Hugging Face / Together (via HfApiModel)

from smolagents import HfApiModel
model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct", provider="together")
OpenAI-compatible
from smolagents import OpenAIServerModel
model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base="https://api.openai.com/v1",
    temperature=0.2,
)
Tip: Keep the rest of your agent code identical; only the model line changes.
🔐 Secure code execution
LocalPythonExecutor – run Python with an import allow-list, blocked shell escapes, and loop/time limits.
E2B Sandbox – optional remote, isolated environment.
Example (strict local executor):
from smolagents.local_python_executor import LocalPythonExecutor
executor = LocalPythonExecutor(
    allowed_imports=["math", "random", "numpy", "pandas"],
    timeout=10,          # seconds per code block
    max_output_chars=8000
)
Common patterns:
Don’t pass ! shell commands (they’re treated as invalid syntax on purpose).
Whitelist only the libraries your tools truly need.
For long-running or network installs, switch to E2B (executor_type="e2b").
🔭 Tracing & evaluation (Phoenix + OpenTelemetry)
Register a tracer once per kernel:
import os
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

tracer_provider = register(
    project_name="Customer-Success",
    endpoint=os.getenv("DLAI_LOCAL_URL").format(port="6006") + "v1/traces",
)
SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)
Then run your agent as usual:
from smolagents import CodeAgent
agent = CodeAgent(model=model, tools=[], max_steps=5)
agent.run("What is the 100th Fibonacci number?")
Open Phoenix (the printed URL) to explore spans, tool calls, inputs/outputs, and timing.
If you see “Connection refused”: launch Phoenix first (px.launch_app()), or disable tracing by skipping registration.
🧪 Scoring tool use (from the evaluation notebook)
The notebook shows how to:
Pull spans into a Pandas DataFrame.
Extract AGENT tasks and TOOL invocations (column names vary by version; the code includes fallbacks).
Compare expected vs. actual tool calls.
If you get empty tool sets:
Ensure your tools are actually called (the agent can answer directly without tools).
Lower max_steps may end before tool planning; try max_steps=5+.
Make tools discoverable: write precise docstrings and input schemas.
Verify tracing is enabled before the run (spans are created at runtime).
🍦 Example: “Ice-cream” tools
Docstrings must describe every parameter, or smolagents will raise DocstringParsingException.
from smolagents import tool

menu_prices = {"crepe nutella": 1.50, "vanilla ice cream": 2.0, "maple pancake": 1.0}
ORDER_BOOK = {}

@tool
def place_order(quantities: dict, session_id: int) -> None:
    """Places a pre-order of snacks.

    Args:
        quantities: Mapping of item name to quantity. Keys must exist in the menu.
        session_id: Unique identifier of the client session.
    """
    assert all(k in menu_prices for k in quantities), "Unknown menu item"
    ORDER_BOOK[session_id] = quantities

@tool
def get_prices(quantities: dict) -> str:
    """Computes the total price for the requested quantities.

    Args:
        quantities: Mapping of item name to quantity. Keys must exist in the menu.
    """
    assert all(k in menu_prices for k in quantities), "Unknown menu item"
    total = sum(menu_prices[k] * v for k, v in quantities.items())
    return f"Menu: {menu_prices}\nTotal: ${total:.2f}"
🧩 Deep-research agent (high level)
Pipeline inside Build_a_deep_research_agent.ipynb:
Plan: decompose the query into sub-questions.
Search: web queries with diversified operators/time windows.
Read: fetch pages, convert to markdown, chunk.
Extract: claims/numbers/citations into structured cards.
Synthesize: answer with sources & uncertainty.
Critique: red-team the result; propose next evidence.
🛠 Troubleshooting
openinference cannot be pip-installed
Use arize-phoenix and openinference-instrumentation-smolagents (installed above). Import tracing from phoenix.otel and instrument via SmolagentsInstrumentor.
Phoenix errors: Connection refused
Launch the app first: import phoenix as px; px.launch_app() and keep that kernel running. The endpoint should be http://127.0.0.1:6006/v1/traces.
Overriding of current TracerProvider is not allowed
Only call register(...) once per Python process. Restart kernel if needed.
DocstringParsingException for tools
Every function argument must be documented in the docstring’s Args: block.
Agent ends with {final_answer} but no tools
That’s valid: the LLM can answer directly. If you require a tool, state it in the task and ensure the tool description matches the need.
RecursionError when calling final_answer
Don’t call final_answer from your own code; let agent.run(...) return the final string.
🧷 Repo structure
.
├── Build_a_deep_research_agent.ipynb
├── Introduction_to_Code_Agents.ipynb
├── Monitoring_and_Evaluating_your_Agent.ipynb
├── Secure_Code_Execution.ipynb
└── README.md
📎 Acknowledgements
smolagents by Hugging Face
Arize Phoenix for OpenTelemetry tracing & analysis
📜 License
Add a LICENSE file (e.g., MIT) to clarify reuse.
🤝 Contributing
Issues and PRs welcome: fix typos, add examples, extend tools, or contribute evaluation recipes.
Minimal example
import os
from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)
agent = CodeAgent(model=model, tools=[], max_steps=3)
print(agent.run("Compute the 100th Fibonacci number."))
Happy agent-building! If you want, I can also provide a requirements.txt and a tiny Makefile for one-command setup.
