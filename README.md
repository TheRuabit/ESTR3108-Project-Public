# Goob AI: LangGraph Agent Scaffold

Goob AI is a small LangGraph-based agent framework for prototyping tool-using
agents. It includes a Gradio interface for single prompts and Hugging Face
evaluation submissions.

## How This Implementation Uses LangChain and LangGraph

### Division of Responsibilities

This project uses **LangChain integrations for retrieval and search helpers**,
not LangChain's prebuilt agent executor. `toolkit.py` optionally uses
`WikipediaLoader` and `ArxivLoader` from `langchain_community`,
`RecursiveCharacterTextSplitter` from `langchain_text_splitters`, and
`TavilySearch` from `langchain_tavily`. When an optional integration is
unavailable, the relevant tool uses its smaller built-in fallback where one is
implemented. For example, splitting falls back to overlapping character
slices, and web search can call Tavily's HTTP API directly.

The `wiki_retrieve` tool shows the retrieval pipeline clearly: it loads
Wikipedia documents, preserves their metadata while splitting them into
chunks, pre-ranks chunks with TF-IDF (or token overlap), optionally re-ranks
them with a CrossEncoder, and returns the best source-labelled contexts. It
does not generate the final answer; the planner receives those contexts on the
next graph cycle.

**LangGraph provides the agent control flow.** `graph.py` creates a
`StateGraph[AgentState]` rather than relying on a fixed agent loop. The
project's tools are plain typed Python callables held in a `ToolRegistry`, so
they can be replaced or extended without converting them into LangChain Tool
objects.

### Graph State and Nodes

`AgentState` in `types_new.py` is the shared state passed between nodes. It
holds the conversation messages, selected plan, pending tool call, most recent
result, available tool names and short tool descriptions, retained facts,
action history, and per-tool attempt counters. Before planning, the graph adds
the registered tool names and docstrings to the state so an LLM-backed planner
knows what it may call.

The compiled graph contains three nodes:

1. **`plan`** selects the configured planner: `simple`, `ollama`, or Azure
   OpenAI. LLM planners return a two-line `plan:` / `tool:` response, which is
   parsed into a plan and an optional tool call.
2. **`act`** looks up the requested tool in the registry, calls it with the
   parsed arguments, appends its output as an assistant message, and clears
   the completed tool call. With no tool call, it emits the planner text as
   the final response.
3. **`assess`** records useful tool output for the next planning cycle. It
   retains more text for web search, calculation, and table analysis; stores a
   shorter summary for other tools; then rebuilds the compact `facts_text`
   supplied to the planner.

The normal execution path is `plan → act → assess → plan`. When `act` emits a
final response rather than running a tool, the graph ends. A per-tool attempt
limit prevents one tool from being selected indefinitely; it is `3` for
graph-only use and `2` when the Gradio `BasicAgent` sets no override.

### Application Entry Points

`BasicAgent` in `app.py` builds the compiled graph once, creates an initial
state containing the user's message, and invokes it with a recursion limit of
50. The Gradio **Single Input** tab explicitly uses the Ollama planner. The
**Full Evaluation & Submission** tab reads `PLANNER` (defaulting to Ollama),
runs the same agent for each Hugging Face question, and submits the resulting
answers. `runner.py` is the smaller programmatic entry point and lets the
graph select the planner from `PLANNER` (defaulting to `simple`).

## Requirements

- Python 3.10 or later
- An accessible Ollama server and model when using Ollama (including the
  **Single Input** tab)

Install the project and the Ollama client package:

```bash
python -m pip install -e . ollama
```

The project imports the Ollama client even when another planner is selected,
so `ollama` must currently be installed. To load variables from a `.env` file,
also install the optional package:

```bash
python -m pip install python-dotenv
```

## Run the Web Interface

Set an Ollama model that is available from your server, then start the app:

```powershell
$env:OLLAMA_MODEL = "your-model"
python src/goob_ai/app.py
```

The interface normally opens at `http://127.0.0.1:7860`; use the URL printed
by Gradio if that port is already occupied. The **Single Input** tab accepts a
prompt and optional image files. The **Full Evaluation & Submission** tab
fetches the Hugging Face evaluation questions, runs the agent on each one, and
submits the answers after you sign in to Hugging Face.

For Windows Command Prompt, replace `$env:NAME = "value"` with
`set NAME=value`.

## Planner Configuration

Set `PLANNER` before launching the app or calling `build_agent`:

| Value | Behaviour |
| --- | --- |
| `simple` | Uses the deterministic built-in planner without an LLM backend. |
| `ollama` | Uses an Ollama-compatible HTTP endpoint. This is the default for full evaluation runs. |
| `azure` or `azure_sdk` | Uses the Azure OpenAI chat-completions planner. |

The standalone `runner` and `build_agent()` default to `simple` when
`PLANNER` is unset. The **Single Input** tab explicitly uses `ollama`, while
the **Full Evaluation** tab uses `PLANNER` and otherwise defaults to
`ollama`.

### Ollama

```powershell
$env:PLANNER = "ollama"
$env:OLLAMA_HOST = "http://localhost:11434" # optional; this is the default
$env:OLLAMA_MODEL = "your-model"
python src/goob_ai/app.py
```

If `OLLAMA_MODEL` is unset, the code requests `deepseek-v3.1:671b-cloud`.
Choose a different value when that model is unavailable on your server. An
Ollama request failure produces a direct planner-error response; it does not
switch to the simple planner automatically.

### Azure OpenAI

```powershell
$env:PLANNER = "azure_sdk"
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com"
$env:AZURE_OPENAI_API_VERSION = "2024-02-01"
$env:AZURE_API_KEY = "your-key"
$env:AZURE_OPENAI_DEPLOYMENT = "your-deployment-name"
python src/goob_ai/app.py
```

You may put the same `NAME=value` entries in `.env` when `python-dotenv` is
installed. The Azure planner reads `AZURE_API_KEY`, not `AZURE_OPENAI_KEY`.

### Other Runtime Variables

| Variable | Purpose |
| --- | --- |
| `TOOL_ATTEMPT_LIMIT` | Maximum attempts for each tool. The Gradio app sets `2` when it is unset; graph-only usage defaults to `3`. |
| `TAVILY_API_KEY` | Required by `web_search` to query the Tavily web-search API. |
| `SPACE_ID` | Hugging Face Space identifier used to construct the public code URL in an evaluation submission. |

## Built-in Tools

The default tool registry contains the following tools:

- `web_search`: search the web.
- `wiki_retrieve`: retrieve and rank Wikipedia context chunks; it does not
  generate an answer itself.
- `arxiv_search`: search arXiv entries.
- `calculator`: evaluate a restricted arithmetic expression.
- `analyze_table`: summarise CSV text.
- `analyze_image` and `analyze_remote_image`: inspect local image paths or
  image URLs, with optional OCR support.
- `analyze_video` and `analyze_video_by_chapter`: inspect video metadata and,
  where available, subtitles for a requested time range.

The planner chooses a tool and its arguments. You can supply a custom tool
registry to `build_graph()` or `build_agent()` when you need different tools.

## Project Structure

- `src/goob_ai/app.py`: Gradio user interface and Hugging Face evaluation flow.
- `src/goob_ai/graph.py`: LangGraph construction and tool-execution loop.
- `src/goob_ai/planners.py`: simple, Ollama, and Azure OpenAI planners.
- `src/goob_ai/nodes.py`: actor and assessment nodes.
- `src/goob_ai/toolkit.py`: built-in tools and the default tool registry.
- `src/goob_ai/types_new.py`: typed agent state, messages, and tool protocol.
- `src/goob_ai/runner.py`: programmatic single-turn entry point.

## Programmatic Use

```python
from goob_ai.runner import run_once

result = run_once("What is 2 * (3 + 4)?")
print(result["result"])
```

Set `PLANNER` and any backend variables in the environment before calling
`run_once`.

## Tests

Install the test runner, then execute the suite:

```bash
python -m pip install pytest
python -m pytest
```

The current tests need maintenance before they can serve as a passing
verification suite: `test_agent.py` expects an `echo` tool that is not in the
default registry, and `test_wiki_retrieve.py` expects an HTTP fallback that
the current `wiki_retrieve()` implementation does not provide.
