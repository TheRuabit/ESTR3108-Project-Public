"""Runner entrypoints for the minimal LangGraph agent."""

from __future__ import annotations

from typing import List

import os
import sys

# Ensure parent directory is in sys.path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from goob_ai.graph import build_graph
from goob_ai.types_new import AgentState, Message


def run_once(user_input: str) -> AgentState:
    """Run a single-turn interaction through the graph.

    Args:
        user_input: Text from the user.

    Returns:
        Final state after running the graph.
    """

    messages: List[Message] = [{"role": "user", "content": user_input}]
    state: AgentState = {"messages": messages}
    # Graph chooses planner by env var internally (PLANNER=simple|ollama|azure)
    graph = build_graph()
    compiled = graph.compile()
    return compiled.invoke(state, config={"recursion_limit": 50})


if __name__ == "__main__":  # pragma: no cover - manual quickstart
    final = run_once("hello world")
    print(final.get("result"))


