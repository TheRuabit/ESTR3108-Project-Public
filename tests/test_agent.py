"""Tests for the minimal LangGraph agent.

This module validates the end-to-end run and tool execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.goob_ai.runner import run_once

if TYPE_CHECKING:  # pragma: no cover - type-checking only imports
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def test_echo_tool_end_to_end() -> None:
    """Ensure an 'echo' command routes to tool and returns expected output."""

    result = run_once("echo hello world")
    assert result.get("result") == "echo: text=hello world"
    assert result.get("messages") and result["messages"][-1]["content"] == "echo: text=hello world"


def test_no_tool_path() -> None:
    """When no tool is selected, assistant should just reflect the plan."""

    result = run_once("just say hi")
    assert isinstance(result.get("result"), str)
    assert "Respond directly" in (result.get("result") or "")


