"""Tests for wiki_retrieve tool (retrieve -> split -> rank) without external calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import re
import pytest
import sys
import os
import sys

# Add src/ to sys.path to allow absolute import of goob_ai modules for test discovery
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from goob_ai import toolkit

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def _fake_wiki_blocks() -> str:
    """Return a deterministic fake wiki_search payload with two documents."""
    return (
        '<Document source="Wikipedia" title="Alpha" url="https://example.org/a"/>\n'
        "Alpha is the first letter of the Greek alphabet. It is often used in science.\n"
        "</Document>\n\n---\n\n"
        '<Document source="Wikipedia" title="Beta" url="https://example.org/b"/>\n'
        "Beta is the second letter of the Greek alphabet. It is used in beta testing.\n"
        "</Document>\n"
    )


@pytest.mark.parametrize("query", ["Greek alphabet", "Alpha letter"])
def test_wiki_retrieve_offline_monkeypatched(
    monkeypatch: MonkeyPatch, query: str
) -> None:
    """Ensure wiki_retrieve works via HTTP-fallback path with deterministic output."""

    # Force LC community path to be unavailable, so we go through HTTP fallback
    monkeypatch.setattr(toolkit, "_HAS_LC_COMMUNITY", False, raising=False)
    # Monkeypatch wiki_search to avoid real HTTP
    monkeypatch.setattr(toolkit, "wiki_search", lambda **_: _fake_wiki_blocks())

    out = toolkit.wiki_retrieve(query=query, max_results=2, top_k=2, rerank_top_n=2)
    assert isinstance(out, str) and out.strip(), "Output must be non-empty text"

    # Basic structural checks
    assert out.startswith("contexts:"), "Should return contexts header"
    assert "\n---" in out, "Should contain block separators"

    # Should enumerate contexts
    assert re.search(r"^\[1\]\s", out, re.M), "First context numbered [1]"

    # Should contain source/title metadata surfaced into the header line
    assert ("Alpha" in out) or ("Beta" in out), "Should expose document titles"


print(toolkit.wiki_retrieve(query="Moon perigee distance kilometers", max_results=5))