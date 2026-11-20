"""Gradio helpers (no live UI)."""

from __future__ import annotations

from codebase_qa_agent.gradio_app import format_hits_markdown
from codebase_qa_agent.schemas import Chunk, RetrievalHit


def test_format_hits_markdown_empty():
    assert format_hits_markdown([]) == "_No hits._"


def test_format_hits_markdown_includes_citation():
    chunk = Chunk(
        chunk_id="a",
        path="libs/langgraph/langgraph/graph/state.py",
        start_line=1,
        end_line=10,
        symbol="StateGraph",
        chunk_type="code",
        text="class StateGraph",
    )
    md = format_hits_markdown([RetrievalHit(chunk=chunk, score=0.9, source="hybrid")])
    assert "state.py:1-10" in md
    assert "StateGraph" in md