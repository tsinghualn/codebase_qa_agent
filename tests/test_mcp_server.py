"""MCP helpers and read_file behavior (no live stdio transport)."""

from __future__ import annotations

from pathlib import Path

from codebase_qa_agent.corpus_read import read_line_range
from codebase_qa_agent.mcp_server import hit_to_dict
from codebase_qa_agent.schemas import Chunk, RetrievalHit


def test_hit_to_dict_includes_citation():
    chunk = Chunk(
        chunk_id="x",
        path="libs/langgraph/foo.py",
        start_line=10,
        end_line=20,
        symbol="bar",
        chunk_type="code",
        text="def bar(): pass",
    )
    payload = hit_to_dict(RetrievalHit(chunk=chunk, score=0.42, source="hybrid"))
    assert payload["citation"] == "libs/langgraph/foo.py:10-20"
    assert payload["score"] == 0.42


def test_read_line_range_rejects_traversal(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inside.txt").write_text("safe\n", encoding="utf-8")
    assert read_line_range(tmp_path, "ok.txt", 1, 2) == "line1\nline2"
    assert read_line_range(tmp_path, "../ok.txt", 1, 1) is None