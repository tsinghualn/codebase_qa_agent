"""FastMCP server: search and read the pinned LangGraph corpus (stdio for Cursor)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from codebase_qa_agent.corpus_read import corpus_root, read_line_range
from codebase_qa_agent.schemas import RetrievalHit
from codebase_qa_agent.settings import get_settings

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "fastmcp is not installed. Run: uv pip install -e '.[dev,ml]'"
    ) from exc

mcp = FastMCP(
    "langgraph-codebase",
    instructions=(
        "Search and read a pinned LangGraph repository (libs/langgraph + docs). "
        "Use search_code for implementation questions, search_docs for guides, "
        "and read_file with path and line range for exact citations."
    ),
)

TEXT_PREVIEW_CHARS = 400


def hit_to_dict(hit: RetrievalHit, *, include_text: bool = True) -> dict[str, Any]:
    c = hit.chunk
    out: dict[str, Any] = {
        "path": c.path,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "symbol": c.symbol,
        "chunk_type": c.chunk_type,
        "score": round(hit.score, 6),
        "source": hit.source,
        "citation": f"{c.path}:{c.start_line}-{c.end_line}",
    }
    if include_text:
        text = c.text.strip()
        if len(text) > TEXT_PREVIEW_CHARS:
            text = text[:TEXT_PREVIEW_CHARS] + "…"
        out["text"] = text
    return out


def hits_payload(hits: list[RetrievalHit]) -> dict[str, Any]:
    return {"count": len(hits), "results": [hit_to_dict(h) for h in hits]}


@lru_cache(maxsize=1)
def get_engine():
    from codebase_qa_agent.retrieval import RetrievalEngine

    settings = get_settings()
    mode = "hybrid" if settings.retrieval_mode == "hybrid" else "vector"
    return RetrievalEngine(
        settings.chroma_path,
        settings.embedding_model,
        mode=mode,
        settings=settings,
    )


def _project_paths() -> tuple[Path, Path]:
    settings = get_settings()
    return settings.project_root, corpus_root(settings.project_root)


@mcp.tool
def search_code(query: str, top_k: int = 5) -> dict[str, Any]:
    """Hybrid search over Python source in the pinned corpus (BM25 + vectors)."""
    top_k = max(1, min(top_k, 20))
    hits = get_engine().search(query, top_k=top_k, chunk_type="code")
    return hits_payload(hits)


@mcp.tool
def search_docs(query: str, top_k: int = 5) -> dict[str, Any]:
    """Hybrid search over documentation markdown in the pinned corpus."""
    top_k = max(1, min(top_k, 20))
    hits = get_engine().search(query, top_k=top_k, chunk_type="doc")
    return hits_payload(hits)


@mcp.tool
def read_file(path: str, start_line: int, end_line: int) -> dict[str, Any]:
    """Read a line range from a file under the pinned corpus (paths are repo-relative)."""
    _, corpus_dir = _project_paths()
    content = read_line_range(corpus_dir, path, start_line, end_line)
    if content is None:
        return {
            "ok": False,
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "error": "File not found, path escapes corpus, or invalid line range.",
        }
    return {
        "ok": True,
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "citation": f"{path}:{start_line}-{end_line}",
        "content": content,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()