"""AST-aware Python chunking and Markdown chunking by heading."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from codebase_qa_agent.schemas import Chunk, ChunkType

LONG_FUNCTION_LINES = 80
LINE_WINDOW = 60
LINE_OVERLAP = 10
MAX_CLASS_LINES = 120


def _chunk_id(path: str, start: int, end: int, symbol: str) -> str:
    raw = f"{path}:{start}:{end}:{symbol}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _split_line_windows(lines: list[str], path: str, symbol: str, start_base: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    i = 0
    while i < len(lines):
        window = lines[i : i + LINE_WINDOW]
        start = start_base + i
        end = start_base + i + len(window) - 1
        text = "".join(window)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(path, start, end, f"{symbol}#L{i}"),
                path=path,
                start_line=start,
                end_line=end,
                symbol=symbol,
                chunk_type="code",
                text=text,
            )
        )
        if i + LINE_WINDOW >= len(lines):
            break
        i += LINE_WINDOW - LINE_OVERLAP
    return chunks


def chunk_python_file(relative_path: str, source: str) -> list[Chunk]:
    lines = source.splitlines(keepends=True)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _split_line_windows(lines, relative_path, "<module>", 1)

    chunks: list[Chunk] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno
        end = node.end_lineno or start
        body_lines = lines[start - 1 : end]
        symbol = node.name
        if len(body_lines) <= LONG_FUNCTION_LINES:
            text = "".join(body_lines)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(relative_path, start, end, symbol),
                    path=relative_path,
                    start_line=start,
                    end_line=end,
                    symbol=symbol,
                    chunk_type="code",
                    text=text,
                )
            )
        else:
            chunks.extend(_split_line_windows(body_lines, relative_path, symbol, start))

    if not chunks and lines:
        return _split_line_windows(lines, relative_path, "<module>", 1)
    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _chunk_markdown_langchain(relative_path: str, source: str) -> list[Chunk] | None:
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
    except ImportError:
        return None

    headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)
    lc_docs = splitter.split_text(source)
    if not lc_docs:
        return None

    chunks: list[Chunk] = []
    cursor = 0
    for doc in lc_docs:
        text = doc.page_content
        idx = source.find(text, cursor)
        if idx < 0:
            idx = source.find(text)
        if idx < 0:
            continue
        cursor = idx + len(text)
        start_line = source[:idx].count("\n") + 1
        end_line = start_line + max(text.count("\n"), 0)
        meta = doc.metadata or {}
        symbol = meta.get("h4") or meta.get("h3") or meta.get("h2") or meta.get("h1") or "doc"
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(relative_path, start_line, end_line, str(symbol)),
                path=relative_path,
                start_line=start_line,
                end_line=end_line,
                symbol=str(symbol),
                chunk_type="doc",
                text=text,
            )
        )
    return chunks or None


def chunk_markdown_file(relative_path: str, source: str) -> list[Chunk]:
    lc_chunks = _chunk_markdown_langchain(relative_path, source)
    if lc_chunks:
        return lc_chunks

    matches = list(_HEADING_RE.finditer(source))
    if not matches:
        return [
            Chunk(
                chunk_id=_chunk_id(relative_path, 1, max(1, source.count("\n") + 1), "doc"),
                path=relative_path,
                start_line=1,
                end_line=max(1, source.count("\n") + 1),
                symbol="doc",
                chunk_type="doc",
                text=source,
            )
        ]

    chunks: list[Chunk] = []
    line_offsets = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))

    for idx, match in enumerate(matches):
        start_char = match.start()
        end_char = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
        text = source[start_char:end_char]
        start_line = source[:start_char].count("\n") + 1
        end_line = source[:end_char].count("\n") + 1
        title = match.group(2).strip()
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(relative_path, start_line, end_line, title),
                path=relative_path,
                start_line=start_line,
                end_line=end_line,
                symbol=title,
                chunk_type="doc",
                text=text,
            )
        )
    return chunks


def chunk_file(relative_path: str, full_path: Path) -> list[Chunk]:
    text = full_path.read_text(encoding="utf-8", errors="replace")
    if relative_path.endswith(".py"):
        return chunk_python_file(relative_path, text)
    if relative_path.endswith((".md", ".mdx")):
        return chunk_markdown_file(relative_path, text)
    return []