"""Safe reads from the pinned corpus on disk."""

from __future__ import annotations

from pathlib import Path


def corpus_root(project_root: Path) -> Path:
    return project_root / "data" / "corpus"


def resolve_corpus_file(corpus_dir: Path, relative_path: str) -> Path | None:
    relative_path = relative_path.lstrip("/")
    if ".." in Path(relative_path).parts:
        return None
    root = corpus_dir.resolve()
    target = (corpus_dir / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    if not target.is_file():
        return None
    return target


def read_line_range(
    corpus_dir: Path,
    relative_path: str,
    start_line: int,
    end_line: int,
    *,
    max_lines: int = 250,
) -> str | None:
    path = resolve_corpus_file(corpus_dir, relative_path)
    if path is None:
        return None
    if end_line < start_line:
        return None
    if end_line - start_line + 1 > max_lines:
        end_line = start_line + max_lines - 1
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if start_line < 1 or start_line > len(lines):
        return None
    end_line = min(end_line, len(lines))
    return "\n".join(lines[start_line - 1 : end_line])