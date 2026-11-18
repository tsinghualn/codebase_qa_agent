"""Retrieval eval helpers (no heavy ML imports)."""

from __future__ import annotations

from codebase_qa_agent.schemas import RetrievalHit


def path_matches_expected(chunk_path: str, expected_files: tuple[str, ...]) -> bool:
    for expected in expected_files:
        exp = expected.rstrip("/")
        if chunk_path == exp or chunk_path.startswith(exp + "/") or exp in chunk_path:
            return True
    return False


def context_precision_at_k(hits: list[RetrievalHit], expected_files: tuple[str, ...]) -> float:
    if not expected_files or not hits:
        return 0.0
    relevant = sum(1 for h in hits if path_matches_expected(h.chunk.path, expected_files))
    return relevant / len(hits)


def context_recall_at_k(hits: list[RetrievalHit], expected_files: tuple[str, ...]) -> float:
    if not expected_files:
        return 1.0
    hit_paths = {h.chunk.path for h in hits}
    found = 0
    for expected in expected_files:
        exp = expected.rstrip("/")
        if any(p == exp or p.startswith(exp + "/") or exp in p for p in hit_paths):
            found += 1
    return found / len(expected_files)


def hit_in_top_k(hits: list[RetrievalHit], expected_files: tuple[str, ...]) -> bool:
    return any(path_matches_expected(h.chunk.path, expected_files) for h in hits)