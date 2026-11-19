"""Programmatic citation validation (gate before user-visible answers)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codebase_qa_agent.answer_schemas import AnswerDraft, Citation, Claim
from codebase_qa_agent.corpus_read import read_line_range, resolve_corpus_file
from codebase_qa_agent.schemas import Chunk


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid_claims: list[Claim]
    dropped_claims: list[Claim]
    ok: bool


def _line_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or b_end < a_start)


def _path_match(chunk_path: str, cite_path: str) -> bool:
    return chunk_path == cite_path or chunk_path.endswith(cite_path) or cite_path in chunk_path


def citation_in_retrieved(cite: Citation, chunks: list[Chunk]) -> bool:
    for chunk in chunks:
        if not _path_match(chunk.path, cite.file):
            continue
        if _line_overlap(chunk.start_line, chunk.end_line, cite.start_line, cite.end_line):
            return True
    return False


def citation_on_disk(corpus_dir: Path, cite: Citation) -> bool:
    if resolve_corpus_file(corpus_dir, cite.file) is None:
        return False
    return read_line_range(corpus_dir, cite.file, cite.start_line, cite.end_line) is not None


def validate_claim(claim: Claim, chunks: list[Chunk], corpus_dir: Path) -> bool:
    if not claim.citations:
        return False
    for cite in claim.citations:
        if not citation_on_disk(corpus_dir, cite):
            return False
        if not citation_in_retrieved(cite, chunks):
            return False
    return True


def validate_draft(draft: AnswerDraft, chunks: list[Chunk], corpus_dir: Path) -> ValidationResult:
    valid: list[Claim] = []
    dropped: list[Claim] = []
    for claim in draft.claims:
        if validate_claim(claim, chunks, corpus_dir):
            valid.append(claim)
        else:
            dropped.append(claim)
    return ValidationResult(valid_claims=valid, dropped_claims=dropped, ok=bool(valid))


def format_markdown(valid_claims: list[Claim]) -> str:
    parts: list[str] = []
    for claim in valid_claims:
        cites = ", ".join(
            f"[{c.file}:{c.start_line}-{c.end_line}]" for c in claim.citations
        )
        parts.append(f"- {claim.text} {cites}")
    return "\n".join(parts)