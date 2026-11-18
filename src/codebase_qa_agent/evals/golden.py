"""Load and validate the hand-written golden Q&A set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GoldenCase:
    id: str
    question: str
    expected_files: tuple[str, ...]
    reference_answer: str
    unanswerable: bool
    tags: tuple[str, ...]


def _parse_line(raw: dict[str, Any]) -> GoldenCase:
    return GoldenCase(
        id=raw["id"],
        question=raw["question"],
        expected_files=tuple(raw.get("expected_files") or []),
        reference_answer=raw["reference_answer"],
        unanswerable=bool(raw.get("unanswerable", False)),
        tags=tuple(raw.get("tags") or []),
    )


def load_golden(path: Path | None = None) -> list[GoldenCase]:
    path = path or Path(__file__).resolve().parents[3] / "evals" / "golden.jsonl"
    cases: list[GoldenCase] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(_parse_line(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as err:
                raise ValueError(f"Invalid golden case at {path}:{lineno}") from err
    return cases


def golden_stats(cases: list[GoldenCase]) -> dict[str, int]:
    unanswerable = sum(1 for c in cases if c.unanswerable)
    return {
        "total": len(cases),
        "answerable": len(cases) - unanswerable,
        "unanswerable": unanswerable,
    }