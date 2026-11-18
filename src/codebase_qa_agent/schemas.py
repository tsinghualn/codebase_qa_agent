"""Shared data structures for chunks and retrieval hits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ChunkType = Literal["code", "doc"]


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    chunk_type: ChunkType
    text: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "chunk_type": self.chunk_type,
        }

    @classmethod
    def from_metadata(cls, text: str, chunk_id: str, meta: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=chunk_id,
            path=str(meta["path"]),
            start_line=int(meta["start_line"]),
            end_line=int(meta["end_line"]),
            symbol=str(meta.get("symbol") or ""),
            chunk_type=meta.get("chunk_type", "code"),  # type: ignore[arg-type]
            text=text,
        )


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    source: str