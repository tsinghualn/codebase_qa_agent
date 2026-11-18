"""BM25 lexical index persisted alongside Chroma."""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from codebase_qa_agent.schemas import Chunk

BM25_META = "bm25_meta.json"


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._chunks[i], float(s)) for i, s in ranked if s > 0]

    def save(self, chroma_path: Path) -> None:
        chroma_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "path": c.path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "symbol": c.symbol,
                    "chunk_type": c.chunk_type,
                    "text": c.text,
                }
                for c in self._chunks
            ]
        }
        path = chroma_path / BM25_META
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, chroma_path: Path) -> BM25Index:
        path = chroma_path / BM25_META
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                path=item["path"],
                start_line=item["start_line"],
                end_line=item["end_line"],
                symbol=item.get("symbol", ""),
                chunk_type=item.get("chunk_type", "code"),
                text=item["text"],
            )
            for item in data["chunks"]
        ]
        return cls(chunks)