"""Record which embedding model built the Chroma index (query should match ingest)."""

from __future__ import annotations

import json
from pathlib import Path

META_FILE = "index_meta.json"


def write_index_meta(chroma_path: Path, *, embedding_model: str) -> None:
    chroma_path.mkdir(parents=True, exist_ok=True)
    with (chroma_path / META_FILE).open("w", encoding="utf-8") as f:
        json.dump({"embedding_model": embedding_model}, f, indent=2)
        f.write("\n")


def read_embedding_model(chroma_path: Path, default: str) -> str:
    path = chroma_path / META_FILE
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return str(data.get("embedding_model") or default)