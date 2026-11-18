#!/usr/bin/env python3
"""Eval harness: golden-set checks (M0) + retrieval metrics (M1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from codebase_qa_agent.bm25_index import BM25_META  # noqa: E402
from codebase_qa_agent.evals.golden import golden_stats, load_golden  # noqa: E402
from codebase_qa_agent.retrieval import RetrievalEngine  # noqa: E402
from codebase_qa_agent.retrieval_metrics import (  # noqa: E402
    context_precision_at_k,
    context_recall_at_k,
    hit_in_top_k,
)
from codebase_qa_agent.settings import get_settings  # noqa: E402


def load_eval_config() -> dict:
    path = ROOT / "eval-config.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    config = load_eval_config()
    cases = load_golden()
    stats = golden_stats(cases)
    top_k = int(config.get("top_k", 5))
    mode = config.get("retrieval_mode", "hybrid")

    print("=== Codebase Q&A Agent — eval ===")
    print(f"Golden cases: {stats['total']} (answerable={stats['answerable']}, unanswerable={stats['unanswerable']})")
    print(f"top_k={top_k}, retrieval_mode={mode}")
    print()

    errors: list[str] = []
    min_count = int(config.get("require_golden_count_min", 15))
    if stats["total"] < min_count:
        errors.append(f"Need at least {min_count} golden cases, found {stats['total']}.")
    if stats["unanswerable"] < 2:
        errors.append("Need at least 2 unanswerable cases.")

    verify_markers = sum(1 for c in cases if c.reference_answer.startswith("VERIFY:"))
    if verify_markers:
        errors.append(f"{verify_markers} golden answers still use VERIFY: placeholders.")

    pin_path = ROOT / "data" / "pin.json"
    with pin_path.open(encoding="utf-8") as f:
        pin = json.load(f)
    if pin.get("commit_sha") in (None, "", "REPLACE_AFTER_FIRST_PIN"):
        errors.append("data/pin.json commit_sha is not pinned (run: make ingest).")

    settings = get_settings()
    index_ready = (settings.chroma_path / BM25_META).is_file()
    retrieval_errors: list[str] = []
    avg_precision = avg_recall = top3_rate = 0.0
    answerable = [c for c in cases if not c.unanswerable]

    if index_ready:
        engine = RetrievalEngine(settings.chroma_path, settings.embedding_model, mode=mode)  # type: ignore[arg-type]
        precisions: list[float] = []
        recalls: list[float] = []
        top3_hits = 0
        for case in answerable:
            hits = engine.search(case.question, top_k=top_k)
            precisions.append(context_precision_at_k(hits, case.expected_files))
            recalls.append(context_recall_at_k(hits, case.expected_files))
            if hit_in_top_k(hits[:3], case.expected_files):
                top3_hits += 1
        if answerable:
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
            top3_rate = top3_hits / len(answerable)
        min_prec = float(config.get("context_precision_min", 0.8))
        if top3_rate < min_prec:
            retrieval_errors.append(
                f"top-3 file hit rate {top3_rate:.2%} < threshold {min_prec:.0%} "
                "(golden expected_files may need tuning at this pin)."
            )
    else:
        retrieval_errors.append(
            f"No index at {settings.chroma_path} — run `make ingest` first."
        )

    print("| metric | value |")
    print("| --- | --- |")
    if index_ready and answerable:
        print(f"| context_precision@{top_k} | {avg_precision:.3f} |")
        print(f"| context_recall@{top_k} | {avg_recall:.3f} |")
        print(f"| top-3 expected-file hit rate | {top3_rate:.1%} |")
    else:
        print(f"| context_precision@{top_k} | N/A |")
        print(f"| context_recall@{top_k} | N/A |")
        print("| top-3 expected-file hit rate | N/A |")
    print("| citation_precision | N/A (M2) |")
    print("| faithfulness | N/A (M2/M3) |")

    exit_code = 0
    if errors or retrieval_errors:
        print("\nIssues:")
        for err in errors + retrieval_errors:
            print(f"  - {err}")
        exit_code = 1
    else:
        print("\nAll checks passed.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())