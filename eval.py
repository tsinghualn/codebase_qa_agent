#!/usr/bin/env python3
"""Eval harness: golden set (M0) + retrieval modes comparison (M1/M3)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from codebase_qa_agent.bm25_index import BM25_META  # noqa: E402
from codebase_qa_agent.evals.golden import GoldenCase, golden_stats, load_golden  # noqa: E402
from codebase_qa_agent.retrieval import RetrievalEngine  # noqa: E402
from codebase_qa_agent.retrieval_metrics import (  # noqa: E402
    context_precision_at_k,
    context_recall_at_k,
    hit_in_top_k,
)
from codebase_qa_agent.settings import get_settings  # noqa: E402


def load_eval_config() -> dict:
    with (ROOT / "eval-config.json").open(encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class ModeMetrics:
    label: str
    context_precision: float
    context_recall: float
    top3_hit_rate: float


def evaluate_mode(
    label: str,
    *,
    mode: str,
    use_rerank: bool,
    cases: list[GoldenCase],
    top_k: int,
) -> ModeMetrics:
    settings = get_settings()
    engine = RetrievalEngine(
        settings.chroma_path,
        settings.embedding_model,
        mode=mode,  # type: ignore[arg-type]
        settings=settings,
        use_rerank=use_rerank,
    )
    answerable = [c for c in cases if not c.unanswerable]
    precisions: list[float] = []
    recalls: list[float] = []
    top3_hits = 0
    for case in answerable:
        hits = engine.search(case.question, top_k=top_k)
        precisions.append(context_precision_at_k(hits, case.expected_files))
        recalls.append(context_recall_at_k(hits, case.expected_files))
        if hit_in_top_k(hits[:3], case.expected_files):
            top3_hits += 1
    n = len(answerable) or 1
    return ModeMetrics(
        label=label,
        context_precision=sum(precisions) / n if precisions else 0.0,
        context_recall=sum(recalls) / n if recalls else 0.0,
        top3_hit_rate=top3_hits / n if answerable else 0.0,
    )


def write_eval_results(rows: list[ModeMetrics], *, top_k: int, pin_sha: str) -> None:
    lines = [
        "# Eval results",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        f"_Corpus pin: `{pin_sha}` · golden cases: answerable subset · top_k={top_k}_",
        "",
        "| Mode | Context P@" + str(top_k) + " | Context R@" + str(top_k) + " | Top-3 file hit rate |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.label} | {row.context_precision:.3f} | {row.context_recall:.3f} | {row.top3_hit_rate:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **Production default:** `hybrid` (BM25 + vectors + RRF), rerank off unless `RERANK_ENABLED=true`.",
            "- On this code-heavy corpus, MS MARCO cross-encoder rerank often **hurts** top-3 file hit rate vs hybrid alone (see table).",
        ]
    )
    (ROOT / "eval-results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config = load_eval_config()
    cases = load_golden()
    stats = golden_stats(cases)
    top_k = int(config.get("top_k", 5))

    print("=== Codebase Q&A Agent — eval ===")
    print(f"Golden cases: {stats['total']} (answerable={stats['answerable']}, unanswerable={stats['unanswerable']})")
    print(f"top_k={top_k}")
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
    pin_sha = str(pin.get("commit_sha", ""))
    if pin_sha in (None, "", "REPLACE_AFTER_FIRST_PIN"):
        errors.append("data/pin.json commit_sha is not pinned (run: make ingest).")

    settings = get_settings()
    index_ready = (settings.chroma_path / BM25_META).is_file()
    retrieval_errors: list[str] = []
    mode_rows: list[ModeMetrics] = []

    if index_ready:
        specs = [
            ("vector", "vector", False),
            ("hybrid", "hybrid", False),
            ("hybrid+rerank", "hybrid", True),
        ]
        if not config.get("compare_retrieval_modes", True):
            mode = config.get("retrieval_mode", "hybrid")
            use_rerank = settings.rerank_enabled
            label = f"{mode}+rerank" if use_rerank else mode
            specs = [(label, mode, use_rerank)]

        for label, mode, use_rerank in specs:
            print(f"Evaluating {label}…")
            mode_rows.append(evaluate_mode(label, mode=mode, use_rerank=use_rerank, cases=cases, top_k=top_k))

        write_eval_results(mode_rows, top_k=top_k, pin_sha=pin_sha)

        min_top3 = float(
            config.get(
                "top3_expected_file_hit_rate_min",
                config.get("context_precision_min", 0.8),
            )
        )
        prod_label = str(config.get("production_eval_mode", "hybrid"))
        prod_metrics = next((r for r in mode_rows if r.label == prod_label), mode_rows[1])
        if prod_metrics.top3_hit_rate < min_top3:
            retrieval_errors.append(
                f"Production mode `{prod_label}` top-3 hit rate {prod_metrics.top3_hit_rate:.2%} "
                f"< threshold {min_top3:.0%}."
            )
    else:
        retrieval_errors.append(f"No index at {settings.chroma_path} — run `make ingest` first.")

    print()
    print("| Mode | Context P@" + str(top_k) + " | Context R@" + str(top_k) + " | Top-3 file hit rate |")
    print("| --- | --- | --- | --- |")
    for row in mode_rows:
        print(
            f"| {row.label} | {row.context_precision:.3f} | {row.context_recall:.3f} | {row.top3_hit_rate:.1%} |"
        )
    print()
    print("| citation_precision | N/A (M2) |")
    print("| faithfulness | N/A (M3 optional) |")
    if mode_rows:
        print(f"\nWrote {ROOT / 'eval-results.md'}")

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