"""Gradio web UI: cited agent answers + retrieval-only search."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

# Avoid noisy tokenizers warnings when Gradio runs handlers in a thread pool.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from codebase_qa_agent.agent import answer_question
from codebase_qa_agent.corpus import load_pin
from codebase_qa_agent.schemas import RetrievalHit
from codebase_qa_agent.settings import Settings, get_settings

EXAMPLE_QUESTIONS = [
    "How does LangGraph persist graph state across steps (checkpointing)?",
    "How do you stream graph execution outputs?",
    "How are conditional edges defined on a StateGraph?",
]

LOADING_SEARCH = (
    "⏳ **Running hybrid search…**  \n"
    "First request can take **15–45s** while embedding models load."
)
LOADING_ASK = (
    "⏳ **Running agent…**  \n"
    "Retrieval + LLM draft usually **20–60s** on the first click (model cold start)."
)


def index_ready(chroma_path: Path) -> bool:
    return chroma_path.is_dir() and (chroma_path / "bm25_meta.json").is_file()


def index_status_message(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    pin = load_pin(settings.project_root)
    sha = str(pin.get("commit_sha") or "unknown")[:12]
    chroma = settings.chroma_path
    if index_ready(chroma):
        return f"Index ready at `{chroma}` · pin `{sha}…`"
    return (
        f"**Index missing** — run `make ingest` locally (expected `{chroma}`). "
        f"Pin commit: `{sha}…`."
    )


def format_hits_markdown(hits: list[RetrievalHit]) -> str:
    if not hits:
        return "_No hits._"
    lines: list[str] = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        cite = f"`{c.path}:{c.start_line}-{c.end_line}`"
        lines.append(f"**{i}.** {cite} · _{c.symbol or '—'}_ · {hit.source} ({hit.score:.4f})")
        preview = c.text.strip().replace("\n", " ")[:280]
        lines.append(f"> {preview}…" if len(c.text) > 280 else f"> {preview}")
        lines.append("")
    return "\n".join(lines).strip()


def run_retrieval(question: str, top_k: int | float = 5) -> str:
    settings = get_settings()
    if not index_ready(settings.chroma_path):
        return index_status_message(settings)
    from codebase_qa_agent.retrieval import RetrievalEngine

    top_k = max(1, min(int(top_k), 15))
    engine = RetrievalEngine(
        settings.chroma_path,
        settings.embedding_model,
        mode=settings.retrieval_mode,  # type: ignore[arg-type]
        settings=settings,
    )
    hits = engine.search(question.strip(), top_k=top_k)
    header = index_status_message(settings)
    return f"{header}\n\n---\n\n{format_hits_markdown(hits)}"


def run_retrieval_stream(question: str, top_k: int | float = 5) -> Iterator[str]:
    yield LOADING_SEARCH
    try:
        yield run_retrieval(question, top_k)
    except Exception as err:
        yield f"**Error:** `{type(err).__name__}` — {err}"


def run_agent(question: str) -> str:
    q = question.strip()
    if not q:
        return "_Enter a question._"
    settings = get_settings()
    if not index_ready(settings.chroma_path):
        return index_status_message(settings)
    return answer_question(q, settings=settings)


def run_agent_stream(question: str) -> Iterator[str]:
    q = question.strip()
    if not q:
        yield "_Enter a question._"
        return
    yield LOADING_ASK
    try:
        yield run_agent(question)
    except Exception as err:
        yield f"**Error:** `{type(err).__name__}` — {err}"


def _gradio_major() -> int:
    import gradio as gr

    return int(gr.__version__.split(".", maxsplit=1)[0])


def create_demo():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "gradio is not installed. Run: uv pip install -e '.[dev,ml,ui]'"
        ) from exc

    settings = get_settings()
    pin_banner = index_status_message(settings)
    theme = gr.themes.Soft()
    block_kw: dict = {"title": "Codebase Q&A — LangGraph (pinned)"}
    if _gradio_major() < 6:
        block_kw["theme"] = theme

    with gr.Blocks(**block_kw) as demo:
        gr.Markdown(
            "# Codebase Q&A Agent\n"
            "Unofficial demo over a **pinned** `langchain-ai/langgraph` snapshot. "
            "Answers include **file:line** citations validated against retrieved chunks.\n\n"
            f"{pin_banner}"
        )
        with gr.Tabs():
            with gr.Tab("Ask (agent + citations)"):
                ask_q = gr.Textbox(
                    label="Question",
                    placeholder="How does checkpointing persist state?",
                    lines=2,
                )
                ask_btn = gr.Button("Ask", variant="primary")
                ask_out = gr.Markdown(value="_Results appear here — allow up to a minute on first run._")
                gr.Markdown("**Example questions** (click to fill, then press **Ask**):")
                with gr.Row():
                    for prompt in EXAMPLE_QUESTIONS:
                        short = prompt if len(prompt) <= 52 else prompt[:49] + "…"
                        ex = gr.Button(short, size="sm")
                        ex.click(lambda p=prompt: p, outputs=ask_q)

                click_kw = dict(show_progress="full")
                ask_btn.click(run_agent_stream, inputs=ask_q, outputs=ask_out, **click_kw)
                ask_q.submit(run_agent_stream, inputs=ask_q, outputs=ask_out, **click_kw)

            with gr.Tab("Search (retrieval only)"):
                gr.Markdown("Hybrid BM25 + vectors — no LLM cost.")
                search_q = gr.Textbox(
                    label="Query",
                    lines=2,
                    placeholder="checkpointing",
                )
                top_k = gr.Slider(1, 15, value=5, step=1, label="Top K")
                search_btn = gr.Button("Search", variant="secondary")
                search_out = gr.Markdown(value="_Results appear here after Search._")
                search_btn.click(
                    run_retrieval_stream,
                    inputs=[search_q, top_k],
                    outputs=search_out,
                    **click_kw,
                )
                search_q.submit(
                    run_retrieval_stream,
                    inputs=[search_q, top_k],
                    outputs=search_out,
                    **click_kw,
                )

        gr.Markdown(
            "_Keys stay on the server (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in env). "
            "Never paste secrets in the chat._"
        )
    demo.queue(default_concurrency_limit=2)
    return demo


def main() -> None:
    if os.getenv("SPACE_ID") or os.getenv("SYSTEM") == "spaces":
        host = "0.0.0.0"
    else:
        host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "").lower() in ("1", "true", "yes")

    import gradio as gr

    launch_kw = dict(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
    )
    if _gradio_major() >= 6:
        launch_kw["theme"] = gr.themes.Soft()
    create_demo().launch(**launch_kw)


if __name__ == "__main__":
    main()