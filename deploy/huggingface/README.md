# Deploy on Hugging Face Spaces

Host the Gradio UI as a **Gradio Space** (SDK). The Space runs `app.py` at the repo root.

## Prerequisites

- This repository (or a mirror) as the Space git repo
- **Secrets** (Space → Settings → Secrets): `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`, `LLM_PROVIDER`
- **Build artifacts**: `data/chroma/` + `data/corpus/` on the Space disk (not in git)

## Option A — Persistent storage + one-time ingest (recommended)

1. Create a **Gradio** Space (Python 3.12).
2. Push this project; ensure root `app.py` and `pyproject.toml` are present.
3. Add a **persistent volume** (if available) mounted at `/data` and set:
   - `CHROMA_PATH=/data/chroma`
   - Clone corpus to `/data/corpus` or set ingest output there (adjust `data/pin.json` paths only if you customize ingest).
4. In a **one-off terminal** on the Space:
   ```bash
   pip install uv && uv venv .venv && source .venv/bin/activate
   uv pip install -e ".[ml,ui]"
   python -m codebase_qa_agent.ingest --write-pin
   ```
5. Restart the Space. `app.py` should show **Index ready**.

Ingest is CPU-heavy (embeddings); run it once and keep the volume.

## Option B — Dockerfile Space

Use a custom `Dockerfile` that:

1. `uv pip install -e ".[ml,ui]"`
2. Runs `python -m codebase_qa_agent.ingest` at **image build** time (slow builds, but no manual step).

Pin the LangGraph commit in `data/pin.json` before building so demos are reproducible.

## Space README frontmatter

If the Space uses a separate card README, include:

```yaml
---
title: Codebase Q&A Agent
emoji: 📎
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.23.0"
app_file: app.py
pinned: false
license: mit
---
```

## Environment

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Agent tab (server-side only) |
| `CHROMA_PATH` | Default `data/chroma` |
| `RETRIEVAL_MODE` | `hybrid` (default) |
| `GRADIO_SERVER_NAME` | Set `0.0.0.0` on Spaces (Gradio often sets this automatically) |

The **Search** tab works without an LLM key; **Ask** requires one.

## Security

- Do not commit `.env` or API keys.
- Do not commit `data/corpus/` or `data/chroma/` (large + license).
- Users of the public Space only see answers — not your secrets.