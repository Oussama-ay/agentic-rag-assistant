# RAGent — Agentic Document Assistant

A learning project built step by step toward answering questions about documents
using retrieval augmented generation (RAG), agent tools, and source citations.

## Current status

Milestone 0: Python project setup and a working FastAPI health endpoint.
Document processing, retrieval, and agents are planned and are not implemented yet.

## Requirements

- Python 3.12 (the version used for this milestone)
- Git

## Run locally

```bash
git clone git@github.com:Oussama-ay/agentic-rag-assistant.git
cd agentic-rag-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

No API key or `.env` file is required for this milestone. `.env.example` holds
placeholders for future integrations; the app does not load it yet.

Open the interactive API documentation at http://127.0.0.1:8000/docs.

In another terminal, check the application:

```bash
curl -i http://127.0.0.1:8000/health
```

Expected: HTTP `200 OK` with this JSON body:

```json
{"status":"ok"}
```

This endpoint confirms the API is responding. It does not check any model,
database, or retrieval service.

## Project structure

```text
app/
  __init__.py       Python package marker
  main.py           FastAPI application and health route
data/
  .gitkeep          Keeps the empty data directory in Git
.env.example       Future configuration placeholders, without secrets
.gitignore         Excludes local environments, secrets, and uploaded data
requirements.txt   Direct application dependencies
```

Additional modules will be introduced as their milestones need them.

## Dependencies

- **FastAPI** defines HTTP endpoints and generates interactive API documentation.
- **Uvicorn** runs the application and listens for HTTP requests.

Their supporting packages are installed automatically by pip. This setup uses
Python's built-in `venv` to keep project packages separate from system Python.
See the [FastAPI environment guide](https://fastapi.tiangolo.com/virtual-environments/).

## Roadmap

- [x] Project setup and health endpoint
- [ ] Document loading and chunking
- [ ] Embeddings and vector storage
- [ ] Retrieval
- [ ] Grounded answers with sources
- [ ] Retrieval tools and agent orchestration
- [ ] Subagent analysis
- [ ] Document upload and question API endpoints
- [ ] Automated tests
- [ ] Docker packaging
- [ ] Architecture documentation and a reproducible demo
- [ ] Optional frontend
