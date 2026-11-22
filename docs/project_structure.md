# Egyptian Legal AI — Project Structure

## Project Overview

Egyptian Legal AI is an Arabic-first legal Retrieval-Augmented Generation (RAG) system designed to retrieve, analyze, and explain Egyptian legal documents using authoritative sources.

## Directory Structure

```text
Egyptian-Legal-AI/
│
├── src/
│   ├── __init__.py
│   ├── config/
│   ├── domain/
│   ├── ingestion/
│   │   ├── parsers/
│   │   └── ocr/
│   ├── preprocessing/
│   ├── chunking/
│   ├── embeddings/
│   ├── vector_store/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   ├── llm/
│   │   ├── providers/
│   │   └── monitoring/
│   ├── rag/
│   ├── storage/
│   │   ├── database/
│   │   │   ├── models/
│   │   │   ├── repositories/
│   │   │   └── migrations/
│   │   ├── document_store/
│   │   └── cache/
│   ├── services/
│   ├── security/
│   ├── evaluation/
│   ├── monitoring/
│   └── utils/
│
├── api/
│   ├── main.py
│   ├── lifespan.py
│   ├── dependencies.py
│   ├── exceptions.py
│   ├── schemas/
│   ├── middleware/
│   └── routes/
│
├── frontend/
│   ├── app.py
│   ├── components/
│   ├── pages/
│   └── utils/
│
├── workers/
│   └── ingestion_worker.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── api/
│   └── evaluation/
│
├── notebooks/
│
├── data/
│   ├── raw/
│   │   ├── official_laws/
│   │   └── user_uploads/
│   ├── processed/
│   ├── evaluation/
│   └── temp/
│
├── artifacts/
│   ├── embeddings/
│   ├── models/
│   └── indexes/
│
├── deployment/
│   ├── docker/
│   └── nginx/
│
├── logs/
│
├── docs/
├── .env
├── .env.example
├── .gitignore
├── .dockerignore
├── pyproject.toml
├── Makefile
├── docker-compose.yml
├── README.md
└── LICENSE
```

## Architectural Rules

1. `src/` contains core application logic only.
2. `api/` contains FastAPI application code and HTTP interfaces.
3. `frontend/` contains Streamlit presentation code.
4. `workers/` contains asynchronous/background processing.
5. `tests/` contains automated tests.
6. `data/` contains datasets and documents.
7. `artifacts/` contains generated models, embeddings, and indexes.
8. `deployment/` contains container and deployment configurations.
9. Avoid duplicated implementations across modules.
10. Keep business logic independent of the API and frontend.

## Import Convention

Core modules should be imported through the `src` package.

Example:

```python
from src.config.settings import settings
from src.services.chat_service import ChatService
from src.rag.pipeline import RAGPipeline
```

API and frontend should communicate with core services through clear interfaces. The Streamlit frontend should communicate with FastAPI over HTTP rather than directly accessing internal RAG modules.