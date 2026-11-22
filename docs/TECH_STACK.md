# Egyptian Legal AI — Technology Stack

## Development Environment

| Component | Selection |
|---|---|
| OS | Ubuntu 24.04 LTS |
| Python | 3.11 |
| RAM | 32 GB |
| GPU | NVIDIA Quadro P2000 Mobile |
| VRAM | 4 GB |
| Development | VS Code / Terminal |
| Environment | Python virtual environment |

## Backend

| Technology | Purpose |
|---|---|
| FastAPI | REST API |
| Uvicorn | ASGI server |
| Pydantic | Data validation |
| Pydantic Settings | Configuration |
| SQLAlchemy | Database ORM |
| Alembic | Database migrations |

## Frontend

| Technology | Purpose |
|---|---|
| Streamlit | User interface |
| Requests / HTTPX | Backend communication |

## Document Processing

| Technology | Purpose |
|---|---|
| PyMuPDF | PDF extraction |
| python-docx | DOCX parsing |
| Pandas | Tabular data |
| BeautifulSoup | HTML parsing |
| PaddleOCR | OCR, subject to hardware and language validation |

## Retrieval

| Technology | Purpose |
|---|---|
| Sentence Transformers | Embeddings |
| Qdrant | Vector database |
| BM25 | Sparse retrieval |
| Cross-Encoder | Reranking |
| Reciprocal Rank Fusion | Hybrid score fusion |

## LLM Integration

| Technology | Purpose |
|---|---|
| Groq API | Primary generation provider |
| Hugging Face Transformers | Local fallback |
| OpenAI API | Optional provider |
| Gemini API | Optional provider |

## Database and Caching

| Technology | Purpose |
|---|---|
| PostgreSQL | Structured persistent data |
| Qdrant | Vector storage |
| Redis | Optional caching and job support |

## Testing and Quality

| Technology | Purpose |
|---|---|
| Pytest | Automated tests |
| Ruff | Linting and formatting |
| MyPy | Static type checking |
| Coverage | Test coverage |

## Deployment

| Technology | Purpose |
|---|---|
| Docker | Containerization |
| Docker Compose | Multi-service orchestration |
| Nginx | Reverse proxy |

## Hardware Considerations

The development machine has 4 GB of dedicated GPU memory.

Therefore:
- Avoid assuming large models fit in GPU memory.
- Support CPU inference.
- Use configurable batch sizes.
- Avoid loading multiple large models simultaneously.
- Monitor memory consumption.
- Consider smaller quantized local models.
- Benchmark models before selecting production defaults.

## Selection Policy

The listed technologies are proposed defaults, not mandatory dependencies.

Before adding a dependency:
1. Confirm that it solves a real requirement.
2. Check Python compatibility.
3. Check Ubuntu compatibility.
4. Consider installation size and GPU requirements.
5. Avoid introducing duplicate libraries for the same responsibility.

Actual package versions must be pinned and tested in `pyproject.toml`.