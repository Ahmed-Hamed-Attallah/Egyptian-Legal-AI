# Egyptian Legal AI — System Architecture

## 1. Overview

Egyptian Legal AI is a modular, Arabic-first RAG application designed to answer questions using Egyptian legal documents.

The architecture separates the user interface, API, core intelligence, storage, and model providers.

## 2. High-Level Architecture

```text
                USER
                  |
                  v
          STREAMLIT FRONTEND
                  |
             HTTP / REST
                  |
                  v
             FASTAPI
                  |
                  v
           APPLICATION SERVICES
                  |
                  v
             RAG PIPELINE
                  |
       +----------+-----------+
       |                      |
       v                      v
    RETRIEVAL              GENERATION
       |                      |
       v                      v
 DENSE + SPARSE         LLM PROVIDERS
       |                      |
       v                      v
     QDRANT             API / LOCAL
       |
       v
  RERANKING
       |
       v
  LEGAL CONTEXT
       |
       v
 CITATION VALIDATION
       |
       v
   FINAL ANSWER
```

## 3. Main Components

### Frontend

Technology: Streamlit.

Responsibilities:
- Chat interface.
- Document upload.
- Search interface.
- Display citations.
- Conversation management.
- API communication.

### Backend API

Technology: FastAPI.

Responsibilities:
- HTTP endpoints.
- Request validation.
- Authentication and authorization.
- Dependency injection.
- Error handling.
- Communication with application services.

### Core Application

Location: `src/`.

Responsibilities:
- Document ingestion.
- Arabic preprocessing.
- Legal structure extraction.
- Chunking.
- Embedding.
- Retrieval.
- Reranking.
- LLM integration.
- RAG orchestration.
- Citation validation.

### Storage

PostgreSQL:
- Users.
- Conversations.
- Document metadata.
- Application records.

Qdrant:
- Document embeddings.
- Chunk vectors.
- Retrieval metadata.

Redis:
- Optional caching and background job support.

### LLM Layer

Primary generation:
- API-based provider.

Fallback generation:
- Local Hugging Face Transformers model.

The application should use a common provider interface.

## 4. Document Ingestion Flow

```text
Document Upload
      |
      v
File Validation
      |
      v
Format Detection
      |
      v
Document Parser / OCR
      |
      v
Arabic Normalization
      |
      v
Legal Structure Extraction
      |
      v
Metadata Extraction
      |
      v
Legal-Aware Chunking
      |
      v
Embedding Generation
      |
      v
Qdrant Storage
```

## 5. Query Processing Flow

```text
User Question
      |
      v
Query Validation
      |
      v
Arabic Query Processing
      |
      v
Dense Retrieval + BM25
      |
      v
Score Fusion (RRF)
      |
      v
Metadata Filtering
      |
      v
Cross-Encoder Reranking
      |
      v
Context Construction
      |
      v
LLM Generation
      |
      v
Citation Validation
      |
      v
Final Response
```

## 6. Legal Source Management

Every legal source should preserve its identity and provenance.

Recommended metadata:
- Law name.
- Law number.
- Publication year.
- Article number.
- Chapter.
- Official publication source.
- Effective date.
- Amendment references.
- Legal status.
- Document version.
- Page number.

The system must distinguish:
- Original legal text.
- Amended legal text.
- Repealed provisions.
- Current consolidated text.
- User-uploaded documents.

## 7. Source Authority

Official legal sources and user-uploaded documents must be treated differently.

User-uploaded documents must never automatically become authoritative legal sources.

The application should preserve source provenance and clearly communicate uncertainty.

## 8. Design Principles

- Modular architecture.
- Separation of concerns.
- Dependency injection.
- Typed interfaces.
- Testability.
- Reproducibility.
- Source traceability.
- Arabic-first processing.
- Controlled provider fallback.
- Secure document handling.

## 9. Important Constraint

The LLM is not the legal knowledge database. Retrieved legal sources provide the factual basis for answers. The LLM is responsible for interpreting and explaining the retrieved context, not inventing legislation.