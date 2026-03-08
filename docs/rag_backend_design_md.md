# RAG Backend System Design

## 1. Project Overview

This project implements a **RAG (Retrieval-Augmented Generation) backend service** that allows users to upload documents and query them using LLMs.

The system retrieves relevant document chunks from a vector store and augments the prompt before sending it to the LLM.

Key capabilities:

- Document ingestion and chunking
- Vector embedding and retrieval
- Chat with documents
- Streaming responses
- Multi-model LLM support
- Modular provider architecture

Primary goals:

- Build a scalable LAG backend architecture
- Support multiple LLM providers
- Provide a clean API for frontends
- Demonstrate production-style RAG design

## 2. System Architecture

The system follows a layered architecture.

```text
Client (Web UI)
        │
        ▼
    FastAPI Backend
        │
 ┌──────┴──────────┐
 │                 │
 ▼                 ▼
Chat API      Document API
 │                 │
 ▼                 ▼
      RAG Pipeline
           │
           ▼
       LLM Service
           │
           ▼
    Provider Adapter Layer
           │
 ┌─────────┼──────────┐
 ▼         ▼          ▼
OpenAI   Anthropic   Gemini
           │
           ▼
      Vector Store
           │
           ▼
        Database
```

Core architectural principles:

- Clear separation of concerns
- Provider abstraction for multi-model support
- Streaming-first design
- Modular RAG pipeline

## 3. Core Components

### 3.1 API Layer

The API layer exposes endpoints for chat and document management.

Responsibilities:

- Request validation
- Authentication
- Streaming responses
- Routing to services

Framework:

```text
FastAPI
```

### 3.2 RAG Pipeline

The RAG pipeline retrieves relevant document chunks and constructs the prompt.

Pipeline flow:

```text
User Query
    │
    ▼
Query Embedding
    │
    ▼
Vector Search
    │
    ▼
Top-K Chunks
    │
    ▼
Prompt Builder
    │
    ▼
LLM Generation
```

Steps:

1. Convert query to embedding
2. Retrieve relevant chunks
3. Construct context prompt
4. Call LLM service

### 3.3 LLM Service

The LLM service provides a unified interface for different model providers.

Responsibilities:

- Provider routing
- Request normalization
- Streaming output

Example interface:

```python
generate(request: ChatRequest) -> ChatResponse

stream_generate(request: ChatRequest) -> Iterator[StreamChunk]
```

### 3.4 Provider Adapter Layer

Each model provider implements a standardized interface.

Example providers:

- OpenAI
- Anthropic
- Gemini
- DeepSeek

Responsibilities:

- Convert internal request format
- Call provider SDK
- Normalize responses
- Handle streaming events

Example:

```text
providers/
    openai_provider.py
    anthropic_provider.py
    gemini_provider.py
```

## 4. API Design

### Chat

#### POST /chat

Send a chat request.

Request:

```json
{
  "messages": [
    {"role": "user", "content": "What is RAG?"}
  ],
  "model": "gpt-4.1-mini"
}
```

Response:

```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": [
    {"document": "paper.pdf", "chunk_id": 12}
  ]
}
```

#### POST /chat/stream

Streaming chat response using SSE.

Response stream:

```text
data: {"text": "RAG stands"}
data: {"text": " for Retrieval"}
data: {"done": true}
```

### Documents

#### POST /documents/upload

Upload a document.

Request:

```text
multipart/form-data
```

Response:

```json
{
  "document_id": "doc_123",
  "status": "processing"
}
```

#### GET /documents

List uploaded documents.

Response:

```json
[
  {
    "id": "doc_123",
    "filename": "paper.pdf"
  }
]
```

#### DELETE /documents/{id}

Delete a document.

## 5. Database Schema

### Users

```text
users
-----
id
email
password_hash
created_at
```

### Conversations

```text
conversations
-------------
id
user_id
title
created_at
```

### Messages

```text
messages
--------
id
conversation_id
role
content
created_at
```

### Documents

```text
documents
---------
id
user_id
filename
created_at
```

### Document Chunks

```text
document_chunks
---------------
id
document_id
chunk_text
embedding_vector
chunk_index
```

## 6. Vector Storage

Embeddings are stored in a vector database.

Possible options:

- pgvector
- Qdrant
- Pinecone
- Weaviate

Example workflow:

```text
Document
  │
  ▼
Text Splitter
  │
  ▼
Chunks
  │
  ▼
Embedding Model
  │
  ▼
Vector Store
```

## 7. Streaming Design

Streaming responses use **Server-Sent Events (SSE)**.

Advantages:

- Browser-native support
- Low overhead
- Easy integration with FastAPI

Example server response:

```text
data: {"text": "Hello"}
data: {"text": " world"}
data: {"done": true}
```

Client receives tokens incrementally.

## 8. LLM Abstraction

Internal request schema:

```python
class ChatRequest:
    model: str
    messages: List[ChatMessage]
    temperature: float
    stream: bool
```

Internal response schema:

```python
class ChatResponse:
    content: str
    usage: UsageInfo
```

Stream chunk schema:

```python
class StreamChunk:
    text: str
    done: bool
```

This abstraction ensures:

- Provider independence
- Easier testing
- Future extensibility

## 9. Project Structure

```text
backend
│
├── app
│   ├── api
│   │   ├── chat.py
│   │   └── documents.py
│   │
│   ├── rag
│   │   ├── pipeline.py
│   │   ├── retriever.py
│   │   └── prompt_builder.py
│   │
│   ├── llm
│   │   ├── base.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   └── providers
│   │       └── openai_provider.py
│   │
│   ├── db
│   │   └── models.py
│   │
│   └── core
│       └── config.py
│
└── main.py
```

## 10. Future Improvements

Potential future enhancements:

### Retrieval

- Hybrid search (vector + keyword)
- Reranking models
- Query rewriting

### LLM

- Model routing
- Cost optimization
- Tool calling

### Infrastructure

- Background ingestion jobs
- Rate limiting
- Observability
- Usage tracking

### Product

- Multi-tenant support
- Document permissions
- Web UI improvements

## 11. Technology Stack

Backend:

- FastAPI
- Python 3.11
- Pydantic
- SQLAlchemy

Vector DB:

- pgvector / Qdrant

LLM Providers:

- OpenAI
- Anthropic
- Gemini

Frontend (optional):

- Next.js
- React

## 12. Development Roadmap

### Phase 1

- Basic chat API
- OpenAI integration
- Streaming responses

### Phase 2

- Document upload
- Chunking and embeddings
- Vector retrieval

### Phase 3

- RAG pipeline
- Citation support

### Phase 4

- User authentication
- Multi-user support

### Phase 5

- Multi-model support
- Retrieval improvements

