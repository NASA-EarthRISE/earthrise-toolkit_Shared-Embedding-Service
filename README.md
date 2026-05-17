# earthrise-toolkit — Shared Embedding Service

[![Python: 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![EarthRISE: Development](https://img.shields.io/badge/EarthRISE-Development-b50000?labelColor=191f4c)](https://appliedsciences.nasa.gov/what-we-do/capacity-building/develop)

A centralized [FastAPI](https://fastapi.tiangolo.com/) service that provides
text embedding generation and vector storage for multiple applications. Each
registered application receives its own API key and an isolated data namespace,
so no app can read or write another app's data.

Part of the [EarthRISE Toolkit](https://github.com/orgs/NASA-EarthRISE/repositories?q=earthrise-toolkit_).

---

## Why this exists

Without this service, every Django application that needs RAG capability must
load a `sentence-transformers` model (~400 MB) into each worker process. The
Shared Embedding Service loads the model once and serves all applications over
HTTP, reducing memory consumption and centralizing the model version used
across the platform.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Shared Embedding Service                   │
│                                                             │
│   FastAPI  ──►  sentence-transformers model (single copy)  │
│               ──►  ChromaDB PersistentClient               │
│                     app_a__collection  (isolated)          │
│                     app_b__collection  (isolated)          │
│                     app_c__collection  (isolated)          │
└─────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
    X-API-Key: A         X-API-Key: B         X-API-Key: C
    Django App A         Django App B         Django App C
```

Each application authenticates with its own `ses-...` API key. Collections are
namespaced internally; an application can only see and modify its own
collections even if it knows the name of a collection belonging to another app.

Client applications have two usage modes:

- **Full hosted API** — the service handles embeddings and vector storage.
  Client apps make HTTP calls; no local ChromaDB or model required.
- **Remote embedding function** — the client keeps a local ChromaDB instance
  but delegates embedding generation to the service via
  `RemoteEmbeddingFunction`. Useful when migrating from a local setup.

---

## Requirements

- Python 3.11+
- 512 MB RAM minimum (model + service overhead)
- Disk space for the ChromaDB vector store and SQLite app registry

---

## Quick Start (local development)

```bash
# 1. Clone the repository
git clone https://github.com/NASA-EarthRISE/earthrise-toolkit_shared-embedding-service
cd earthrise-toolkit_shared-embedding-service

# 2. Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a local .env (copy from the example and adjust if needed)
cp .env.example .env

# 5. Start the development server
uvicorn main:app --reload
```

The service starts at `http://127.0.0.1:8000`.
Interactive API docs are available at `http://127.0.0.1:8000/docs`.

### Register your first application

```bash
curl -s -X POST http://127.0.0.1:8000/apps/register \
     -H "Content-Type: application/json" \
     -d '{"name": "my-app"}' | python3 -m json.tool
```

Save the returned `api_key`. It is shown only once.

---

## Configuration

All settings are read from environment variables or a `.env` file in the
project root. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any `sentence-transformers` model name |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Directory where ChromaDB stores vector data |
| `DB_PATH` | `./apps.db` | Path to the SQLite app-registry database |
| `HF_HOME` | *(system default)* | Override Hugging Face model cache location |

> **Changing `EMBEDDING_MODEL`** invalidates all existing embeddings. Every
> application must re-ingest its documents after this change.

---

## API Reference

All endpoints except `/`, `/apps/register`, require the header:

```
X-API-Key: ses-...
```

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Returns service status and version |

### App Registration

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/apps/register` | None | Register a new app; returns a one-time API key |
| `GET` | `/apps/me` | Required | Returns the authenticated app's name and ID |

**Register request body:**
```json
{ "name": "my-app" }
```

### Embeddings

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/embed` | Required | Generate embedding vectors for a list of texts |

**Request body:**
```json
{ "texts": ["first text", "second text"] }
```

**Response:**
```json
{
  "embeddings": [[0.12, -0.34, ...], [0.56, 0.78, ...]],
  "model": "all-MiniLM-L6-v2",
  "count": 2
}
```

### Collections

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/collections` | Required | Create a new collection |
| `GET` | `/collections` | Required | List all collections for this app (with document counts) |
| `DELETE` | `/collections/{name}` | Required | Delete a collection and all its documents |

**Create request body:**
```json
{ "name": "my-collection" }
```

### Documents

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/collections/{collection}/documents` | Required | Add documents (auto-embeds) |
| `PUT` | `/collections/{collection}/documents` | Required | Upsert documents (add if new, overwrite if existing) |
| `GET` | `/collections/{collection}/documents` | Required | Retrieve documents (paginated via `limit` and `offset`) |
| `DELETE` | `/collections/{collection}/documents` | Required | Delete documents by ID |
| `POST` | `/collections/{collection}/query` | Required | Semantic search |

**Add / Upsert request body:**
```json
{
  "documents": ["First document text.", "Second document text."],
  "ids":       ["doc-1", "doc-2"],
  "metadatas": [{"source": "manual"}, {"source": "manual"}]
}
```
`metadatas` is optional. `documents` and `ids` must be the same length.

**Query request body:**
```json
{
  "query_texts": ["search phrase"],
  "n_results": 10,
  "where": { "source": "manual" }
}
```
`n_results` defaults to `10`. `where` is an optional
[ChromaDB metadata filter](https://docs.trychroma.com/guides#using-where-filters).

**Delete request body:**
```json
{ "ids": ["doc-1", "doc-2"] }
```

---

## Client Usage

### Option A — Full hosted API (no local ChromaDB needed)

```python
import requests

BASE    = "http://localhost:8000"
HEADERS = {"X-API-Key": "ses-...", "Content-Type": "application/json"}

# Add documents
requests.post(f"{BASE}/collections/my-docs/documents", headers=HEADERS, json={
    "documents": ["EarthRISE supports open science.", "FastAPI powers this service."],
    "ids":       ["1", "2"],
})

# Semantic search
results = requests.post(f"{BASE}/collections/my-docs/query", headers=HEADERS, json={
    "query_texts": ["open science tools"],
    "n_results": 5,
}).json()
```

### Option B — RemoteEmbeddingFunction with local ChromaDB

Copy `client/embedding_function.py` into your project and use it as a drop-in
ChromaDB `EmbeddingFunction`. Your local ChromaDB instance handles data storage
while the service handles embedding generation.

```python
import chromadb
from embedding_function import RemoteEmbeddingFunction

ef = RemoteEmbeddingFunction(
    api_key="ses-...",
    url="http://localhost:8000",
)

client     = chromadb.Client()
collection = client.create_collection("my-docs", embedding_function=ef)

# Embeddings are generated by the remote service
collection.add(documents=["EarthRISE supports open science."], ids=["1"])

# Query embedding is also generated remotely
results = collection.query(query_texts=["open science tools"], n_results=5)
```

---

## Django Integration

See **[django_integration.md](django_integration.md)** for step-by-step
instructions on migrating an existing Django RAG setup to use this service,
including a data migration script for moving existing ChromaDB data to the
hosted service.

---

## Production Deployment

See **[production_installation.md](production_installation.md)** for a complete
guide covering:

- Amazon Linux system preparation
- Gunicorn + Uvicorn workers + Unix socket (same stack as Django)
- systemd service unit with security hardening
- Nginx reverse proxy configuration
- Start / stop / reload / update procedures
- Log rotation and data backup

---

## Repository Structure

```
├── main.py                        # FastAPI application entry point
├── app/
│   ├── config.py                  # Settings via .env / environment variables
│   ├── database.py                # SQLite app registry
│   ├── auth.py                    # API key authentication dependency
│   ├── embeddings.py              # sentence-transformers wrapper (singleton)
│   ├── chroma_manager.py          # ChromaDB client with per-app namespacing
│   └── routers/
│       ├── apps.py                # POST /apps/register  GET /apps/me
│       ├── embed.py               # POST /embed
│       ├── collections.py         # Collection CRUD
│       └── documents.py           # Document add / upsert / query / get / delete
├── client/
│   └── embedding_function.py      # RemoteEmbeddingFunction (copy into client apps)
├── requirements.txt
├── .env.example
├── django_integration.md          # Migration guide for Django RAG apps
└── production_installation.md     # Server deployment guide
```

---

## License

[MIT](LICENSE) — Copyright (c) 2026 NASA EarthRISE
