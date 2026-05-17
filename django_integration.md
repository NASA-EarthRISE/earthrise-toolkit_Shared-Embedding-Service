# Django Integration Guide

This guide explains how to migrate a Django app's `rag.py` from the current
local-model setup to the Shared Embedding Service.

The existing `rag.py` uses `ChromaVectorStore`, which loads a
`sentence-transformers` model **inside every Django process**. That means each
app running on each worker carries the full model in memory (~400 MB for
`all-mpnet-base-v2`). The Shared Embedding Service loads the model once and
serves all apps over HTTP.

---

## Two migration paths

| | **Option A — Full hosted API** | **Option B — Remote embeddings only** |
|---|---|---|
| Where data lives | Shared Embedding Service (central ChromaDB) | Local ChromaDB inside each Django app |
| Where the model runs | Shared Embedding Service | Shared Embedding Service |
| Code change size | Medium — new store class | Minimal — swap one function |
| `where`-filter deletes | Requires workaround (see below) | Works unchanged |
| Good when | You want no ChromaDB install in each app | You want to keep existing local data |

Both options eliminate the `sentence-transformers` dependency and the in-process
model load from every Django worker.

---

## Prerequisites

1. The Shared Embedding Service is running and reachable from your Django
   servers (e.g. `http://embedding-service:8000`).
2. You have registered your app and have an API key:

```bash
curl -X POST http://embedding-service:8000/apps/register \
     -H "Content-Type: application/json" \
     -d '{"name": "pi-assist"}'
```

Save the returned `api_key`. It is shown only once.

---

## One-time Django settings

Add to `settings.py` (or `.env` / secrets manager):

```python
# Shared Embedding Service
SHARED_EMBEDDING_SERVICE_URL     = "http://embedding-service:8000"
SHARED_EMBEDDING_SERVICE_API_KEY = "ses-..."   # from /apps/register

# Set to "shared" to use the service, or keep "local" / "proxy" unchanged
RAG_MODE = "shared"
```

---

## Option A — Full hosted API

Replace `ChromaVectorStore` with a new `SharedEmbeddingServiceStore` class that
speaks to the service's REST API. The public interface (`upsert`, `search`,
`delete_session_docs`) is identical to `ChromaVectorStore`, so `get_store()`,
`build_context_snippets()`, and all call sites stay the same.

### 1. Copy `client/embedding_function.py` into your Django app

You only need this if you also want to use **Option B** (local ChromaDB with
remote embeddings). For Option A alone you can skip this step — everything goes
through `requests` directly.

### 2. Add `SharedEmbeddingServiceStore` to `rag.py`

Insert this class anywhere before `get_store()`:

```python
# ---------- Shared Embedding Service store ----------
class SharedEmbeddingServiceStore:
    """
    Drop-in replacement for ChromaVectorStore that delegates embedding
    generation and vector storage to the Shared Embedding Service.

    The public interface is identical: upsert / search / delete_session_docs.
    """

    def __init__(self):
        self.base = (
            getattr(settings, "SHARED_EMBEDDING_SERVICE_URL", "http://localhost:8000")
        ).rstrip("/")
        self.api_key = getattr(settings, "SHARED_EMBEDDING_SERVICE_API_KEY", "")
        self.collection = getattr(settings, "CHROMA_COLLECTION", "pi_assist_docs")
        self._headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        LOG.info(
            f"[RAG] Shared Embedding Service: {self.base}  "
            f"collection: {self.collection}"
        )

    def upsert(self, docs: List[Dict[str, Any]]):
        if not docs:
            return
        payload = {
            "documents": [d["text"] for d in docs],
            "ids":       [d["id"]   for d in docs],
            "metadatas": [d.get("metadata", {}) for d in docs],
        }
        resp = requests.put(
            f"{self.base}/collections/{self.collection}/documents",
            json=payload,
            headers=self._headers,
            timeout=120,
        )
        resp.raise_for_status()
        LOG.info(f"[RAG] Upserted {len(docs)} doc chunks via shared service.")

    def search(
        self, query: str, top_k: int = DEFAULT_TOP_K, session_id: str = None
    ) -> List[Dict[str, Any]]:
        where = None
        if session_id:
            where = {"$or": [{"session_id": session_id}, {"is_global": True}]}
        else:
            where = {"is_global": True}

        payload = {
            "query_texts": [query],
            "n_results": top_k,
            "where": where,
        }
        resp = requests.post(
            f"{self.base}/collections/{self.collection}/query",
            json=payload,
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        res = resp.json()

        ids       = res.get("ids",       [[]])[0]
        docs      = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0] if "distances" in res else [None] * len(ids)

        results = [
            {"id": ids[i], "text": docs[i], "metadata": metadatas[i], "distance": distances[i]}
            for i in range(len(ids))
        ]
        LOG.info(
            f"[RAG] Shared-service search (session={session_id}) returned "
            f"{len(results)} result(s) for query='{query[:80]}...'"
        )
        return results

    def delete_session_docs(self, session_id: str):
        """
        Delete all chunks for a session.

        The service's DELETE endpoint takes explicit IDs, so we first fetch
        all documents for this collection and filter by session_id locally,
        then delete the matching IDs in one call.
        """
        if not session_id:
            return
        try:
            # Fetch all IDs + metadata (paginate if collection is large)
            resp = requests.get(
                f"{self.base}/collections/{self.collection}/documents",
                params={"limit": 1000},
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            ids       = data.get("ids", [])
            metadatas = data.get("metadatas", [])

            to_delete = [
                doc_id
                for doc_id, meta in zip(ids, metadatas)
                if (meta or {}).get("session_id") == session_id
            ]

            if not to_delete:
                return

            del_resp = requests.delete(
                f"{self.base}/collections/{self.collection}/documents",
                json={"ids": to_delete},
                headers=self._headers,
                timeout=30,
            )
            del_resp.raise_for_status()
            LOG.info(
                f"[RAG] Deleted {len(to_delete)} session docs for session {session_id}"
            )
        except Exception as e:
            LOG.error(f"[RAG] Error deleting session docs via shared service: {e}")
```

> **Note on `delete_session_docs`:** The shared service's DELETE endpoint
> accepts a list of IDs rather than a `where` filter. The implementation above
> fetches up to 1 000 documents and filters locally. If a collection can grow
> beyond 1 000 documents, either increase the `limit` parameter or add a
> `where` parameter to `GET /collections/{name}/documents` in the service
> (pass it through to `col.get(where=...)`).

### 3. Update `get_store()`

Replace the existing `get_store` function (lines 186–194 in `rag.py`):

```python
# Before
def get_store():
    global _store_instance
    if _store_instance is None:
        mode = (getattr(settings, "RAG_MODE", "local") or "local").lower()
        if mode == "proxy":
            _store_instance = LiteLLMRagClient()
        else:
            _store_instance = ChromaVectorStore()
    return _store_instance
```

```python
# After
def get_store():
    global _store_instance
    if _store_instance is None:
        mode = (getattr(settings, "RAG_MODE", "local") or "local").lower()
        if mode == "proxy":
            _store_instance = LiteLLMRagClient()
        elif mode == "shared":
            _store_instance = SharedEmbeddingServiceStore()
        else:
            _store_instance = ChromaVectorStore()
    return _store_instance
```

### 4. Update `diag_info()`

```python
# After
def diag_info() -> dict:
    mode = (getattr(settings, "RAG_MODE", "local") or "local").lower()
    info = {
        "mode": mode,
        "persist_dir": CHROMA_PERSIST_DIR,
        "collection": CHROMA_COLLECTION,
        "exists_on_disk": os.path.isdir(CHROMA_PERSIST_DIR),
        "count": None,
    }
    try:
        store = get_store()
        if isinstance(store, ChromaVectorStore):
            info["count"] = store.collection.count()
        elif isinstance(store, SharedEmbeddingServiceStore):
            resp = requests.get(
                f"{store.base}/collections",
                headers=store._headers,
                timeout=10,
            )
            resp.raise_for_status()
            for col in resp.json():
                if col["name"] == store.collection:
                    info["count"] = col["count"]
                    break
    except Exception as e:
        info["error"] = str(e)
    return info
```

### 5. Remove unused imports

Once all apps are migrated to `mode = "shared"`, these imports in `rag.py` are
no longer needed and can be removed:

```python
# Can be removed after migration
from pathlib import Path
# _resolve_persist_dir and CHROMA_PERSIST_DIR can also be removed
# _build_local_embedding_function can be removed
# ChromaVectorStore class can be removed
```

Leave them in place during a staged rollout so you can switch back by changing
`RAG_MODE`.

---

## Option B — Local ChromaDB with remote embeddings

This is the minimal change. Each Django app keeps its own local ChromaDB on
disk; only embedding generation moves to the shared service. No data migration
is needed.

### 1. Copy `client/embedding_function.py` into your Django app

Place it at e.g. `pi_assist/embedding_function.py`.

### 2. Add settings

```python
SHARED_EMBEDDING_SERVICE_URL     = "http://embedding-service:8000"
SHARED_EMBEDDING_SERVICE_API_KEY = "ses-..."
```

`RAG_MODE` does **not** need to change — you are still using `ChromaVectorStore`.

### 3. Replace `_build_local_embedding_function` in `rag.py`

```python
# Before (lines 51–59)
def _build_local_embedding_function():
    from chromadb.utils import embedding_functions as ef
    model_name = getattr(settings, "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    LOG.info(f"[RAG] Using local embedding model: {model_name}")
    return ef.SentenceTransformerEmbeddingFunction(model_name=model_name)
```

```python
# After
def _build_local_embedding_function():
    from pi_assist.embedding_function import RemoteEmbeddingFunction  # adjust import path
    url     = getattr(settings, "SHARED_EMBEDDING_SERVICE_URL", "http://localhost:8000")
    api_key = getattr(settings, "SHARED_EMBEDDING_SERVICE_API_KEY", "")
    LOG.info(f"[RAG] Using remote embedding service: {url}")
    return RemoteEmbeddingFunction(api_key=api_key, url=url)
```

That is the only change required. `ChromaVectorStore.__init__` already passes
the embedding function into `get_or_create_collection`, so ChromaDB will call
the remote service automatically on every `upsert` and `query`.

> **Important:** `RemoteEmbeddingFunction` calls the shared service's
> `POST /embed` endpoint. The model running on the service must be compatible
> with any embeddings already stored locally. If you are switching models,
> delete the local `chroma_db` directory and re-run your ingest script — the
> same rule that already applies when changing `LOCAL_EMBEDDING_MODEL`.

### 4. Uninstall `sentence-transformers` from the Django app

```bash
pip uninstall sentence-transformers transformers torch  # or update requirements.txt
```

---

## Choosing between Option A and Option B

| Situation | Recommendation |
|---|---|
| Starting fresh / no existing data | **Option A** — simplest long-term setup |
| Existing data in local ChromaDB | **Option B** first; migrate to A later if desired |
| Multiple workers per Django process | Either — both are stateless per-request |
| Collections need `where`-filter deletes on large datasets | **Option B** — local ChromaDB handles these natively |
| You want zero vector-DB ops in the Django app | **Option A** |

---

## Migrating existing data to Option A

If you have data in a local ChromaDB that you want to move into the shared
service, run a one-time script from within the Django environment:

```python
# migrate_to_shared.py
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
django.setup()

import chromadb, requests
from django.conf import settings

LOCAL_DIR        = settings.CHROMA_PERSIST_DIR
COLLECTION_NAME  = settings.CHROMA_COLLECTION
SERVICE_URL      = settings.SHARED_EMBEDDING_SERVICE_URL
API_KEY          = settings.SHARED_EMBEDDING_SERVICE_API_KEY
HEADERS          = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
BATCH_SIZE       = 100

local_client = chromadb.PersistentClient(path=LOCAL_DIR)
col          = local_client.get_collection(COLLECTION_NAME)
total        = col.count()
offset       = 0

print(f"Migrating {total} documents...")

while offset < total:
    batch = col.get(
        limit=BATCH_SIZE,
        offset=offset,
        include=["documents", "metadatas"],
    )
    if not batch["ids"]:
        break

    resp = requests.put(
        f"{SERVICE_URL}/collections/{COLLECTION_NAME}/documents",
        json={
            "documents": batch["documents"],
            "ids":       batch["ids"],
            "metadatas": batch["metadatas"],
        },
        headers=HEADERS,
        timeout=120,
    )
    resp.raise_for_status()
    offset += len(batch["ids"])
    print(f"  {offset}/{total}")

print("Migration complete.")
```

Run once, verify the count via `GET /collections` on the service, then switch
`RAG_MODE = "shared"` and restart Django.
