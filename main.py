from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import apps, collections, documents, embed


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise the SQLite app-registry on startup
    init_db()
    from app.embeddings import get_model
    import asyncio
    await asyncio.get_event_loop().run_in_executor(None, get_model)
    yield


app = FastAPI(
    title="Shared Embedding Service",
    description=(
        "A centralised embedding service that lets multiple apps store and query "
        "vector embeddings while keeping each app's data fully isolated.\n\n"
        "**Quick start**\n"
        "1. `POST /apps/register` — create an app and get an API key.\n"
        "2. Use `X-API-Key: <key>` on every subsequent request.\n"
        "3. `POST /collections/{name}/documents` — add documents (auto-embedded).\n"
        "4. `POST /collections/{name}/query` — semantic search.\n"
        "5. Or use `POST /embed` with `RemoteEmbeddingFunction` in your own "
        "ChromaDB client.\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(apps.router)
app.include_router(embed.router)
app.include_router(collections.router)
app.include_router(documents.router)


@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "service": "Shared Embedding Service", "version": "1.0.0"}
