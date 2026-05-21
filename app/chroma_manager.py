from functools import lru_cache
import chromadb
from .config import settings


@lru_cache(maxsize=1)
def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


# ---------------------------------------------------------------------------
# Internal helpers — collections are namespaced as  {chroma_prefix}_{name}
# so each app's data is completely isolated.
# ---------------------------------------------------------------------------

def _full_name(chroma_prefix: str, name: str) -> str:
    return f"{chroma_prefix}_{name}"


def _user_name(chroma_prefix: str, full: str) -> str:
    prefix = f"{chroma_prefix}_"
    return full[len(prefix):] if full.startswith(prefix) else full


# ---------------------------------------------------------------------------
# Public API (all take chroma_prefix from the authenticated app record)
# ---------------------------------------------------------------------------

# All collections use cosine distance so that query distances are in the [0, 1]
# range expected by callers (e.g. DISTANCE_THRESHOLD = 0.65 in Django RAG apps).
# L2 (the ChromaDB default) returns squared Euclidean distances on normalised
# vectors, which are roughly 2× larger than cosine distances for the same pair
# of documents — causing every result to exceed a cosine-calibrated threshold.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def create_collection(chroma_prefix: str, name: str) -> chromadb.Collection:
    return get_client().create_collection(
        _full_name(chroma_prefix, name),
        metadata=_COLLECTION_METADATA,
    )


def get_collection(chroma_prefix: str, name: str) -> chromadb.Collection:
    return get_client().get_collection(_full_name(chroma_prefix, name))


def get_or_create_collection(chroma_prefix: str, name: str) -> chromadb.Collection:
    return get_client().get_or_create_collection(
        _full_name(chroma_prefix, name),
        metadata=_COLLECTION_METADATA,
    )


def list_collections(chroma_prefix: str) -> list[str]:
    prefix = f"{chroma_prefix}_"
    all_cols = get_client().list_collections()
    return [_user_name(chroma_prefix, c.name) for c in all_cols if c.name.startswith(prefix)]


def delete_collection(chroma_prefix: str, name: str) -> None:
    get_client().delete_collection(_full_name(chroma_prefix, name))
