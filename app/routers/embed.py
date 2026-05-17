from fastapi import APIRouter, Depends
from pydantic import BaseModel
from ..auth import get_current_app
from ..embeddings import generate_embeddings
from ..config import settings

router = APIRouter(prefix="/embed", tags=["Embeddings"])


class EmbedRequest(BaseModel):
    texts: list[str]


@router.post(
    "",
    summary="Generate embeddings for a list of texts",
    description=(
        "Use this endpoint to implement `RemoteEmbeddingFunction` in client apps. "
        "Each app can only call this with its own API key."
    ),
)
def embed_texts(body: EmbedRequest, app: dict = Depends(get_current_app)):
    embeddings = generate_embeddings(body.texts)
    return {
        "embeddings": embeddings,
        "model": settings.embedding_model,
        "count": len(embeddings),
    }
