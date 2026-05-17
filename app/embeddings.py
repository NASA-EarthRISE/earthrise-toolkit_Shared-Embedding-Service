from functools import lru_cache
from sentence_transformers import SentenceTransformer
from .config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()
