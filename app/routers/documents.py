from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, model_validator
from ..auth import get_current_app
from .. import chroma_manager
from ..embeddings import generate_embeddings

router = APIRouter(prefix="/collections", tags=["Documents"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AddDocumentsRequest(BaseModel):
    documents: list[str]
    ids: list[str]
    metadatas: Optional[list[dict]] = None

    @model_validator(mode="after")
    def lengths_match(self):
        if len(self.documents) != len(self.ids):
            raise ValueError("'documents' and 'ids' must have the same length")
        if self.metadatas is not None and len(self.metadatas) != len(self.ids):
            raise ValueError("'metadatas' and 'ids' must have the same length")
        return self


class UpsertDocumentsRequest(AddDocumentsRequest):
    pass


class QueryRequest(BaseModel):
    query_texts: list[str]
    n_results: int = 10
    where: Optional[dict] = None


class DeleteDocumentsRequest(BaseModel):
    ids: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{collection}/documents",
    summary="Add documents (auto-embeds using the shared model)",
)
def add_documents(
    collection: str,
    body: AddDocumentsRequest,
    app: dict = Depends(get_current_app),
):
    embeddings = generate_embeddings(body.documents)
    col = chroma_manager.get_or_create_collection(app["chroma_prefix"], collection)
    col.add(
        documents=body.documents,
        embeddings=embeddings,
        ids=body.ids,
        metadatas=body.metadatas,
    )
    return {"added": len(body.documents), "collection": collection}


@router.put(
    "/{collection}/documents",
    summary="Upsert documents (add if new, update if existing)",
)
def upsert_documents(
    collection: str,
    body: UpsertDocumentsRequest,
    app: dict = Depends(get_current_app),
):
    embeddings = generate_embeddings(body.documents)
    col = chroma_manager.get_or_create_collection(app["chroma_prefix"], collection)
    col.upsert(
        documents=body.documents,
        embeddings=embeddings,
        ids=body.ids,
        metadatas=body.metadatas,
    )
    return {"upserted": len(body.documents), "collection": collection}


@router.post(
    "/{collection}/query",
    summary="Find the most similar documents for one or more query strings",
)
def query_documents(
    collection: str,
    body: QueryRequest,
    app: dict = Depends(get_current_app),
):
    try:
        col = chroma_manager.get_collection(app["chroma_prefix"], collection)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

    query_embeddings = generate_embeddings(body.query_texts)
    results = col.query(
        query_embeddings=query_embeddings,
        n_results=body.n_results,
        where=body.where,
    )
    return results


@router.get(
    "/{collection}/documents",
    summary="Retrieve documents from a collection (paginated)",
)
def get_documents(
    collection: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    app: dict = Depends(get_current_app),
):
    try:
        col = chroma_manager.get_collection(app["chroma_prefix"], collection)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

    return col.get(limit=limit, offset=offset, include=["documents", "metadatas"])


@router.delete(
    "/{collection}/documents",
    summary="Delete specific documents by ID",
)
def delete_documents(
    collection: str,
    body: DeleteDocumentsRequest,
    app: dict = Depends(get_current_app),
):
    try:
        col = chroma_manager.get_collection(app["chroma_prefix"], collection)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

    col.delete(ids=body.ids)
    return {"deleted": len(body.ids), "collection": collection}
