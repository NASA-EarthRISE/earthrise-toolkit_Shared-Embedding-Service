from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, model_validator
from ..auth import get_current_app
from .. import chroma_manager
from ..embeddings import generate_embeddings, get_reranker

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
    use_hybrid: bool = False   # merge BM25 keyword results with vector results
    rerank: bool = False       # cross-encoder reranking after retrieval
    top_k_vector: int = 20     # vector candidates to fetch (when hybrid or rerank is on)
    top_k_bm25: int = 20       # BM25 candidates to merge (when use_hybrid is on)


class DeleteDocumentsRequest(BaseModel):
    ids: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hybrid_rerank_query(col, query_texts, query_embeddings, body):
    """
    Enhanced retrieval: BM25 + vector hybrid merge with optional cross-encoder reranking.

    Returns results in ChromaDB format: {ids, documents, metadatas, distances}.
    When rerank=True the 'distances' values are cross-encoder scores (higher = more relevant).
    """
    from rank_bm25 import BM25Okapi

    candidate_k = max(body.top_k_vector, body.n_results)
    total = col.count()
    fetch_k = min(candidate_k, total) if total > 0 else 1

    ids_out, documents_out, metadatas_out, distances_out = [], [], [], []

    # Fetch full corpus once — shared across all queries in this request.
    # The where filter is applied here so BM25 only sees the same subset as vector search.
    all_data = col.get(include=["documents", "metadatas"], where=body.where)
    all_ids: list[str] = all_data["ids"]
    all_docs: list[str] = all_data["documents"]
    all_metas: list[dict] = all_data["metadatas"] or [{} for _ in all_ids]

    # Build BM25 index once for the collection
    if body.use_hybrid and all_docs:
        tokenized = [d.lower().split() for d in all_docs]
        bm25 = BM25Okapi(tokenized)
    else:
        bm25 = None

    for q_idx, query_text in enumerate(query_texts):
        # --- Vector search ---
        vec_res = col.query(
            query_embeddings=[query_embeddings[q_idx]],
            n_results=fetch_k,
            where=body.where,
            include=["documents", "metadatas", "distances"],
        )
        vec_ids: list[str] = vec_res["ids"][0]
        vec_docs: list[str] = vec_res["documents"][0]
        vec_metas: list[dict] = vec_res["metadatas"][0]

        # id → (doc, meta) map for deduplication
        id_to_data: dict[str, tuple[str, dict]] = {
            vid: (vdoc, vmeta)
            for vid, vdoc, vmeta in zip(vec_ids, vec_docs, vec_metas)
        }
        merged_ids = list(vec_ids)

        # --- BM25 search & merge ---
        if bm25 is not None:
            bm25_scores = bm25.get_scores(query_text.lower().split())
            ranked_bm25 = sorted(
                zip(all_ids, all_docs, all_metas, bm25_scores),
                key=lambda x: x[3],
                reverse=True,
            )[: body.top_k_bm25]

            for bid, bdoc, bmeta, _ in ranked_bm25:
                if bid not in id_to_data:
                    id_to_data[bid] = (bdoc, bmeta)
                    merged_ids.append(bid)

        merged_docs = [id_to_data[mid][0] for mid in merged_ids]
        merged_metas = [id_to_data[mid][1] for mid in merged_ids]

        # --- Cross-encoder reranking ---
        if body.rerank and merged_docs:
            reranker = get_reranker()
            pairs = [[query_text, d] for d in merged_docs]
            scores = reranker.predict(pairs)

            ranked = sorted(
                zip(merged_ids, merged_docs, merged_metas, scores.tolist()),
                key=lambda x: x[3],
                reverse=True,
            )[: body.n_results]

            final_ids = [r[0] for r in ranked]
            final_docs = [r[1] for r in ranked]
            final_metas = [r[2] for r in ranked]
            final_scores = [r[3] for r in ranked]
        else:
            final_ids = merged_ids[: body.n_results]
            final_docs = merged_docs[: body.n_results]
            final_metas = merged_metas[: body.n_results]
            final_scores = [0.0] * len(final_ids)

        ids_out.append(final_ids)
        documents_out.append(final_docs)
        metadatas_out.append(final_metas)
        distances_out.append(final_scores)

    return {
        "ids": ids_out,
        "documents": documents_out,
        "metadatas": metadatas_out,
        "distances": distances_out,
    }


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

    # Fast path: pure vector search (original behaviour, fully backward-compatible)
    if not body.use_hybrid and not body.rerank:
        results = col.query(
            query_embeddings=query_embeddings,
            n_results=body.n_results,
            where=body.where,
        )
        return results

    # Enhanced path: hybrid BM25+vector and/or cross-encoder reranking
    return _hybrid_rerank_query(col, body.query_texts, query_embeddings, body)


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
