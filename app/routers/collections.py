from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..auth import get_current_app
from .. import chroma_manager

router = APIRouter(prefix="/collections", tags=["Collections"])


class CollectionCreate(BaseModel):
    name: str


@router.post("", summary="Create a new collection for this app")
def create_collection(body: CollectionCreate, app: dict = Depends(get_current_app)):
    try:
        chroma_manager.create_collection(app["chroma_prefix"], body.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"name": body.name, "count": 0}


@router.get("", summary="List all collections belonging to this app")
def list_collections(app: dict = Depends(get_current_app)):
    names = chroma_manager.list_collections(app["chroma_prefix"])
    result = []
    for name in names:
        col = chroma_manager.get_collection(app["chroma_prefix"], name)
        result.append({"name": name, "count": col.count()})
    return result


@router.delete("/{name}", summary="Delete a collection and all its documents")
def delete_collection(name: str, app: dict = Depends(get_current_app)):
    try:
        chroma_manager.delete_collection(app["chroma_prefix"], name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": name}
