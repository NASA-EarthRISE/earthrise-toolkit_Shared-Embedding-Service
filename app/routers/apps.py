from fastapi import APIRouter, Depends
from pydantic import BaseModel
from ..database import create_app, list_apps
from ..auth import get_current_app

router = APIRouter(prefix="/apps", tags=["Apps"])


class AppCreate(BaseModel):
    name: str


@router.post("/register", summary="Register a new app and receive an API key")
def register_app(body: AppCreate):
    app = create_app(body.name)
    return {
        "app_id": app["app_id"],
        "name": app["name"],
        "api_key": app["api_key"],
        "created_at": app["created_at"],
        "note": "Store your API key securely — it will not be shown again.",
    }


@router.get("/me", summary="Get info about the currently authenticated app")
def get_me(app: dict = Depends(get_current_app)):
    return {
        "app_id": app["app_id"],
        "name": app["name"],
        "created_at": app["created_at"],
    }
