from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from .database import get_app_by_api_key

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def get_current_app(api_key: str = Security(api_key_header)) -> dict:
    app = get_app_by_api_key(api_key)
    if not app:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return app
