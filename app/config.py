from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./chroma_data"
    db_path: str = "./apps.db"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
