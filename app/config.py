from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    chroma_persist_dir: str = "./chroma_data"
    db_path: str = "./apps.db"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
