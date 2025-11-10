from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global application settings loaded from environment variables or defaults.
    """

    # Project
    PROJECT_NAME: str = "EKA - Enterprise Knowledge Assistant"
    ENV: str = Field(default="dev", description="Environment: dev | prod")

    # Data directories
    DATA_RAW_DIR: str = "data/raw"
    DATA_QA_DIR: str = "data/qa_dataset"
    DATA_PROCESSED_DIR: str = "data/processed"

    # Database
    COLLECTION_NAME: str = "medical_knowledge_base"
    TMP_COLLECTION_NAME: str = "medical_knowledge_base_tmp"
    CHROMA_API_IMPL: str = "rest"
    CHROMA_SERVER_HOST: str = "localhost"
    CHROMA_SERVER_PORT: int = 8000

    # Model configs
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    FALLBACK_LLM_MODEL: str = "gpt-4o-mini"
    LOCAL_LLM_MODEL: str = "google/flan-t5-base"

    # Redis
    REDIS_HOST: str = Field(default="127.0.0.1", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour default

    # Conversational memory
    MAX_TURNS: int = 10

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100

    # API keys
    OPENAI_API_KEY: str | None = Field(default=None, env="OPENAI_API_KEY")

    # Logging
    LOG_LEVEL: str = "INFO"

    # Backend API URL
    API_URL: str = "http://0.0.0.0:8001/query"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
