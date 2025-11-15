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

    # Training config
    BASE_LLM_MODEL: str = "google/flan-t5-base"
    TRAINING_DATA_PATH: str = "data/qa_dataset/train.jsonl"
    EVAL_DATA_PATH: str = "data/qa_dataset/train.jsonl"
    LORA_PATH: str = "src/models/outputs/flan-t5-finetuned"
    # Core training hyperparameters
    PER_DEVICE_TRAIN_BATCH_SIZE: int = 4
    PER_DEVICE_EVAL_BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 4
    LEARNING_RATE: float = 1e-4
    LR_SCHEDULER_TYPE: str = "cosine"
    OPTIMIZER: str = "adamw_torch"
    WARMUP_RATIO: float = 0.05
    NUM_TRAIN_EPOCHS: int = 5
    USE_FP16: bool = True
    # Logging and checkpointing
    LOGGING_STEPS: int = 50
    SAVE_STEPS: int = 500
    SAVE_TOTAL_LIMIT: int = 3
    REPORT_TO: str = "none"

    # LoRA configuration
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_TARGET_MODULES: list[str] = ["q", "v"]
    LORA_DROPOUT: float = 0.05
    LORA_BIAS: str = "none"
    LORA_TASK_TYPE: str = "SEQ_2_SEQ_LM"
    LORA_FAN_IN_FAN_OUT: bool = False
    LORA_MODULES_TO_SAVE: list | None = None
    LORA_LAYERS_TO_TRANSFORM: list | None = None
    LORA_LAYERS_PATTERN: str | None = None
    LORA_INFERENCE_MODE: bool = False

    # MlFlow configuration
    ENABLE_MLFLOW_LOGGING: bool = True
    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
    MLFLOW_EXPERIMENT_NAME: str = "flan_t5_finetuning"
    MLFLOW_RUN_NAME: str = "flan_t5_lora_run_01"

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
    API_URL: str = "http://api:8001/query"  # "http://0.0.0.0:8001/query"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
