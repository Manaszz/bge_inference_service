from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


SparseTokenMapping = Literal["hash", "tokenizer"]


class Settings(BaseSettings):
    # Service
    service_name: str = Field(default="bge-inference-service", env="SERVICE_NAME")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8011, env="PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Models
    embedding_model_name: str = Field(default="BAAI/bge-m3", env="EMBEDDING_MODEL_NAME")
    reranker_model_name: str = Field(default="BAAI/bge-reranker-v2-m3", env="RERANKER_MODEL_NAME")
    # BGE-M3 dense embedding dimensionality (default 1024)
    embedding_size: int = Field(default=1024, env="EMBEDDING_SIZE")

    # Runtime
    device: str = Field(default="cuda", env="DEVICE")  # cuda|cpu
    use_fp16: bool = Field(default=True, env="USE_FP16")

    # Sparse mapping (must match current project algorithms)
    sparse_token_mapping: SparseTokenMapping = Field(default="hash", env="SPARSE_TOKEN_MAPPING")
    sparse_index_space: int = Field(default=2**20, env="SPARSE_INDEX_SPACE")
    tokenizer_trust_remote_code: bool = Field(default=True, env="TOKENIZER_TRUST_REMOTE_CODE")

    # Limits
    max_text_chars: int = Field(default=10000, env="MAX_TEXT_CHARS")
    max_batch_size: int = Field(default=64, env="MAX_BATCH_SIZE")
    inference_batch_size: int = Field(default=256, env="INFERENCE_BATCH_SIZE")

    # Microbatching (service-level batching across concurrent HTTP requests)
    embedding_microbatch_enabled: bool = Field(default=False, env="EMBEDDING_MICROBATCH_ENABLED")
    embedding_microbatch_max_wait_ms: int = Field(default=10, env="EMBEDDING_MICROBATCH_MAX_WAIT_MS")
    embedding_microbatch_max_batch_texts: int = Field(default=256, env="EMBEDDING_MICROBATCH_MAX_BATCH_TEXTS")
    embedding_microbatch_queue_maxsize: int = Field(default=2048, env="EMBEDDING_MICROBATCH_QUEUE_MAXSIZE")

    # API
    openai_default_model_alias: str = Field(default="bge-m3", env="OPENAI_DEFAULT_MODEL_ALIAS")

    class Config:
        # Always read service-local env (do not depend on repo root .env)
        env_file = Path(__file__).resolve().parent / ".env"
        case_sensitive = False
        extra = "ignore"

settings = Settings()
