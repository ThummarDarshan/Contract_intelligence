from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Centralized configuration for the Contract Intelligence Platform."""
    
    # --- App ---
    app_name: str = "Contract Intelligence Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379/0")
    
    # --- Qdrant ---
    qdrant_url: str = Field(default="http://localhost:6333")
    collection_name: str = "contracts"
    vector_size: int = 384  # all-MiniLM-L6-v2
    
    # --- Ollama ---
    ollama_url: str = Field(default="http://localhost:11434/api/generate")
    ollama_model: str = "qwen2.5:7b"
    ollama_temperature: float = 0.1
    ollama_max_tokens: int = 256
    ollama_timeout: int = 60
    
    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # --- Chunking ---
    chunk_max_chars: int = 800
    chunk_overlap: int = 100
    
    # --- Upload ---
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf", ".docx", ".png", ".jpg", ".jpeg"]
    upload_dir: str = "data/uploads"
    
    # --- QA ---
    confidence_high: float = 6.0
    confidence_low: float = 3.0
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
