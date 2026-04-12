"""
config/settings.py
Central configuration management using Pydantic Settings.
All environment variables are validated and typed here.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# Project root
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"

# Ensure directories exist
for d in [DATA_DIR, LOG_DIR, DATA_DIR / "chroma_db", DATA_DIR / "cache", DATA_DIR / "uploads"]:
    d.mkdir(parents=True, exist_ok=True)


class LLMSettings(BaseSettings):
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-70b-8192", env="GROQ_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3", env="OLLAMA_MODEL")

    model_config = {"env_file": ".env", "extra": "ignore"}


class Neo4jSettings(BaseSettings):
    uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    password: str = Field(default="password", env="NEO4J_PASSWORD")
    database: str = Field(default="neo4j", env="NEO4J_DATABASE")
    use_networkx_fallback: bool = Field(default=False, env="USE_NETWORKX_FALLBACK")

    model_config = {"env_file": ".env", "extra": "ignore"}


class VectorDBSettings(BaseSettings):
    persist_dir: str = Field(default=str(DATA_DIR / "chroma_db"), env="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="kg_rag_collection", env="CHROMA_COLLECTION_NAME")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    model_config = {"env_file": ".env", "extra": "ignore"}


class RetrievalSettings(BaseSettings):
    top_k: int = Field(default=5, env="TOP_K_RESULTS")
    max_context_tokens: int = Field(default=3000, env="MAX_CONTEXT_TOKENS")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, env="CHUNK_OVERLAP")

    model_config = {"env_file": ".env", "extra": "ignore"}


class CacheSettings(BaseSettings):
    cache_dir: str = Field(default=str(DATA_DIR / "cache"), env="CACHE_DIR")
    ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")

    model_config = {"env_file": ".env", "extra": "ignore"}


class LogSettings(BaseSettings):
    level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default=str(LOG_DIR / "app.log"), env="LOG_FILE")

    model_config = {"env_file": ".env", "extra": "ignore"}


class Settings:
    """Aggregated settings object."""
    def __init__(self):
        self.llm = LLMSettings()
        self.neo4j = Neo4jSettings()
        self.vector_db = VectorDBSettings()
        self.retrieval = RetrievalSettings()
        self.cache = CacheSettings()
        self.log = LogSettings()

    def is_groq_configured(self) -> bool:
        return bool(self.llm.groq_api_key and self.llm.groq_api_key != "your_groq_api_key_here")

    def is_neo4j_configured(self) -> bool:
        return not self.neo4j.use_networkx_fallback


# Singleton
settings = Settings()
