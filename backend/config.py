from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"

    # Embeddings — HuggingFace model, runs locally via sentence-transformers
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"

    # ChromaDB — persisted to disk so you only ingest once
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "techdocs"

    # Document source
    docs_dir: str = "./data/docs"

    # Retrieval behaviour
    top_k: int = 10
    max_retry_loops: int = 2

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

# Single instance imported everywhere
settings = Settings()