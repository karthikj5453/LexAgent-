from pydantic_settings import BaseSettings
from pydantic import Field
import os

class Settings(BaseSettings):
    # NIM / LLM
    nim_api_key: str = Field(default="")
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    nim_reasoning_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    nim_fast_model: str = "nvidia/llama-3.1-8b-instruct"
    nim_guardrail_model: str = "nvidia/llama-guard-3-8b"
    nim_embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    
    # Retrieval
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "lexagent_contracts"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5
    rrf_k: int = 60
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_file_size_mb: int = 50
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
