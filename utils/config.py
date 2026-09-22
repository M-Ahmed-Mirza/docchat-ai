"""
Configuration module for DocChat AI.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_config() -> dict:
    """
    Get application configuration from environment variables.
    
    Returns:
        Dictionary with configuration values.
    """
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "model_name": os.getenv("MODEL_NAME", "gpt-4o-mini"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        "top_k": int(os.getenv("TOP_K", "4")),
    }
