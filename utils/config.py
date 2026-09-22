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
        "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
        "model_name": os.getenv("MODEL_NAME", "gemini-2.0-flash"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "models/text-embedding-004"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        "top_k": int(os.getenv("TOP_K", "4")),
    }
