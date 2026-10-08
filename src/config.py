"""
config.py - Central configuration for the RAG pipeline.

Loads environment variables and defines constants used across modules.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ============================================================
# API Keys
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it in your .env file.\n"
        "Get a free key from: https://console.groq.com/keys"
    )


# ============================================================
# LLM Configuration
# ============================================================
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 1024


# ============================================================
# Embeddings Configuration
# ============================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# Text Chunking Configuration
# ============================================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# Vector Store Configuration
# ============================================================
VECTORSTORE_DIR = "chroma_db"


# ============================================================
# Retrieval Configuration
# ============================================================
RETRIEVAL_K = 4