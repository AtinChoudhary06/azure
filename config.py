"""
Central place for all configuration. Everything is loaded from environment
variables (see .env.example). Nothing is hardcoded here - no keys, no
endpoints - so this file is safe to commit to git.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Check your .env file against .env.example."
        )
    return value


class Settings:
    # ---- Azure Blob Storage ----
    STORAGE_CONNECTION_STRING = _require("AZURE_STORAGE_CONNECTION_STRING")
    STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "pdf-documents")

    # ---- Azure Document Intelligence ----
    DOCINTEL_ENDPOINT = _require("AZURE_DOCINTEL_ENDPOINT")
    DOCINTEL_KEY = _require("AZURE_DOCINTEL_KEY")

    # ---- Azure OpenAI - Embeddings ----
    # IMPORTANT: this must be the BASE resource URL only, e.g.
    #   https://<your-resource>.cognitiveservices.azure.com/
    # NOT the full /openai/deployments/.../embeddings?api-version=... path.
    # The SDK builds that path internally.
    EMBEDDING_ENDPOINT = _require("AZURE_EMBEDDING_ENDPOINT")
    EMBEDDING_KEY = _require("AZURE_EMBEDDING_KEY")
    EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    EMBEDDING_DIM = int(os.getenv("AZURE_EMBEDDING_DIM", "1536"))

    # ---- Azure AI Foundry - Chat model (e.g. gpt-oss-120b) ----
    # Base "/models" endpoint, e.g.
    #   https://<your-resource>.services.ai.azure.com/models
    CHAT_ENDPOINT = _require("AZURE_CHAT_ENDPOINT")
    CHAT_KEY = _require("AZURE_CHAT_KEY")
    CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-oss-120b")

    # ---- Azure AI Search ----
    SEARCH_ENDPOINT = _require("AZURE_SEARCH_ENDPOINT")
    SEARCH_KEY = _require("AZURE_SEARCH_KEY")
    SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-docs-index")

    # ---- RAG behavior ----
    CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "300"))
    CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "30"))
    TOP_K = int(os.getenv("RAG_TOP_K", "5"))


settings = Settings()