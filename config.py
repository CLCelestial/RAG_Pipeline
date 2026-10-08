"""Shared settings for ingestion.py, rag.py, app.py and eval.py."""
import os
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # stop chroma from sending usage stats

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"            # the 5 use case PDFs go here
CHROMA_DIR = BASE_DIR / "chroma_db"     # created by ingestion.py
RESULTS_DIR = BASE_DIR / "results"      # created by eval.py
COLLECTION = "ai_use_cases"

EXPECTED_USE_CASES = ["KJS-AGR-01", "KJS-CES-01", "KJS-CES-02", "KJS-HLC-01", "KJS-SRS-01"]

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_TEMPERATURE = 0.2
LLM_NUM_CTX = 8192   # Ollama's default context is small and it silently cuts long prompts

# chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# retrieval
TOP_K = 5
RETRIEVAL_MODE = "hybrid"   # "dense", "bm25" or "hybrid"


class PrefixedOllamaEmbeddings(Embeddings):
    """nomic-embed-text is trained with task prefixes, so documents and queries get different ones."""

    PREFIXES = {"nomic-embed-text": ("search_document: ", "search_query: ")}

    def __init__(self, model=EMBED_MODEL, base_url=OLLAMA_URL):
        self.model = model
        self.inner = OllamaEmbeddings(model=model, base_url=base_url)
        self.doc_prefix, self.query_prefix = self.PREFIXES.get(model.split(":")[0], ("", ""))

    def embed_documents(self, texts):
        return self.inner.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text):
        return self.inner.embed_query(self.query_prefix + text)


def get_embeddings(model=EMBED_MODEL):
    return PrefixedOllamaEmbeddings(model=model)
