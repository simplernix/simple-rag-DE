import os
from dotenv import load_dotenv

load_dotenv()

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500          # tokens ~ 400-600 words
CHUNK_OVERLAP = 100

# Vector DB
VECTOR_STORE_PATH = "vector_store/faiss_index"

# Ollama
LLM_MODEL = "llama3"        # or "mistral", "phi3", "gemma2:9b", etc.
OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieval
TOP_K = 6