from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = ROOT_DIR / "raw_docs"
DATA_DIR = ROOT_DIR / "data"

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
CHUNK_IDS_PATH = DATA_DIR / "chunk_ids.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

BM25_TOP_K = 20
VECTOR_TOP_K = 5

EMBED_MODEL = os.environ.get("EMBED_MODEL", "google/EmbeddingGemma-300m")
EMBED_BATCH_SIZE = 32
EMBED_MAX_LENGTH = 512

GEN_MODEL = os.environ.get("GEN_MODEL", "google/gemma-4-E4B-it")
GEN_MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 0.1
GEN_DO_SAMPLE = False
GEN_MAX_CONTEXT_CHUNKS = 5

NO_RESULT_ANSWER = "資料庫中無相關資訊"
