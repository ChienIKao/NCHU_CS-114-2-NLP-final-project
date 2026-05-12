from pathlib import Path
import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = ROOT_DIR / "raw_docs"
DATA_DIR = ROOT_DIR / "data"

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
CHUNK_IDS_PATH = DATA_DIR / "chunk_ids.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

SEMANTIC_CHUNKING = _env_bool("SEMANTIC_CHUNKING", True)
SEMANTIC_BREAKPOINT_PERCENTILE = 75  # similarities below this percentile become breakpoints
SEMANTIC_MAX_CHUNK_CHARS = 200       # small chunk hard cap (used for retrieval)
SEMANTIC_PARENT_CHUNK_CHARS = 1000   # parent chunk hard cap (sent to generator)
SEMANTIC_MIN_CHUNK_CHARS = 60        # chunks shorter than this are merged into the previous
SEMANTIC_OVERLAP_RATIO = float(os.environ.get("SEMANTIC_OVERLAP_RATIO", "0.2"))  # 0–1: fraction of sentences to carry over

PARENT_CHUNKS_PATH = DATA_DIR / "parent_chunks.jsonl"

BM25_TOP_K = 20
VECTOR_TOP_K = 5

EMBED_MODEL = os.environ.get("EMBED_MODEL", "google/EmbeddingGemma-300m")
EMBED_BATCH_SIZE = 32
EMBED_MAX_LENGTH = 512

LLM_QUERY_REWRITE = _env_bool("LLM_QUERY_REWRITE", True)

ANSWER_MODE = os.environ.get("ANSWER_MODE", "gemma")
GEN_MODEL = os.environ.get("GEN_MODEL", "google/gemma-4-E4B-it")
GEN_DRAFT_MODEL = os.environ.get("GEN_DRAFT_MODEL", "google/gemma-4-E4B-it-assistant")
GEN_MTP_ENABLED = _env_bool("GEN_MTP_ENABLED", True)
GEN_DEVICE_MAP = os.environ.get("GEN_DEVICE_MAP", "auto")
GEN_MAX_NEW_TOKENS = _env_int("GEN_MAX_NEW_TOKENS", 512)
GEN_TEMPERATURE = 0.1
GEN_DO_SAMPLE = False
GEN_MAX_CONTEXT_CHUNKS = 5

NO_RESULT_ANSWER = "資料庫中無相關資訊"
