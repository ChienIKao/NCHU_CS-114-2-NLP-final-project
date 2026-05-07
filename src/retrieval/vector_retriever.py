import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src import config


def _batches(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _read_faiss_index(path: str | Path):
    data = np.frombuffer(Path(path).read_bytes(), dtype="uint8")
    return faiss.deserialize_index(data)


def _write_faiss_index(index, path: str | Path) -> None:
    serialized = faiss.serialize_index(index)
    Path(path).write_bytes(serialized.tobytes())


class VectorRetriever:
    def __init__(
        self,
        model_name: str = config.EMBED_MODEL,
        index_path: str | Path = config.FAISS_INDEX_PATH,
        chunk_ids_path: str | Path = config.CHUNK_IDS_PATH,
    ):
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.chunk_ids_path = Path(chunk_ids_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.index = _read_faiss_index(self.index_path) if self.index_path.exists() else None
        if self.chunk_ids_path.exists():
            with self.chunk_ids_path.open("r", encoding="utf-8") as file:
                self.chunk_ids = json.load(file)
        else:
            self.chunk_ids = []

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        vectors = []
        for batch in _batches(texts, config.EMBED_BATCH_SIZE):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.EMBED_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            vectors.append(embeddings.cpu().numpy().astype("float32"))
        return np.vstack(vectors)

    def rerank(self, query: str, candidates: list[dict], top_k: int = config.VECTOR_TOP_K) -> list[dict]:
        if not candidates:
            return []
        q_vec = self.embed([query])
        c_vecs = self.embed([candidate["text"] for candidate in candidates])
        scores = (q_vec @ c_vecs.T).flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            chunk = dict(candidates[int(idx)])
            chunk["vector_score"] = float(scores[int(idx)])
            results.append(chunk)
        return results


def build_faiss_index(
    chunks: list[dict],
    model_name: str = config.EMBED_MODEL,
    index_path: str | Path = config.FAISS_INDEX_PATH,
    chunk_ids_path: str | Path = config.CHUNK_IDS_PATH,
) -> None:
    retriever = VectorRetriever(model_name=model_name, index_path=index_path, chunk_ids_path=chunk_ids_path)
    texts = [chunk["text"] for chunk in chunks]
    vectors = retriever.embed(texts)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path = Path(index_path)
    chunk_ids_path = Path(chunk_ids_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    _write_faiss_index(index, index_path)
    with chunk_ids_path.open("w", encoding="utf-8") as file:
        json.dump([chunk["id"] for chunk in chunks], file, ensure_ascii=False, indent=2)


def append_faiss_index(
    chunks: list[dict],
    model_name: str = config.EMBED_MODEL,
    index_path: str | Path = config.FAISS_INDEX_PATH,
    chunk_ids_path: str | Path = config.CHUNK_IDS_PATH,
) -> None:
    if not chunks:
        return

    index_path = Path(index_path)
    chunk_ids_path = Path(chunk_ids_path)
    if not index_path.exists() or not chunk_ids_path.exists():
        build_faiss_index(chunks, model_name=model_name, index_path=index_path, chunk_ids_path=chunk_ids_path)
        return

    retriever = VectorRetriever(model_name=model_name, index_path=index_path, chunk_ids_path=chunk_ids_path)
    vectors = retriever.embed([chunk["text"] for chunk in chunks])
    index = _read_faiss_index(index_path)
    index.add(vectors)

    with chunk_ids_path.open("r", encoding="utf-8") as file:
        chunk_ids = json.load(file)
    chunk_ids.extend(chunk["id"] for chunk in chunks)

    _write_faiss_index(index, index_path)
    with chunk_ids_path.open("w", encoding="utf-8") as file:
        json.dump(chunk_ids, file, ensure_ascii=False, indent=2)
