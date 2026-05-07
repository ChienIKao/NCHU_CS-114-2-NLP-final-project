import json
import pickle
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from src import config


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^\w\s一-鿿]", " ", text.lower())
    tokens = list(jieba.cut(text))
    return [token.strip() for token in tokens if token.strip()]


def build_bm25_index(chunks: list[dict], index_path: str | Path = config.BM25_INDEX_PATH) -> None:
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as file:
        pickle.dump(bm25, file)


class BM25Retriever:
    def __init__(
        self,
        index_path: str | Path = config.BM25_INDEX_PATH,
        chunks_path: str | Path = config.CHUNKS_PATH,
    ):
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        with self.index_path.open("rb") as file:
            self.bm25 = pickle.load(file)
        with self.chunks_path.open("r", encoding="utf-8") as file:
            self.chunks = [json.loads(line) for line in file if line.strip()]

    def retrieve(self, query: str, top_k: int = config.BM25_TOP_K) -> list[dict]:
        tokens = tokenize(query)
        if not tokens or not self.chunks:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            chunk = dict(self.chunks[int(idx)])
            chunk["bm25_score"] = score
            results.append(chunk)
        return results
