from src import config
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever | None = None, vector: VectorRetriever | None = None):
        self.bm25 = bm25 if bm25 is not None else BM25Retriever()
        self.vector = vector if vector is not None else VectorRetriever()

    def retrieve(self, query: str, bm25_k: int = config.BM25_TOP_K, final_k: int = config.VECTOR_TOP_K) -> list[dict]:
        candidates = self.bm25.retrieve(query, top_k=bm25_k)
        if not candidates:
            return []
        return self.vector.rerank(query, candidates, top_k=final_k)
