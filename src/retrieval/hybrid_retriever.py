from src import config
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever | None = None, vector: VectorRetriever | None = None):
        self.bm25 = bm25 if bm25 is not None else BM25Retriever()
        self.vector = vector if vector is not None else VectorRetriever()

    def retrieve(self, query: str, bm25_k: int = config.BM25_TOP_K, final_k: int = config.VECTOR_TOP_K) -> list[dict]:
        bm25_results = self.bm25.retrieve(query, top_k=bm25_k)
        faiss_results = self.vector.retrieve(query, top_k=bm25_k)

        # Merge and deduplicate by chunk id, preferring the copy that already has a score
        seen: dict[str, dict] = {}
        for chunk in bm25_results + faiss_results:
            cid = chunk["id"]
            if cid not in seen:
                seen[cid] = chunk
            else:
                # Keep both scores if the other path added one
                if "bm25_score" in chunk and "bm25_score" not in seen[cid]:
                    seen[cid]["bm25_score"] = chunk["bm25_score"]
                if "vector_score" in chunk and "vector_score" not in seen[cid]:
                    seen[cid]["vector_score"] = chunk["vector_score"]

        candidates = list(seen.values())
        if not candidates:
            return []
        return self.vector.rerank(query, candidates, top_k=final_k)
