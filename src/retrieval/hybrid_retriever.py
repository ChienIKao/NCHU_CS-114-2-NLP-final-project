import json

from src import config
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever

_RRF_K = 60  # standard constant that dampens the impact of high ranks


def _rrf_score(ranks: list[int]) -> float:
    return sum(1.0 / (_RRF_K + r) for r in ranks)


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever | None = None, vector: VectorRetriever | None = None):
        self.bm25 = bm25 if bm25 is not None else BM25Retriever()
        self.vector = vector if vector is not None else VectorRetriever()

        self._parents: dict[str, dict] = {}
        if config.PARENT_CHUNKS_PATH.exists():
            with config.PARENT_CHUNKS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        self._parents[chunk["id"]] = chunk

    def retrieve(self, query: str, bm25_k: int = config.BM25_TOP_K, final_k: int = config.VECTOR_TOP_K) -> list[dict]:
        bm25_results = self.bm25.retrieve(query, top_k=bm25_k)
        faiss_results = self.vector.retrieve(query, top_k=bm25_k)

        # Collect per-chunk ranks from each system (1-indexed)
        bm25_ranks = {c["id"]: rank for rank, c in enumerate(bm25_results, 1)}
        faiss_ranks = {c["id"]: rank for rank, c in enumerate(faiss_results, 1)}

        # Merge unique chunks, preserve scores for display
        seen: dict[str, dict] = {}
        for chunk in bm25_results + faiss_results:
            cid = chunk["id"]
            if cid not in seen:
                seen[cid] = chunk
            else:
                for score_key in ("bm25_score", "vector_score"):
                    if score_key in chunk and score_key not in seen[cid]:
                        seen[cid][score_key] = chunk[score_key]

        if not seen:
            return []

        # Compute RRF score for each candidate
        for cid, chunk in seen.items():
            ranks = []
            if cid in bm25_ranks:
                ranks.append(bm25_ranks[cid])
            if cid in faiss_ranks:
                ranks.append(faiss_ranks[cid])
            chunk["rrf_score"] = _rrf_score(ranks)

        # Sort by RRF score descending, take top final_k
        ranked = sorted(seen.values(), key=lambda c: c["rrf_score"], reverse=True)[:final_k]

        if self._parents:
            return self._expand_to_parents(ranked)
        return ranked

    def _expand_to_parents(self, small_chunks: list[dict]) -> list[dict]:
        seen_parents: set[str] = set()
        result: list[dict] = []
        for chunk in small_chunks:
            parent_id = chunk.get("parent_id")
            if parent_id and parent_id in self._parents:
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    parent = dict(self._parents[parent_id])
                    for score_key in ("bm25_score", "vector_score", "rrf_score"):
                        if score_key in chunk:
                            parent[score_key] = chunk[score_key]
                    result.append(parent)
            else:
                result.append(chunk)
        return result
