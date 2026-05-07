from src import config
from src.generation.gemma_generator import GemmaGenerator
from src.retrieval.hybrid_retriever import HybridRetriever


class Pipeline:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: GemmaGenerator | None = None,
    ):
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.generator = generator if generator is not None else GemmaGenerator()

    def query(self, text: str, bm25_k: int = config.BM25_TOP_K, final_k: int = config.VECTOR_TOP_K) -> dict:
        query_text = text.strip()
        if not query_text:
            return {"answer": config.NO_RESULT_ANSWER, "chunks": []}
        chunks = self.retriever.retrieve(query_text, bm25_k=bm25_k, final_k=final_k)
        if not chunks:
            return {"answer": config.NO_RESULT_ANSWER, "chunks": []}
        answer = self.generator.generate(query_text, chunks)
        return {"answer": answer, "chunks": chunks}
