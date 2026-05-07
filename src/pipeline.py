from src import config
from src.generation.gemma_generator import GemmaGenerator
from src.retrieval.hybrid_retriever import HybridRetriever


class Pipeline:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: GemmaGenerator | None = None,
        answer_mode: str | None = None,
    ):
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.generator = generator
        self.answer_mode = answer_mode or config.ANSWER_MODE

    def _get_generator(self) -> GemmaGenerator:
        if self.generator is None:
            self.generator = GemmaGenerator()
        return self.generator

    def _extractive_answer(self, chunks: list[dict]) -> str:
        if not chunks:
            return config.NO_RESULT_ANSWER
        return chunks[0]["text"].strip()[:500].strip()

    def query(self, text: str, bm25_k: int = config.BM25_TOP_K, final_k: int = config.VECTOR_TOP_K) -> dict:
        query_text = text.strip()
        if not query_text:
            return {"answer": config.NO_RESULT_ANSWER, "chunks": []}
        chunks = self.retriever.retrieve(query_text, bm25_k=bm25_k, final_k=final_k)
        if not chunks:
            return {"answer": config.NO_RESULT_ANSWER, "chunks": []}
        if self.answer_mode.lower() == "extractive":
            answer = self._extractive_answer(chunks)
        else:
            answer = self._get_generator().generate(query_text, chunks)
        return {"answer": answer, "chunks": chunks}
