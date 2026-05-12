import re

from src import config
from src.retrieval.hybrid_retriever import HybridRetriever

# Instruction verbs/phrases that carry no retrieval signal
_INSTRUCTION_RE = re.compile(
    r"^("
    r"詳細講解|詳細說明|詳細解釋|詳細描述|詳細介紹|"
    r"簡單說明|簡單解釋|簡單介紹|"
    r"講解|解釋|說明|介紹|描述|定義|"
    r"幫我(了解|說明|解釋|講解|介紹|描述)?|"
    r"請(問|解釋|說明|講解|介紹|描述|定義)?|"
    r"告訴我|"
    r"please\s+(explain|describe|define|tell\s+me\s+about)|"
    r"explain|describe|define"
    r")\s*",
    re.IGNORECASE,
)


class Pipeline:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator=None,
        answer_mode: str | None = None,
    ):
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.generator = generator
        self.answer_mode = answer_mode or config.ANSWER_MODE

    def _get_generator(self):
        if self.generator is None:
            from src.generation.gemma_generator import GemmaGenerator

            self.generator = GemmaGenerator()
        return self.generator

    @staticmethod
    def _rewrite_query(text: str) -> str:
        """Strip instruction prefixes to keep only the semantic core for retrieval."""
        cleaned = text.strip()
        prev = None
        while prev != cleaned:
            prev = cleaned
            cleaned = _INSTRUCTION_RE.sub("", cleaned).strip()
        return cleaned or text.strip()

    def _extractive_answer(self, chunks: list[dict]) -> str:
        if not chunks:
            return config.NO_RESULT_ANSWER
        return chunks[0]["text"].strip()[:500].strip()

    def query(
        self,
        text: str,
        bm25_k: int = config.BM25_TOP_K,
        final_k: int = config.VECTOR_TOP_K,
        answer_mode: str | None = None,
    ) -> dict:
        query_text = text.strip()
        if not query_text:
            return {"answer": config.NO_RESULT_ANSWER, "chunks": []}
        retrieval_query = self._rewrite_query(query_text)
        chunks = self.retriever.retrieve(retrieval_query, bm25_k=bm25_k, final_k=final_k)
        if not chunks:
            return {"answer": config.NO_RESULT_ANSWER, "chunks": []}
        mode = (answer_mode or self.answer_mode).lower()
        if mode == "extractive":
            answer = self._extractive_answer(chunks)
        else:
            answer = self._get_generator().generate(query_text, chunks)
        return {"answer": answer, "chunks": chunks}
