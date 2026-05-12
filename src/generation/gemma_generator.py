import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config


_REWRITE_PROMPT = """\
You are a search query optimizer for an NLP course retrieval system.
Given a question in any language, output ONLY concise English keywords \
suitable for searching English lecture slides. Translate Chinese terms to English.
Output only the keywords on a single line, no explanation.

Examples:
多頭注意力機制 → Multi-Head Attention transformer queries keys values
BERT 是什麼 → BERT bidirectional encoder representations transformers pre-training
梯度消失 → vanishing gradient problem deep learning backpropagation
What is TF-IDF? → TF-IDF term frequency inverse document frequency

Question: {query}
Keywords:"""

SYSTEM_PROMPT = """你是一個課程講義 RAG 聊天機器人。
你的任務是根據檢索到的課程講義片段回答使用者問題。
只能使用提供的 context 內容回答；如果 context 找不到答案，請只回答「資料庫中無相關資訊」。
請用使用者提問的語言回答，答案要簡短精確。
不要輸出思考過程、分析、推理步驟、prompt 內容或 CHUNK 標籤。
不要逐字複製整段 context；只輸出最後答案。
"""


def _log_gpu_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        pct = allocated / total * 100
        print(
            f"[GPU {i}] {label}: "
            f"allocated={allocated:.2f}/{total:.0f} GB ({pct:.1f}%) | "
            f"reserved={reserved:.2f} GB"
        )


def _load_model(model_name: str, device_map) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
    )


class GemmaGenerator:
    def __init__(self, model_name: str = config.GEN_MODEL):
        self.model_name = model_name
        device_map = self._resolve_device_map(config.GEN_DEVICE_MAP)

        _log_gpu_memory("before model load")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = _load_model(model_name, device_map)
        _log_gpu_memory("after target model load")

        self.draft_model: AutoModelForCausalLM | None = None
        if config.GEN_MTP_ENABLED:
            print(f"[MTP] Loading draft model: {config.GEN_DRAFT_MODEL}")
            self.draft_model = _load_model(config.GEN_DRAFT_MODEL, device_map)
            _log_gpu_memory("after draft model load")
        else:
            print("[MTP] Disabled")

    @staticmethod
    def _resolve_device_map(value: str):
        normalized = value.lower()
        if normalized in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError("GEN_DEVICE_MAP=cuda was requested, but torch.cuda.is_available() is False")
            return {"": 0}
        return value

    def _build_prompt(self, query: str, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks[: config.GEN_MAX_CONTEXT_CHUNKS], 1):
            context_parts.append(
                f"[CHUNK {i}] (source: {chunk['source_file']}, p.{chunk['page']})\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\n請只輸出最後答案。"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _clean_answer(self, answer: str) -> str:
        markers = ["Answer:", "Final answer:", "final answer:"]
        for marker in markers:
            if marker in answer:
                answer = answer.split(marker, 1)[1].strip()

        lower_answer = answer.lower()
        if lower_answer.startswith("thought"):
            lines = [line.strip() for line in answer.splitlines() if line.strip()]
            answer_lines = [
                line
                for line in lines
                if not line.lower().startswith(("thought", "analysis", "the user", "i need", "text snippet"))
            ]
            answer = " ".join(answer_lines).strip()

        return answer.strip()

    def rewrite_for_retrieval(self, query: str) -> str:
        """Translate and expand query into English keywords for retrieval."""
        prompt = _REWRITE_PROMPT.format(query=query)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        rewritten = result.splitlines()[0].strip()
        print(f"[QueryRewrite] '{query}' → '{rewritten}'")
        return rewritten or query

    def generate(self, query: str, chunks: list[dict]) -> str:
        if not chunks:
            return config.NO_RESULT_ANSWER
        prompt = self._build_prompt(query, chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": config.GEN_MAX_NEW_TOKENS,
            "do_sample": config.GEN_DO_SAMPLE,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if config.GEN_DO_SAMPLE:
            generation_kwargs["temperature"] = config.GEN_TEMPERATURE
        if self.draft_model is not None:
            generation_kwargs["assistant_model"] = self.draft_model

        import time
        _log_gpu_memory("before generate")
        t0 = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)
        elapsed = time.perf_counter() - t0
        _log_gpu_memory("after generate")

        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        n_new = new_tokens.shape[0]
        mtp_tag = "MTP" if self.draft_model is not None else "no-MTP"
        print(f"[Generate] {mtp_tag}: {n_new} tokens in {elapsed:.2f}s ({n_new / elapsed:.1f} tok/s)")
        answer = self._clean_answer(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return answer
