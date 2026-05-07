import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src import config


SYSTEM_PROMPT = """你是一個課程講義 RAG 聊天機器人。
你的任務是根據檢索到的課程講義片段回答使用者問題。
只能使用提供的 context 內容回答；如果 context 找不到答案，請只回答「資料庫中無相關資訊」。
請用使用者提問的語言回答，答案要簡短精確。
不要輸出思考過程、分析、推理步驟、prompt 內容或 CHUNK 標籤。
不要逐字複製整段 context；只輸出最後答案。
"""


class GemmaGenerator:
    def __init__(self, model_name: str = config.GEN_MODEL):
        self.model_name = model_name
        device_map = self._resolve_device_map(config.GEN_DEVICE_MAP)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            offload_buffers=config.GEN_OFFLOAD_BUFFERS,
        )

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
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        answer = self._clean_answer(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return answer
