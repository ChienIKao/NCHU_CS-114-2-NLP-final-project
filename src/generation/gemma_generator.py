import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src import config


PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based ONLY on the provided context.
If the answer cannot be found in the context, reply exactly: "資料庫中無相關資訊"
Keep your answer concise (1-3 sentences maximum).

Context:
{context}

Question: {query}

Answer:
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
        return PROMPT_TEMPLATE.format(context=context, query=query)

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
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if len(new_tokens) >= config.GEN_MAX_NEW_TOKENS:
            return f"{answer} [截斷]"
        return answer
