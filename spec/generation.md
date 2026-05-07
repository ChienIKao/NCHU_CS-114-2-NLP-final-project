# 答案生成模組 (Generation)

## 模型

| 項目 | 值 |
|------|-----|
| 模型 | `google/gemma-4-E4B-it` |
| 量化 | 4-bit（`bitsandbytes` NF4） |
| 推論裝置 | CUDA GPU |
| 預估 VRAM | ~4GB（4-bit 量化後） |

---

## 1. 模型載入 — `src/generation/gemma_generator.py`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class GemmaGenerator:
    def __init__(self, model_name: str = "google/gemma-4-E4B-it"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

    def generate(self, query: str, chunks: list[dict]) -> str:
        prompt = self._build_prompt(query, chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # 只回傳新生成的部分
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

---

## 2. Prompt 模板

中英文共用同一模板，依查詢語言自動切換指令語言。

```
You are a helpful assistant. Answer the question based ONLY on the provided context.
If the answer cannot be found in the context, reply exactly: "資料庫中無相關資訊"
Keep your answer concise (1-3 sentences maximum).

Context:
[CHUNK 1] (source: {source_file}, p.{page})
{text}

[CHUNK 2] (source: {source_file}, p.{page})
{text}

...（最多 5 個 chunks）

Question: {query}

Answer:
```

### 實作細節

```python
def _build_prompt(self, query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks[:5], 1):
        context_parts.append(
            f"[CHUNK {i}] (source: {chunk['source_file']}, p.{chunk['page']})\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format(context=context, query=query)
```

---

## 3. 安全機制

- 若 `chunks` 為空（BM25 完全沒有命中），直接回傳 `"資料庫中無相關資訊"` 不呼叫模型
- 若模型生成結果包含 hallucination 的跡象（生成長度超過 256 tokens），截斷並加上 `[截斷]` 提示
- `temperature=0.1, do_sample=False`：確保答案穩定、可重現

---

## 4. 超參數（在 `src/config.py` 定義）

```python
GEN_MODEL = "google/gemma-4-E4B-it"
GEN_MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 0.1
GEN_DO_SAMPLE = False
GEN_MAX_CONTEXT_CHUNKS = 5
```

---

## 5. 首次啟動說明

Gemma 模型需登入 HuggingFace 並同意使用授權：

```bash
huggingface-cli login
# 輸入 HuggingFace Access Token（需在 HF 上同意 Gemma 授權）
```

模型會快取於 `~/.cache/huggingface/hub/`（已加入 `.gitignore`）。
