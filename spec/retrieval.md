# 檢索模組 (Retrieval)

## 架構

混合式兩階段檢索：BM25 快速初篩 → 向量語意重排

```
Query
  │
  ▼
BM25Retriever.retrieve(query, top_k=20)
  │  ← rank_bm25 + jieba/split 斷詞
  │
  ▼
VectorRetriever.rerank(query, candidates, top_k=5)
  │  ← google/EmbeddingGemma-300m + FAISS
  │
  ▼
List[Chunk]  (top-5，依相似度排序)
```

---

## 1. BM25 初篩 — `src/retrieval/bm25_retriever.py`

**套件**：`rank_bm25`（`BM25Okapi`）

```python
class BM25Retriever:
    def __init__(self, index_path: str, chunks_path: str): ...
    def retrieve(self, query: str, top_k: int = 20) -> list[dict]: ...
```

### 索引建立

```python
tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)
pickle.dump(bm25, open("data/bm25_index.pkl", "wb"))
```

### 斷詞函式

```python
def tokenize(text: str) -> list[str]:
    import jieba, re
    # 去除標點符號
    text = re.sub(r"[^\w\s一-鿿]", " ", text.lower())
    # 中文用 jieba，英文用 split
    tokens = list(jieba.cut(text))
    return [t.strip() for t in tokens if t.strip()]
```

查詢也用相同 `tokenize()` 函式處理。

---

## 2. 向量重排 — `src/retrieval/vector_retriever.py`

**Embedding 模型**：`google/EmbeddingGemma-300m`  
**向量索引**：`faiss.IndexFlatIP`（內積，向量需 L2 normalized → 等同 cosine similarity）

```python
class VectorRetriever:
    def __init__(self, model_name: str, index_path: str, chunk_ids_path: str): ...
    def embed(self, texts: list[str]) -> np.ndarray: ...          # shape: (n, dim)
    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]: ...
```

### 索引建立

```python
from transformers import AutoTokenizer, AutoModel
import torch, faiss, numpy as np

model_name = "google/EmbeddingGemma-300m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # L2 normalize
    return embeddings.cpu().numpy()

# 批次處理避免 OOM（batch_size=32）
all_vecs = np.vstack([embed(batch) for batch in batches])
index = faiss.IndexFlatIP(all_vecs.shape[1])
index.add(all_vecs)
faiss.write_index(index, "data/faiss.index")
```

### 重排邏輯

```python
def rerank(self, query, candidates, top_k=5):
    # 只對 BM25 候選 chunks 做向量重排（不是全庫搜尋）
    q_vec = self.embed([query])                          # (1, dim)
    c_vecs = self.embed([c["text"] for c in candidates]) # (20, dim)
    scores = (q_vec @ c_vecs.T).flatten()               # cosine scores
    top_idx = scores.argsort()[::-1][:top_k]
    return [candidates[i] for i in top_idx]
```

---

## 3. 混合入口 — `src/retrieval/hybrid_retriever.py`

```python
class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever): ...

    def retrieve(self, query: str, bm25_k: int = 20, final_k: int = 5) -> list[dict]:
        candidates = self.bm25.retrieve(query, top_k=bm25_k)
        if not candidates:
            return []
        return self.vector.rerank(query, candidates, top_k=final_k)
```

---

## 4. 評估指標

在開發階段，用以下指標驗證檢索品質（使用手標的 QA pairs）：

| 指標 | 說明 |
|------|------|
| `Recall@5` | 正確 chunk 是否出現在 top-5 結果中 |
| `MRR` | Mean Reciprocal Rank，正確 chunk 的倒數排名平均 |

評估腳本：`evaluate_retrieval.py`（可選）

---

## 5. 超參數（在 `src/config.py` 定義）

```python
BM25_TOP_K = 20
VECTOR_TOP_K = 5
EMBED_MODEL = "google/EmbeddingGemma-300m"
EMBED_BATCH_SIZE = 32
EMBED_MAX_LENGTH = 512
```
