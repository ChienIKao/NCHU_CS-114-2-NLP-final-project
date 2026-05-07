# 系統總覽 (System Overview)

## 專題目標

建立一個支援**中英文混合**查詢的資訊檢索系統，能從課程教材中找到對應的簡短答案。  
評分：80% 自動測資結果 + 20% 書面報告。

---

## 架構圖

```
使用者查詢 (中文 or 英文)
        │
        ▼
┌─────────────────────┐
│   BM25 初篩          │  ← rank_bm25 + jieba 中文斷詞
│   (top-20 chunks)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   向量重排            │  ← google/EmbeddingGemma-300m
│   (top-5 chunks)    │     FAISS IndexFlatIP (cosine)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Gemma 生成答案      │  ← google/gemma-4-E4B-it (4-bit 量化)
│   (簡短精確回答)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Streamlit 前端     │  顯示答案 + 來源段落
└─────────────────────┘
```

---

## 元件責任表

| 元件 | 檔案 | 職責 |
|------|------|------|
| PDF 解析 | `src/ingestion/pdf_parser.py` | 將 PDF 轉為純文字，保留頁碼 |
| 文字切割 | `src/ingestion/chunker.py` | Sliding window 切成 chunks，附 metadata |
| BM25 檢索 | `src/retrieval/bm25_retriever.py` | 關鍵字初篩，回傳 top-20 |
| 向量重排 | `src/retrieval/vector_retriever.py` | EmbeddingGemma 重排，回傳 top-5 |
| 混合檢索入口 | `src/retrieval/hybrid_retriever.py` | 串接 BM25 + 向量重排 |
| 答案生成 | `src/generation/gemma_generator.py` | 讀取 context，呼叫 Gemma 生成答案 |
| 主流程 | `src/pipeline.py` | 串接全部元件，提供 `query(text)` 介面 |
| 設定 | `src/config.py` | 路徑、超參數、模型名稱常數 |
| 索引建立 CLI | `ingest.py` | 一次性（或增量）建立 BM25 + FAISS 索引 |
| 前端 | `app.py` | Streamlit 應用，呼叫 `pipeline.query()` |

---

## 資料流

```
raw_docs/*.pdf
    │  (ingest.py)
    ▼
data/chunks.jsonl          # {id, source_file, page, text, language}
    │
    ├─► data/bm25_index.pkl     # BM25 序列化索引
    └─► data/faiss.index        # FAISS 向量索引
              + data/chunk_ids.json  # index → chunk id 對照
```

---

## 實作順序

依照下列順序實作，每個階段完成後可獨立驗收，避免整合時出現大量未知問題。

### 階段 1 — 設定與資料處理

> 目標：能從 PDF 產出 `data/chunks.jsonl`

| 步驟 | 檔案 | 驗收方式 |
|------|------|----------|
| 1-1 | `src/config.py` | 確認所有路徑常數可正確 import |
| 1-2 | `src/ingestion/pdf_parser.py` | `parse_pdf("raw_docs/c0_course_introduction.pdf")` 回傳非空 list |
| 1-3 | `src/ingestion/chunker.py` | 單頁文字切出數個 chunk，id 格式正確 |
| 1-4 | `ingest.py`（僅 chunking 部分） | 執行後 `data/chunks.jsonl` 存在且行數 > 100 |

### 階段 2 — BM25 索引與檢索

> 目標：`BM25Retriever` 能對查詢回傳 top-20 候選

| 步驟 | 檔案 | 驗收方式 |
|------|------|----------|
| 2-1 | `src/retrieval/bm25_retriever.py` | — |
| 2-2 | `ingest.py`（加入 BM25 建索引） | `data/bm25_index.pkl` 存在 |
| 2-3 | 手動測試 | `BM25Retriever().retrieve("Markov process", top_k=5)` 回傳 5 筆含相關文字的 chunks |

### 階段 3 — 向量索引與重排

> 目標：`VectorRetriever` 能對 BM25 候選做語意重排

| 步驟 | 檔案 | 驗收方式 |
|------|------|----------|
| 3-1 | `src/retrieval/vector_retriever.py` | — |
| 3-2 | `ingest.py`（加入 FAISS 建索引） | `data/faiss.index` 與 `data/chunk_ids.json` 存在 |
| 3-3 | 手動測試 | `rerank("Markov process", candidates)` 回傳 5 筆，順序與 BM25 不同（語意更相關在前） |

### 階段 4 — 混合檢索

| 步驟 | 檔案 | 驗收方式 |
|------|------|----------|
| 4-1 | `src/retrieval/hybrid_retriever.py` | `HybridRetriever().retrieve("Markov process")` 端對端回傳 5 筆 |

### 階段 5 — Gemma 答案生成

> 注意：需先完成 `huggingface-cli login` 並同意 Gemma 授權

| 步驟 | 檔案 | 驗收方式 |
|------|------|----------|
| 5-1 | `src/generation/gemma_generator.py` | `GemmaGenerator().generate("Markov?", chunks)` 回傳非空字串，不含 prompt 本身 |

### 階段 6 — 主流程串接

| 步驟 | 檔案 | 驗收方式 |
|------|------|----------|
| 6-1 | `src/pipeline.py` | `Pipeline().query("什麼是 Markov process？")` 回傳 `{"answer": ..., "chunks": [...]}` |
| 6-2 | 三個驗收查詢 | 見「驗收查詢範例」，答案語意正確 |

### 階段 7 — Streamlit 前端

| 步驟 | 檔案 | 驗收方式 |
|------|------|----------|
| 7-1 | `app.py` | `streamlit run app.py` 啟動無錯誤，瀏覽器可看到 UI |
| 7-2 | 端對端操作 | 在 UI 輸入查詢，能顯示答案與來源段落 |

---

## 驗收查詢範例

| 查詢 | 預期答案 |
|------|----------|
| 什麼是 Time-homogeneous Markov process？ | When A doesn't depend on t |
| Substitution Cipher 的總可能組合數大約是多少？ | 26! |
| 幾月幾號是期中考？ | 2026/4/22 |
