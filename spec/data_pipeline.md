# 資料處理流程 (Data Pipeline)

## 責任

將 `raw_docs/` 中的 PDF 檔案解析、切割成可檢索的 chunks，並存成後續建索引用的格式。  
入口指令：`python ingest.py`

---

## 1. PDF 解析 — `src/ingestion/pdf_parser.py`

**使用套件**：`pymupdf`（`import fitz`）

```python
def parse_pdf(path: str) -> list[dict]:
    """回傳 [{page: int, text: str}, ...]"""
```

- 逐頁擷取文字，保留頁碼
- 移除頁眉/頁腳（通常是單行、全大寫或含頁碼數字，可用簡單正規表達式過濾）
- 合併跨行斷字（英文單字以 `-\n` 結尾者）
- 輸出每頁一個 dict：`{source_file, page, text}`

---

## 2. 文字切割 — `src/ingestion/chunker.py`

**策略**：Sliding Window（以字元數計算，非 token 數，避免依賴 tokenizer）

| 參數 | 值 | 說明 |
|------|-----|------|
| `chunk_size` | 500 chars | 每個 chunk 的最大字元數 |
| `overlap` | 100 chars | 相鄰 chunk 的重疊字元數 |

```python
def chunk_page(page_dict: dict, chunk_size=500, overlap=100) -> list[dict]:
    """將單頁文字切成 chunks，回傳 [{id, source_file, page, text, language}, ...]"""
```

- `id`：格式為 `{filename_stem}_{page:04d}_{chunk_idx:04d}`
- `language`：偵測方式——若中文字元比例 > 20% 則標為 `zh`，否則 `en`（用 `unicodedata.category(c) == 'Lo'` 計算）
- 切割邊界優先對齊句號 `。.!?` 等標點

---

## 3. 資料格式

**輸出檔案**：`data/chunks.jsonl`（每行一個 JSON 物件）

```jsonc
{
  "id": "c2_probabilistic_models_0012_0003",
  "source_file": "c2_probabilistic_models.pdf",
  "page": 12,
  "text": "A Time-homogeneous Markov process is one where the transition matrix A does not depend on t...",
  "language": "en"
}
```

---

## 4. 中英文斷詞

| 語言 | 斷詞方式 | 用途 |
|------|----------|------|
| 中文 | `jieba.cut(text)` | BM25 token |
| 英文 | `text.lower().split()` | BM25 token |

BM25 索引只用斷詞結果；向量模型直接使用原始 `text`（EmbeddingGemma 自帶 tokenizer）。

---

## 5. 增量新增文件

`ingest.py` 支援兩種模式：

```bash
# 全量重建（清除舊索引）
python ingest.py

# 增量新增單一檔案
python ingest.py --add path/to/new_document.pdf
```

增量模式流程：
1. 讀取現有 `data/chunks.jsonl`（append 新 chunks）
2. 重建 BM25 索引（pkl 序列化）
3. 只對新 chunks 計算向量，append 至 FAISS index（`index.add()`）
4. 更新 `data/chunk_ids.json`（FAISS row idx → chunk id 映射）

---

## 6. `ingest.py` 主流程

```python
# 虛擬碼
for pdf_file in raw_docs_dir.glob("*.pdf"):
    pages = parse_pdf(pdf_file)
    for page in pages:
        chunks.extend(chunk_page(page))

save_jsonl(chunks, "data/chunks.jsonl")
build_bm25_index(chunks)       # → data/bm25_index.pkl
build_faiss_index(chunks)      # → data/faiss.index + data/chunk_ids.json
```

執行時間預估（9 份 PDF、~200 頁）：BM25 < 5s，FAISS embedding ~2 分鐘（GPU）。
