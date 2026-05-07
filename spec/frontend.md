# 前端規格 (Frontend)

## 技術

**Streamlit** 單頁應用，入口：`app.py`

```bash
streamlit run app.py
```

---

## 版面配置

```
┌────────────────────────────────────────────────────────────┐
│  側欄 (st.sidebar)          │  主區域                       │
│                             │                               │
│  ⚙️ 設定                    │  🔍 NLP 課程問答系統            │
│  ─────────────────          │                               │
│  BM25 初篩數量: [20]        │  ┌──────────────────────────┐ │
│  最終答案數量:  [5]         │  │ 輸入您的問題...            │ │
│  顯示推論時間: ☑           │  └──────────────────────────┘ │
│                             │  [送出]                       │
│                             │                               │
│                             │  ▶ 答案                       │
│                             │  ┌──────────────────────────┐ │
│                             │  │ When A doesn't depend    │ │
│                             │  │ on t                     │ │
│                             │  └──────────────────────────┘ │
│                             │  推論時間：1.23s               │
│                             │                               │
│                             │  📄 參考來源（展開）           │
│                             │  ├ [CHUNK 1] c2_prob... p.12 │
│                             │  └ [CHUNK 2] c2_prob... p.13 │
└─────────────────────────────┴───────────────────────────────┘
```

---

## 元件規格

### 側欄 (`st.sidebar`)

| 元件 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| BM25 初篩數量 | `st.slider(1, 50)` | 20 | BM25 top-k |
| 最終答案數量 | `st.slider(1, 10)` | 5 | 向量重排 top-k |
| 顯示推論時間 | `st.checkbox` | True | 顯示耗時 |

### 主區域

| 元件 | 說明 |
|------|------|
| `st.title` | "🔍 NLP 課程問答系統" |
| `st.text_input` | 查詢輸入，按 Enter 或點「送出」觸發 |
| `st.button("送出")` | 觸發查詢 |
| `st.success(answer)` | 顯示生成的答案 |
| `st.caption(f"推論時間：{elapsed:.2f}s")` | 條件顯示 |
| `st.expander("📄 參考來源")` | 展開顯示來源 chunks |

### 來源段落展開區（expander 內）

每個 chunk 顯示：
- 標題：`[CHUNK {i}] {source_file}，第 {page} 頁`
- 內容：`st.markdown(chunk["text"])` 

---

## `app.py` 主流程

```python
import streamlit as st
import time
from src.pipeline import Pipeline

@st.cache_resource
def load_pipeline():
    return Pipeline()  # 只載入一次（模型快取）

pipeline = load_pipeline()

st.title("🔍 NLP 課程問答系統")

with st.sidebar:
    bm25_k = st.slider("BM25 初篩數量", 1, 50, 20)
    final_k = st.slider("最終答案數量", 1, 10, 5)
    show_time = st.checkbox("顯示推論時間", value=True)

query = st.text_input("輸入您的問題", placeholder="例：什麼是 Time-homogeneous Markov process？")

if st.button("送出") and query.strip():
    with st.spinner("檢索中..."):
        t0 = time.time()
        result = pipeline.query(query, bm25_k=bm25_k, final_k=final_k)
        elapsed = time.time() - t0

    st.success(result["answer"])
    if show_time:
        st.caption(f"推論時間：{elapsed:.2f}s")

    with st.expander("📄 參考來源"):
        for i, chunk in enumerate(result["chunks"], 1):
            st.markdown(f"**[CHUNK {i}]** `{chunk['source_file']}`，第 {chunk['page']} 頁")
            st.markdown(chunk["text"])
            st.divider()
```

---

## `src/pipeline.py` 介面

```python
class Pipeline:
    def query(self, text: str, bm25_k: int = 20, final_k: int = 5) -> dict:
        """
        回傳：{
            "answer": str,
            "chunks": list[dict]   # top-5 chunks，附 source_file, page, text
        }
        """
```

---

## 錯誤處理

- 查詢為空字串：不觸發查詢（`if query.strip()` 判斷）
- 無檢索結果：`answer = "資料庫中無相關資訊"`，`chunks = []`
- 模型推論異常：`st.error("推論失敗，請稍後再試")` + `st.exception(e)`
