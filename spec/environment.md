# 環境設定 (Environment)

## 系統需求

| 項目 | 最低需求 | 建議 |
|------|----------|------|
| Python | 3.10+ | 3.11 |
| CUDA | 11.8+ | 12.1+ |
| VRAM | 4GB（4-bit 量化） | 8GB |
| 硬碟空間 | ~10GB（模型快取） | 20GB |
| OS | Linux / Windows + WSL2 | Ubuntu 22.04 |

---

## 1. requirements.txt

```
# PDF 解析
pymupdf>=1.24.0

# 中文斷詞
jieba>=0.42.1

# BM25
rank-bm25>=0.2.2

# 向量搜尋
faiss-cpu>=1.7.4       # 若無 GPU FAISS，用 CPU 版
# faiss-gpu>=1.7.4     # 有 GPU 可改用此版

# Embedding + 生成模型
transformers>=4.40.0
torch>=2.2.0
bitsandbytes>=0.43.0   # 4-bit 量化
accelerate>=0.29.0

# 前端
streamlit>=1.35.0

# 工具
numpy>=1.26.0
tqdm>=4.66.0
```

安裝指令：

```bash
pip install -r requirements.txt
```

PyTorch + CUDA 需單獨安裝（依 CUDA 版本選擇）：

```bash
# CUDA 12.1 範例
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. 專案結構（完整）

```
final_project/
├── spec/                      # 規劃書
├── raw_docs/                  # 原始 PDF（納入 git）
├── data/                      # 索引與處理結果（gitignore）
│   ├── chunks.jsonl
│   ├── bm25_index.pkl
│   ├── faiss.index
│   └── chunk_ids.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py
│   │   └── chunker.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py
│   │   ├── vector_retriever.py
│   │   └── hybrid_retriever.py
│   └── generation/
│       ├── __init__.py
│       └── gemma_generator.py
├── app.py                     # Streamlit 前端
├── ingest.py                  # CLI：建立索引
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 3. .gitignore

```gitignore
# 索引與處理後資料
data/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
*.egg

# HuggingFace 模型快取
~/.cache/huggingface/

# 環境變數
.env

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

---

## 4. .env.example

```env
# HuggingFace Access Token（需同意 Gemma 授權）
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx

# 模型名稱（可覆蓋 config.py 預設值）
# GEN_MODEL=google/gemma-4-E4B-it
# EMBED_MODEL=google/EmbeddingGemma-300m
```

使用前複製：`cp .env.example .env`，填入 Token。  
程式碼中用 `python-dotenv` 載入（或 `os.environ.get()`）。

---

## 5. 快速開始

```bash
# 1. 建立虛擬環境
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. 安裝相依套件
pip install -r requirements.txt

# 3. 設定 HuggingFace Token
cp .env.example .env
# 編輯 .env 填入 HF_TOKEN
huggingface-cli login

# 4. 建立索引（約 2-5 分鐘）
python ingest.py

# 5. 啟動前端
streamlit run app.py
```

---

## 6. Git 分支策略

| 分支 | 用途 |
|------|------|
| `main` | 穩定版本，隨時可 demo |
| `feature/ingestion` | 文件解析與索引建立 |
| `feature/retrieval` | BM25 + 向量檢索 |
| `feature/generation` | Gemma 生成模組 |
| `feature/frontend` | Streamlit 前端 |

**工作流程**：
1. 從 `main` 建立 feature 分支
2. 完成後開 Pull Request → review → merge 回 `main`
3. `main` 只接受可運行的程式碼

---

## 7. 常用指令

```bash
# 建立全量索引
python ingest.py

# 新增單一文件到索引
python ingest.py --add raw_docs/new_file.pdf

# 啟動前端（開發模式，自動 reload）
streamlit run app.py --server.runOnSave true

# 快速測試查詢（不開前端）
python -c "from src.pipeline import Pipeline; p=Pipeline(); print(p.query('什麼是 Markov process？'))"
```
