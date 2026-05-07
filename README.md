# NCHU CS 114-2 NLP Final Project

中興大學 114-2 自然語言處理期末專題：中英文混合課程教材問答檢索系統。

系統會從 `raw_docs/` 的課程講義 PDF 建立索引，使用 BM25 做初步檢索，再用 EmbeddingGemma 做語意重排，最後使用 Gemma 根據檢索到的段落生成簡短答案。

## Features

- 支援中文、英文、以及中英混合查詢
- PDF 逐頁解析並保留來源頁碼
- Sliding window chunking
- BM25 關鍵字初篩
- `google/EmbeddingGemma-300m` 向量重排
- `google/gemma-4-E4B-it` 生成答案
- Streamlit 前端顯示答案與來源段落

## Project Structure

```text
.
├── app.py
├── ingest.py
├── requirements.txt
├── raw_docs/
├── spec/
├── task/
└── src/
    ├── config.py
    ├── pipeline.py
    ├── generation/
    │   └── gemma_generator.py
    ├── ingestion/
    │   ├── chunker.py
    │   └── pdf_parser.py
    └── retrieval/
        ├── bm25_retriever.py
        ├── hybrid_retriever.py
        └── vector_retriever.py
```

Generated artifacts are written to `data/` and are intentionally not committed:

- `data/chunks.jsonl`
- `data/bm25_index.pkl`
- `data/faiss.index`
- `data/chunk_ids.json`

## Requirements

- Python 3.10+
- Hugging Face account and access token
- Access approved for:
  - `google/EmbeddingGemma-300m`
  - `google/gemma-4-E4B-it`
- NVIDIA GPU is recommended for Gemma generation

The project was tested on:

- Windows
- Python 3.10.11
- NVIDIA RTX 4050 Laptop GPU
- PyTorch `2.11.0+cu130`

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If you want GPU inference on a CUDA 13.0 driver, install the CUDA PyTorch wheel:

```powershell
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected GPU output should include `True` for `torch.cuda.is_available()`.

## Hugging Face Login

Before building indexes or running generation, accept the model licenses on Hugging Face:

- <https://huggingface.co/google/embeddinggemma-300m>
- <https://huggingface.co/google/gemma-4-E4B-it>

Then login:

```powershell
huggingface-cli login
```

Use a token with read access to the gated model repositories.

## Build Indexes

Run:

```powershell
python ingest.py
```

This will:

1. Parse PDFs from `raw_docs/`
2. Write chunks to `data/chunks.jsonl`
3. Build BM25 index at `data/bm25_index.pkl`
4. Build FAISS vector index at `data/faiss.index`
5. Write FAISS row mapping to `data/chunk_ids.json`

Incrementally add one PDF:

```powershell
python ingest.py --add raw_docs/new_file.pdf
```

## Run A Query

```powershell
python -c "from src.pipeline import Pipeline; p=Pipeline(); print(p.query('什麼是 Time-homogeneous Markov process？'))"
```

The first run may download model files and take several minutes.

## Performance Options

By default, `Pipeline` uses Gemma generation. If `device_map=auto` offloads Gemma layers to CPU/RAM, generation can be very slow.

On a machine with enough free GPU memory, force Gemma onto GPU:

```bash
GEN_DEVICE_MAP=cuda python3 -c "from src.pipeline import Pipeline; p=Pipeline(); print(p.query('什麼是 Time-homogeneous Markov process？'))"
```

If the GPU does not have enough memory, this mode may fail with CUDA OOM instead of slowly offloading to CPU.

For quick retrieval-only testing without loading Gemma, use extractive answer mode:

```bash
ANSWER_MODE=extractive python3 -c "from src.pipeline import Pipeline; p=Pipeline(); print(p.query('什麼是 Time-homogeneous Markov process？'))"
```

You can also reduce generation length:

```bash
GEN_MAX_NEW_TOKENS=64 python3 -c "from src.pipeline import Pipeline; p=Pipeline(); print(p.query('什麼是 Time-homogeneous Markov process？'))"
```

## Run The UI

```powershell
streamlit run app.py
```

The Streamlit app provides:

- BM25 top-k slider
- Final answer count slider
- Query input
- Generated answer
- Source chunk expander

## Notes

- `data/`, `venv/`, `.env`, caches, and local editor/tool settings are ignored by Git.
- On Windows, Hugging Face may warn that symlink caching is unavailable. This is not fatal; it only uses more disk space.
- If Gemma generation is very slow on a 6GB GPU, it may be offloading layers to CPU/RAM. Retrieval should still work quickly.
- FAISS index reading/writing uses serialized bytes in this project to avoid Windows path issues with non-ASCII directory names.

## Example Questions

- `什麼是 Time-homogeneous Markov process？`
- `Substitution Cipher 的總可能組合數大約是多少？`
- `幾月幾號是期中考？`
