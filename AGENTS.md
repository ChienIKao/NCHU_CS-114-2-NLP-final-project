# AGENTS.md

Guidance for Codex when working in this repository.

## Project

This is an NCHU NLP course final project: a Chinese/English information retrieval QA system over course lecture PDFs.

Primary goal: given a Chinese or English query, retrieve relevant passages from `raw_docs/` and generate a concise answer with cited source chunks.

Grading is mainly automated retrieval/answer correctness, so prioritize correctness, reproducibility, and simple end-to-end behavior over broad abstractions.

## Current State

Implementation code has not been created yet. The repo currently contains:

- `CLAUDE.md`: existing project guidance.
- `spec/`: architecture and implementation specs.
- `raw_docs/`: source lecture PDFs, `c0` through `c7`.
- `task/final_project.pdf`: assignment statement.

There is currently no `.git` repository in this workspace.

## Intended Architecture

Build the system in this shape:

```text
raw_docs/*.pdf
  -> src/ingestion/pdf_parser.py
  -> src/ingestion/chunker.py
  -> data/chunks.jsonl
  -> BM25 index + FAISS/vector artifacts
  -> HybridRetriever
  -> GemmaGenerator
  -> Pipeline.query()
  -> Streamlit app.py
```

Expected files:

- `src/config.py`: paths, model names, and hyperparameters.
- `src/ingestion/pdf_parser.py`: parse PDFs page by page with PyMuPDF.
- `src/ingestion/chunker.py`: sliding-window chunking with metadata.
- `src/retrieval/bm25_retriever.py`: BM25 keyword retrieval using `rank_bm25` and `jieba`.
- `src/retrieval/vector_retriever.py`: EmbeddingGemma reranking.
- `src/retrieval/hybrid_retriever.py`: BM25 candidate retrieval plus vector rerank.
- `src/generation/gemma_generator.py`: Gemma answer generation from top chunks.
- `src/pipeline.py`: public `Pipeline.query(text, bm25_k=20, final_k=5)` interface.
- `ingest.py`: builds `data/chunks.jsonl`, `data/bm25_index.pkl`, `data/faiss.index`, and `data/chunk_ids.json`.
- `app.py`: Streamlit UI.

## Core Interfaces

`parse_pdf(path: str) -> list[dict]`

- Return one dict per page: `{source_file, page, text}`.
- Preserve page numbers.
- Clean obvious headers/footers and repair English hyphen line breaks.

`chunk_page(page_dict: dict, chunk_size=500, overlap=100) -> list[dict]`

- Return chunks with `{id, source_file, page, text, language}`.
- Chunk id format: `{filename_stem}_{page:04d}_{chunk_idx:04d}`.
- Language is `zh` if Chinese character ratio is over 20%, else `en`.
- Prefer sentence punctuation boundaries when splitting.

`Pipeline.query(text: str, bm25_k: int = 20, final_k: int = 5) -> dict`

- Return `{"answer": str, "chunks": list[dict]}`.
- If retrieval finds nothing, return answer exactly: `資料庫中無相關資訊`.

## Models And Retrieval

Default retrieval flow:

1. BM25 initial candidate retrieval, default top 20.
2. Vector reranking with `google/EmbeddingGemma-300m`, default top 5.
3. Answer generation with `google/gemma-4-E4B-it`, 4-bit quantized when available.

Keep model names and tunables in `src/config.py`.

Important constants from the specs:

- `BM25_TOP_K = 20`
- `VECTOR_TOP_K = 5`
- `CHUNK_SIZE = 500`
- `CHUNK_OVERLAP = 100`
- `EMBED_MODEL = "google/EmbeddingGemma-300m"`
- `GEN_MODEL = "google/gemma-4-E4B-it"`
- `GEN_MAX_NEW_TOKENS = 256`
- `GEN_TEMPERATURE = 0.1`

## Frontend

Use Streamlit in `app.py`.

UI requirements:

- Title: `NLP 課程問答系統`.
- Sidebar controls:
  - BM25 initial top-k slider, range 1-50, default 20.
  - Final result count slider, range 1-10, default 5.
  - Show inference time checkbox, default true.
- Main query input with submit button.
- Show generated answer and expandable source chunks.
- Cache the pipeline with `st.cache_resource`.

## Data Artifacts

Generated files should live under `data/` and should not be committed:

- `data/chunks.jsonl`
- `data/bm25_index.pkl`
- `data/faiss.index`
- `data/chunk_ids.json`

Create `.gitignore` when implementation begins and include `data/`, virtualenvs, Python caches, `.env`, IDE files, and OS files.

## Development Order

Prefer implementing in stages:

1. Project scaffold, requirements, config, and ingestion.
2. Chunk generation to `data/chunks.jsonl`.
3. BM25 indexing and retrieval.
4. Vector reranking.
5. Hybrid retrieval.
6. Gemma generation.
7. `Pipeline.query`.
8. Streamlit UI.

At each stage, add a small manual or automated validation path before moving on.

Useful acceptance checks from the specs:

- `parse_pdf("raw_docs/c0_course_introduction.pdf")` returns a non-empty page list.
- `python ingest.py` creates `data/chunks.jsonl` with more than 100 rows.
- `BM25Retriever(...).retrieve("Markov process", top_k=5)` returns relevant chunks.
- `Pipeline().query("什麼是 Markov process？")` returns an answer and source chunks.

## Known Example Queries

- `什麼是 Time-homogeneous Markov process？`
  - Expected answer meaning: transition matrix `A` does not depend on `t`.
- `Substitution Cipher 的總可能組合數大約是多少？`
  - Expected answer: `26!`.
- `幾月幾號是期中考？`
  - Expected answer: `2026/4/22`.

## Implementation Notes

- Prefer simple, inspectable Python.
- Use structured parsing and JSONL rather than ad hoc plain text dumps.
- Keep heavyweight model loading lazy or cached where practical.
- Make CPU fallbacks possible for retrieval development, even if Gemma generation needs CUDA for practical speed.
- Avoid downloading models or installing packages unless the user approves network access.
- Do not remove or rewrite the provided specs or raw PDFs.
