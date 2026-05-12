import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src import config
from src.ingestion.chunker import chunk_document
from src.ingestion.pdf_parser import parse_pdf
from src.retrieval.bm25_retriever import build_bm25_index
from src.retrieval.vector_retriever import VectorRetriever, append_faiss_index, build_faiss_index


def load_chunks(path: Path = config.CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_jsonl(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def parse_and_chunk(pdf_files: list[Path], embed_fn=None) -> tuple[list[dict], list[dict]]:
    """Return (small_chunks, parent_chunks).

    When semantic chunking is off, parent_chunks is empty and small_chunks
    are the regular sliding-window chunks.
    """
    small_chunks: list[dict] = []
    parent_chunks: list[dict] = []
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        pages = parse_pdf(str(pdf_file))
        if config.SEMANTIC_CHUNKING and embed_fn is not None:
            from src.ingestion.semantic_chunker import semantic_chunk_document_with_parents
            s, p = semantic_chunk_document_with_parents(pages, embed_fn)
            small_chunks.extend(s)
            parent_chunks.extend(p)
        else:
            small_chunks.extend(
                chunk_document(pages, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
            )
    return small_chunks, parent_chunks


def build_indexes(chunks: list[dict]) -> None:
    build_bm25_index(chunks, config.BM25_INDEX_PATH)
    build_faiss_index(chunks, config.EMBED_MODEL, config.FAISS_INDEX_PATH, config.CHUNK_IDS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indexes from lecture PDFs.")
    parser.add_argument("--add", type=Path, help="Incrementally add one PDF file.")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    embed_fn = None
    if config.SEMANTIC_CHUNKING:
        print(f"[Ingest] Semantic chunking enabled — loading embed model: {config.EMBED_MODEL}")
        retriever = VectorRetriever(
            model_name=config.EMBED_MODEL,
            index_path=config.FAISS_INDEX_PATH,
            chunk_ids_path=config.CHUNK_IDS_PATH,
        )
        embed_fn = retriever.embed

    if args.add:
        existing_small = load_chunks(config.CHUNKS_PATH)
        existing_parents = load_chunks(config.PARENT_CHUNKS_PATH)
        new_small, new_parents = parse_and_chunk([args.add], embed_fn=embed_fn)
        small_chunks = existing_small + new_small
        parent_chunks = existing_parents + new_parents
        save_jsonl(small_chunks, config.CHUNKS_PATH)
        if parent_chunks:
            save_jsonl(parent_chunks, config.PARENT_CHUNKS_PATH)
        build_bm25_index(small_chunks, config.BM25_INDEX_PATH)
        if config.FAISS_INDEX_PATH.exists() and config.CHUNK_IDS_PATH.exists():
            append_faiss_index(new_small, config.EMBED_MODEL, config.FAISS_INDEX_PATH, config.CHUNK_IDS_PATH)
        else:
            build_faiss_index(small_chunks, config.EMBED_MODEL, config.FAISS_INDEX_PATH, config.CHUNK_IDS_PATH)
    else:
        pdf_files = sorted(config.RAW_DOCS_DIR.glob("*.pdf"))
        small_chunks, parent_chunks = parse_and_chunk(pdf_files, embed_fn=embed_fn)
        save_jsonl(small_chunks, config.CHUNKS_PATH)
        if parent_chunks:
            save_jsonl(parent_chunks, config.PARENT_CHUNKS_PATH)
        build_indexes(small_chunks)


if __name__ == "__main__":
    main()
