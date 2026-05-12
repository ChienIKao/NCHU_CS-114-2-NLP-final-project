import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src import config
from src.ingestion.chunker import chunk_document
from src.ingestion.pdf_parser import parse_pdf
from src.retrieval.bm25_retriever import build_bm25_index
from src.retrieval.vector_retriever import append_faiss_index, build_faiss_index


def load_chunks(path: Path = config.CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_jsonl(chunks: list[dict], path: Path = config.CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def parse_and_chunk(pdf_files: list[Path]) -> list[dict]:
    chunks = []
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        pages = parse_pdf(str(pdf_file))
        chunks.extend(chunk_document(pages, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP))
    return chunks


def build_indexes(chunks: list[dict]) -> None:
    build_bm25_index(chunks, config.BM25_INDEX_PATH)
    build_faiss_index(chunks, config.EMBED_MODEL, config.FAISS_INDEX_PATH, config.CHUNK_IDS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indexes from lecture PDFs.")
    parser.add_argument("--add", type=Path, help="Incrementally add one PDF file.")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.add:
        existing_chunks = load_chunks()
        new_chunks = parse_and_chunk([args.add])
        chunks = existing_chunks + new_chunks
        save_jsonl(chunks)
        build_bm25_index(chunks, config.BM25_INDEX_PATH)
        if config.FAISS_INDEX_PATH.exists() and config.CHUNK_IDS_PATH.exists():
            append_faiss_index(new_chunks, config.EMBED_MODEL, config.FAISS_INDEX_PATH, config.CHUNK_IDS_PATH)
        else:
            build_faiss_index(chunks, config.EMBED_MODEL, config.FAISS_INDEX_PATH, config.CHUNK_IDS_PATH)
    else:
        pdf_files = sorted(config.RAW_DOCS_DIR.glob("*.pdf"))
        chunks = parse_and_chunk(pdf_files)
        save_jsonl(chunks)
        build_indexes(chunks)


if __name__ == "__main__":
    main()
