"""Semantic chunking: split documents at points where embedding similarity drops."""
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np

from src import config
from src.ingestion.chunker import detect_language


# Sentence boundary: Chinese endings, English period/!/?
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|(?<=\n)\s*")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping each non-empty."""
    raw = _SENTENCE_SPLIT_RE.split(text)
    sentences = [s.strip() for s in raw if s.strip()]
    # Merge short fragments into the previous sentence to avoid embedding noise
    # from PDF line-break artefacts (e.g. "independently?" orphaned from the line above)
    merged: list[str] = []
    for s in sentences:
        if merged and len(s) < 40:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def _cosine_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Return cosine similarity between consecutive sentence embeddings."""
    # embeddings are already L2-normalised by VectorRetriever.embed
    sims = (embeddings[:-1] * embeddings[1:]).sum(axis=1)
    return sims


def _find_breakpoints(sims: np.ndarray, percentile: int) -> list[int]:
    """Return sentence indices AFTER which a new chunk should start."""
    if len(sims) == 0:
        return []
    threshold = float(np.percentile(sims, 100 - percentile))
    return [i for i, s in enumerate(sims) if s < threshold]


def _carry_sentences(parts: list[str], ratio: float) -> list[str]:
    """Return the tail sentences to carry into the next chunk."""
    if ratio <= 0 or not parts:
        return []
    n = max(1, math.ceil(len(parts) * ratio))
    return parts[-n:]


def _group_sentences(
    sentences: list[str], breakpoints: set[int], max_chars: int, min_chars: int,
    overlap_ratio: float,
) -> list[str]:
    """Collect sentences into chunk texts, respecting breakpoints and size bounds.

    The trailing `overlap_ratio` fraction of each chunk's sentences is carried
    over as the opening of the next chunk.
    """
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def flush():
        if current_parts:
            chunks.append(" ".join(current_parts))

    for i, sentence in enumerate(sentences):
        # Hard size cap: flush and carry tail sentences into next chunk
        if current_len + len(sentence) > max_chars and current_parts:
            flush()
            carry = _carry_sentences(current_parts, overlap_ratio)
            current_parts.clear()
            current_parts.extend(carry)
            current_len = sum(len(s) for s in carry)

        current_parts.append(sentence)
        current_len += len(sentence)

        # Semantic breakpoint: flush and carry tail sentences
        if i in breakpoints:
            flush()
            carry = _carry_sentences(current_parts, overlap_ratio)
            current_parts.clear()
            current_parts.extend(carry)
            current_len = sum(len(s) for s in carry)

    flush()

    # Merge chunks that are too short into the previous one
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged


def _split_to_small(
    parent: dict, max_chars: int, overlap_ratio: float
) -> list[dict]:
    """Split a parent chunk into small retrieval chunks, each carrying parent_id."""
    sentences = _split_sentences(parent["text"])
    current_parts: list[str] = []
    current_len = 0
    small_texts: list[str] = []

    for sent in sentences:
        if current_len + len(sent) > max_chars and current_parts:
            small_texts.append(" ".join(current_parts))
            carry = _carry_sentences(current_parts, overlap_ratio)
            current_parts = list(carry)
            current_len = sum(len(s) for s in carry)
        current_parts.append(sent)
        current_len += len(sent)
    if current_parts:
        small_texts.append(" ".join(current_parts))

    result = []
    for i, text in enumerate(small_texts):
        if not text.strip():
            continue
        result.append(
            {
                "id": f"{parent['id']}_s{i:03d}",
                "parent_id": parent["id"],
                "source_file": parent["source_file"],
                "page": parent["page"],
                "text": text,
                "language": parent["language"],
            }
        )
    return result


def _build_page_helpers(pages: list[dict]):
    """Return (full_text, page_boundaries, get_page_fn)."""
    full_text = " ".join(" ".join(p["text"].split()) for p in pages if p["text"].strip())
    page_boundaries: list[tuple[int, int, int]] = []
    pos = 0
    for p in pages:
        t = " ".join(p["text"].split())
        if not t:
            continue
        page_boundaries.append((pos, pos + len(t), int(p["page"])))
        pos += len(t) + 1

    def get_page(offset: int) -> int:
        for s, e, pg in page_boundaries:
            if s <= offset < e:
                return pg
        return page_boundaries[-1][2]

    return full_text, get_page


def semantic_chunk_document_with_parents(
    pages: list[dict],
    embed_fn: Callable[[list[str]], np.ndarray],
    breakpoint_percentile: int = config.SEMANTIC_BREAKPOINT_PERCENTILE,
    parent_max_chars: int = config.SEMANTIC_PARENT_CHUNK_CHARS,
    small_max_chars: int = config.SEMANTIC_MAX_CHUNK_CHARS,
    min_chunk_chars: int = config.SEMANTIC_MIN_CHUNK_CHARS,
    overlap_ratio: float = config.SEMANTIC_OVERLAP_RATIO,
) -> tuple[list[dict], list[dict]]:
    """Return (small_chunks, parent_chunks).

    Parent chunks: semantically coherent segments sent to the generator.
    Small chunks: each parent subdivided for precise BM25/vector retrieval,
                  carrying a parent_id for look-up after retrieval.
    """
    if not pages:
        return [], []

    source_file = pages[0]["source_file"]
    stem = Path(source_file).stem

    full_text, get_page = _build_page_helpers(pages)
    if not full_text:
        return [], []

    sentences = _split_sentences(full_text)
    if not sentences:
        return [], []

    embeddings = embed_fn(sentences)

    if len(sentences) > 1:
        sims = _cosine_similarities(embeddings)
        breakpoints = set(_find_breakpoints(sims, breakpoint_percentile))
    else:
        breakpoints = set()

    # Parent chunks: large, no overlap (avoid duplicating context sent to Gemma)
    parent_texts = _group_sentences(sentences, breakpoints, parent_max_chars, min_chunk_chars, overlap_ratio=0)

    language = detect_language(full_text)
    parent_chunks: list[dict] = []
    offset = 0
    for idx, text in enumerate(parent_texts):
        parent_chunks.append(
            {
                "id": f"{stem}_par_{idx:04d}",
                "source_file": source_file,
                "page": get_page(offset),
                "text": text,
                "language": language,
            }
        )
        offset += len(text) + 1

    # Small chunks: split each parent further for retrieval
    small_chunks: list[dict] = []
    for parent in parent_chunks:
        small_chunks.extend(_split_to_small(parent, small_max_chars, overlap_ratio))

    return small_chunks, parent_chunks


def semantic_chunk_document(
    pages: list[dict],
    embed_fn: Callable[[list[str]], np.ndarray],
    breakpoint_percentile: int = config.SEMANTIC_BREAKPOINT_PERCENTILE,
    max_chunk_chars: int = config.SEMANTIC_MAX_CHUNK_CHARS,
    min_chunk_chars: int = config.SEMANTIC_MIN_CHUNK_CHARS,
    overlap_ratio: float = config.SEMANTIC_OVERLAP_RATIO,
) -> list[dict]:
    """Chunk a document using embedding similarity breakpoints.

    Args:
        pages: list of page dicts from pdf_parser (same PDF).
        embed_fn: callable that takes list[str] and returns (N, D) float32 array
                  with L2-normalised embeddings.
        breakpoint_percentile: top-X% similarity drops become breakpoints.
        max_chunk_chars: hard character cap per chunk.
    """
    if not pages:
        return []

    source_file = pages[0]["source_file"]
    stem = Path(source_file).stem

    # Merge all pages into one text, track which page each sentence came from
    full_text = " ".join(
        " ".join(p["text"].split()) for p in pages if p["text"].strip()
    )
    if not full_text:
        return []

    # Build page-start offsets for page tagging
    page_boundaries: list[tuple[int, int, int]] = []
    pos = 0
    for p in pages:
        t = " ".join(p["text"].split())
        if not t:
            continue
        page_boundaries.append((pos, pos + len(t), int(p["page"])))
        pos += len(t) + 1  # +1 for the joining space

    def get_page(offset: int) -> int:
        for s, e, pg in page_boundaries:
            if s <= offset < e:
                return pg
        return page_boundaries[-1][2]

    sentences = _split_sentences(full_text)
    if not sentences:
        return []

    # Embed all sentences in one batch call
    embeddings = embed_fn(sentences)  # shape (N, D)

    # Find semantic breakpoints
    if len(sentences) > 1:
        sims = _cosine_similarities(embeddings)
        breakpoints = set(_find_breakpoints(sims, breakpoint_percentile))
    else:
        breakpoints = set()

    chunk_texts = _group_sentences(sentences, breakpoints, max_chunk_chars, min_chunk_chars, overlap_ratio)

    # Tag each chunk with the page where it starts
    language = detect_language(full_text)
    result: list[dict] = []
    offset = 0
    for idx, text in enumerate(chunk_texts):
        result.append(
            {
                "id": f"{stem}_sem_{idx:04d}",
                "source_file": source_file,
                "page": get_page(offset),
                "text": text,
                "language": language,
            }
        )
        offset += len(text) + 1

    return result
