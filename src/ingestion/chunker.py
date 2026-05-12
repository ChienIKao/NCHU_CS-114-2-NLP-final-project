from pathlib import Path
import unicodedata


SENTENCE_BOUNDARIES = "。.!?！？；;\n"


def detect_language(text: str) -> str:
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return "en"
    chinese_count = sum(1 for c in letters if unicodedata.category(c) == "Lo" and "一" <= c <= "鿿")
    return "zh" if chinese_count / len(letters) > 0.2 else "en"


def _find_split(text: str, start: int, max_end: int) -> int:
    if max_end >= len(text):
        return len(text)
    search_start = max(start, max_end - 120)
    for idx in range(max_end, search_start, -1):
        if text[idx - 1] in SENTENCE_BOUNDARIES:
            return idx
    return max_end


def _sliding_chunks(
    full_text: str,
    source_file: str,
    stem: str,
    page_boundaries: list[tuple],
    chunk_size: int,
    overlap: int,
    language: str,
) -> list[dict]:
    """Sliding-window chunking over full_text, tagging each chunk with its starting page."""

    def get_page(pos: int) -> int:
        for (s, e, p) in page_boundaries:
            if s <= pos < e:
                return p
        return page_boundaries[-1][2]

    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(full_text):
        max_end = min(start + chunk_size, len(full_text))
        end = _find_split(full_text, start, max_end)
        chunk_text = full_text[start:end].strip()
        if chunk_text:
            page = get_page(start)
            chunks.append(
                {
                    "id": f"{stem}_{page:04d}_{chunk_idx:04d}",
                    "source_file": source_file,
                    "page": page,
                    "text": chunk_text,
                    "language": language,
                }
            )
            chunk_idx += 1
        if end >= len(full_text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_document(pages: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Merge all pages of one PDF into a single text, then chunk with sliding window.

    Avoids splitting concepts that span multiple slides/pages.
    """
    if not pages:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    source_file = pages[0]["source_file"]
    stem = Path(source_file).stem

    full_text = ""
    page_boundaries: list[tuple] = []
    for page_dict in pages:
        text = " ".join(page_dict["text"].split())
        if not text:
            continue
        char_start = len(full_text)
        if full_text:
            full_text += " "
        full_text += text
        page_boundaries.append((char_start, len(full_text), int(page_dict["page"])))

    if not full_text:
        return []

    language = detect_language(full_text)
    return _sliding_chunks(full_text, source_file, stem, page_boundaries, chunk_size, overlap, language)


def chunk_page(page_dict: dict, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Split one page record into sliding-window chunks (kept for backward compatibility)."""
    return chunk_document([page_dict], chunk_size=chunk_size, overlap=overlap)
