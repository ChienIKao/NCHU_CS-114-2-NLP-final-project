from pathlib import Path
import unicodedata


SENTENCE_BOUNDARIES = "。.!?！？；;\n"


def detect_language(text: str) -> str:
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return "en"
    chinese_count = sum(1 for c in letters if unicodedata.category(c) == "Lo" and "\u4e00" <= c <= "\u9fff")
    return "zh" if chinese_count / len(letters) > 0.2 else "en"


def _find_split(text: str, start: int, max_end: int) -> int:
    if max_end >= len(text):
        return len(text)
    search_start = max(start, max_end - 120)
    for idx in range(max_end, search_start, -1):
        if text[idx - 1] in SENTENCE_BOUNDARIES:
            return idx
    return max_end


def chunk_page(page_dict: dict, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Split one page record into sliding-window chunks."""
    text = " ".join(page_dict["text"].split())
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    source_file = page_dict["source_file"]
    stem = Path(source_file).stem
    page = int(page_dict["page"])
    language = detect_language(text)

    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(text):
        max_end = min(start + chunk_size, len(text))
        end = _find_split(text, start, max_end)
        chunk_text = text[start:end].strip()
        if chunk_text:
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
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks
