import re
from pathlib import Path

import fitz


def _clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if len(line) <= 4 and re.search(r"\d", line):
            continue
        if len(line) <= 80 and line.isupper() and re.search(r"[A-Z]", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def parse_pdf(path: str) -> list[dict]:
    """Parse a PDF and return one text record per page."""
    pdf_path = Path(path)
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            text = _clean_text(page.get_text())
            if not text:
                continue
            pages.append(
                {
                    "source_file": pdf_path.name,
                    "page": page_idx,
                    "text": text,
                }
            )
    return pages
