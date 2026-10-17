"""Turn the raw sources into heading-aware chunks.

Sources (all public, see README for links and terms):
  data/raw/cms_ncd.jsonl                          US Medicare National Coverage Determinations
  data/raw/irdai_health_master_circular_2024.pdf  IRDAI Master Circular on Health Insurance, 2024
  data/raw/arogya_sanjeevani_new_india_assurance.pdf  Arogya Sanjeevani standard policy wording

Chunks follow the documents' own structure (NCD sections and lettered parts, PDF
numbered clauses) instead of fixed character windows, and every chunk carries its
heading path so "this is covered once per year" still says what "this" is.

Run:  python -m coverage_rag.ingest
"""

import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from .config import DATA

RAW = DATA / "raw"
CHUNKS = DATA / "chunks.jsonl"
MAX_CHARS = 1400


@dataclass
class Chunk:
    id: str
    jurisdiction: str  # "US" | "IN"
    source: str  # human-readable document name
    heading: str  # heading path inside the document
    text: str
    url: str = ""
    page: int | None = None

    @property
    def embed_text(self) -> str:
        return f"{self.source} > {self.heading}\n{self.text}"


# ---------------------------------------------------------------- text helpers

def html_to_text(value: str) -> str:
    """NCD fields are HTML that was entity-escaped twice. Keep paragraph breaks."""
    value = html.unescape(html.unescape(value or ""))
    value = re.sub(r"(?i)<\s*(br|/p|/li|/h\d|/tr)\s*/?>", "\n", value)
    value = re.sub(r"(?i)<\s*li[^>]*>", "\n- ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[ \t ]+", " ", value)
    value = re.sub(r"\n\s*\n+", "\n\n", value)
    return value.strip()


def pack(paragraphs: list[str], max_chars: int = MAX_CHARS) -> list[str]:
    """Greedily join paragraphs into chunks no longer than max_chars.

    A single paragraph longer than max_chars is split on sentence ends.
    """
    pieces: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        sentence_buf = ""
        for sentence in re.split(r"(?<=[.;:])\s+", para):
            if sentence_buf and len(sentence_buf) + len(sentence) + 1 > max_chars:
                pieces.append(sentence_buf)
                sentence_buf = sentence
            else:
                sentence_buf = f"{sentence_buf} {sentence}".strip()
        if sentence_buf:
            pieces.append(sentence_buf)

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > max_chars:
            chunks.append(buf)
            buf = piece
        else:
            buf = f"{buf}\n\n{piece}".strip()
    if buf:
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------- CMS NCDs

LETTERED = re.compile(r"^([A-Z])\.\s+(.{3,90})$")


def split_lettered(text: str) -> list[tuple[str, list[str]]]:
    """Split NCD text on its 'A. General' / 'B. Nationally Covered Indications' parts."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for para in text.split("\n"):
        line = para.strip()
        match = LETTERED.match(line)
        if match and len(line) < 100:
            sections.append((f"{match.group(1)}. {match.group(2).rstrip(':')}", []))
        elif line:
            sections[-1][1].append(line)
    return [(h, p) for h, p in sections if p]


def ncd_chunks(path: Path = RAW / "cms_ncd.jsonl") -> list[Chunk]:
    chunks: list[Chunk] = []
    for line in path.read_text().splitlines():
        doc = json.loads(line)
        ncd_id, title = doc["document_display_id"], doc["title"].strip()
        source = f"Medicare NCD {ncd_id} {title}"
        url = f"https://www.cms.gov/medicare-coverage-database/search.aspx?q=NCD%20{ncd_id}"
        n = 0
        for field_name, label in (("item_service_description", "Item/Service Description"),
                                  ("indications_limitations", "Indications and Limitations")):
            for part, paragraphs in split_lettered(html_to_text(doc.get(field_name, ""))):
                heading = f"{label} > {part}" if part else label
                for text in pack(paragraphs):
                    if len(text) < 25:
                        continue
                    n += 1
                    chunks.append(Chunk(f"ncd-{ncd_id}#{n}", "US", source, heading, text, url))
    return chunks


# ---------------------------------------------------------------- IRDAI PDFs

# "3.31. Medically Necessary Treatment means ..." : a dotted clause number always opens a section.
DOTTED_CLAUSE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.\s+([A-Z(].*)$")
# "2 Products are available ..." / "IV Repeal of ..." : undotted, only when it reads like a title.
# This keeps ALL-CAPS list items ("46 KNEE IMMOBILIZER") as content.
UNDOTTED_CLAUSE = re.compile(r"^(\d{1,2}|[A-H]|[IVX]{1,4})\s+([A-Z][^.]{2,80})$")
LIST_TITLE = re.compile(r"^(List [IVX]+)\s*[–-]\s*(.+)$")


def clause_heading(line: str) -> str | None:
    if m := DOTTED_CLAUSE.match(line):
        title = re.split(r"\s+means\b|[:;,(]", m.group(2), maxsplit=1)[0]
        return f"{m.group(1)} {' '.join(title.split()[:10])}"
    if (m := UNDOTTED_CLAUSE.match(line)) and len(line) < 90 and re.search(r"[a-z]", m.group(2)):
        return f"{m.group(1)} {m.group(2).strip().rstrip(':')}"
    if m := LIST_TITLE.match(line):
        return f"{m.group(1)} {m.group(2).strip()}"
    return None
PAGE_NOISE = re.compile(r"(?i)^\s*(page \d+ of \d+|arogya sanjeevani policy, the new india assurance co\.? ltd\.?)\s*$")


def pdf_chunks(path: Path, doc_id: str, source: str, url: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading = "Preamble"
    n = 0
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        lines = [ln.rstrip() for ln in (page.extract_text() or "").splitlines()]
        sections: list[tuple[str, list[str]]] = [(heading, [])]
        for line in lines:
            if not line.strip() or PAGE_NOISE.match(line):
                continue
            found = clause_heading(line.strip())
            if found:
                heading = found
                # The clause line usually carries body text too ("3.18. Family means, ...").
                sections.append((heading, [line.strip()]))
            else:
                sections[-1][1].append(line.strip())
        for sec_heading, sec_lines in sections:
            # PDF lines are wrapped mid-sentence; rejoin them, keep list items apart.
            text = re.sub(r"\n(?!\s*([ivx]+\.|\d+\.|[a-z]\)|-|•))", " ", "\n".join(sec_lines))
            for chunk_text in pack(text.split("\n")):
                if len(chunk_text) < 40:
                    continue
                n += 1
                chunks.append(Chunk(f"{doc_id}-p{page_no}#{n}", "IN", source, sec_heading, chunk_text, url, page_no))
    return chunks


IRDAI_SOURCES = [
    ("irdai_health_master_circular_2024.pdf", "irdai-mc-2024",
     "IRDAI Master Circular on Health Insurance Business, 2024",
     "https://irdai.gov.in/document-detail?documentId=4942918"),
    ("arogya_sanjeevani_new_india_assurance.pdf", "arogya-sanjeevani",
     "Arogya Sanjeevani Policy wording (New India Assurance, IRDAI standard product)",
     "https://irdai.gov.in/faq-s-on-standard-health-product-arogya-sanjeevani-product"),
]


def build() -> list[Chunk]:
    chunks = ncd_chunks()
    for filename, doc_id, source, url in IRDAI_SOURCES:
        chunks += pdf_chunks(RAW / filename, doc_id, source, url)
    return chunks


def load_chunks(path: Path = CHUNKS) -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in path.read_text().splitlines()]


def main() -> None:
    chunks = build()
    with CHUNKS.open("w") as out:
        for chunk in chunks:
            out.write(json.dumps(asdict(chunk)) + "\n")
    by = {j: sum(1 for c in chunks if c.jurisdiction == j) for j in ("US", "IN")}
    print(f"wrote {len(chunks)} chunks to {CHUNKS} ({by})")


if __name__ == "__main__":
    main()
