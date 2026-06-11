import re

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


REFERENCE_SECTION_PATTERN = re.compile(
    r"(?im)(?:^|\n)\s*(?:=+\s*)?"
    r"(references|external links|further reading|bibliography|notes|citations|sources)"
    r"(?:\s*=+)?\s*(?:\n|$)"
)

# These patterns remove Wikipedia/PDF citation noise before chunking so the
# retriever ranks article content instead of references and footers.
CITATION_PATTERN = re.compile(
    r"\[(?:\d+[a-z]?|[a-z]|citation needed|clarification needed|"
    r"failed verification|better source needed|who\?|note \d+)\]",
    flags=re.IGNORECASE,
)


def strip_reference_sections(text: str) -> str:
    match = REFERENCE_SECTION_PATTERN.search(text)
    if not match:
        return text

    # Keep article body sections named "References" only when they are clearly not
    # terminal citation sections. Wikipedia reference sections usually start a page
    # or appear after the main body.
    if match.start() == 0 or match.start() > 200:
        return text[: match.start()]
    return text


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\[\s*edit\s*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = strip_reference_sections(text)
    text = re.sub(
        r"(?im)^\s*\d+\.\s+.*(?:retrieved|archived|https?://|www\.|doi:).*$",
        " ",
        text,
    )
    text = re.sub(r"(?m)^\s*=+\s*([^=\n]{2,90})\s*=+\s*$", r". \1. ", text)
    text = CITATION_PATTERN.sub(" ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\bdoi:\s*\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bISBN\s+[0-9Xx -]{10,24}", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?im)^\s*(category|categories):\s+.*$", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]
        # Prefer sentence or clause boundaries so retrieved chunks read cleanly.
        if end < len(text):
            split_at = max(window.rfind(". "), window.rfind("; "), window.rfind(", "))
            if split_at > chunk_size * 0.6:
                end = start + split_at + 1
                window = text[start:end]
        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    chunks = []
    for doc_index, document in enumerate(documents):
        for chunk_index, chunk in enumerate(
            chunk_text(document["text"], chunk_size, chunk_overlap)
        ):
            metadata = dict(document["metadata"])
            metadata["doc_index"] = doc_index
            metadata["chunk_index"] = chunk_index
            chunk_id = f"{doc_index}-{chunk_index}"
            chunks.append({"id": chunk_id, "text": chunk, "metadata": metadata})
    return chunks


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 30]
