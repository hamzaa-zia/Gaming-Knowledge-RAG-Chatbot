import re

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\s*edit\s*\]", " ", text, flags=re.IGNORECASE)
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
