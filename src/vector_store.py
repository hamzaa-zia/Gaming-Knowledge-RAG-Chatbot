import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNKS_PATH,
    EMBEDDING_CONFIG_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    INDEX_METADATA_PATH,
)


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    show_progress_bar: bool = False,
) -> np.ndarray:
    # Normalized embeddings let FAISS inner-product search behave like cosine
    # similarity while keeping the index simple and fully local.
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(embeddings, dtype="float32")


def build_source_metadata(chunks: list[dict]) -> tuple[list[dict], list[str]]:
    source_lookup = {}
    for chunk in chunks:
        chunk_metadata = chunk["metadata"]
        title = chunk_metadata.get("source_title", "Unknown")
        source_lookup.setdefault(
            title,
            {
                "title": title,
                "url": chunk_metadata.get("source_url", ""),
                "kind": chunk_metadata.get("source_kind", ""),
            },
        )

    sources = sorted(source_lookup.values(), key=lambda item: item["title"].lower())
    source_titles = [source["title"] for source in sources]
    return sources, source_titles


def build_vector_index(chunks: list[dict], index_dir: Path = INDEX_DIR) -> dict:
    if not chunks:
        raise ValueError("No chunks were created. Add readable documents first.")

    index_dir.mkdir(parents=True, exist_ok=True)
    texts = [chunk["text"] for chunk in chunks]
    model = load_embedding_model()
    embeddings = encode_texts(model, texts, show_progress_bar=True)

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Store chunk metadata beside the FAISS index so search results can return
    # source titles, URLs, pages, and chunk ids without a separate database.
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    embedding_config = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": dimension,
        "normalized_embeddings": True,
        "faiss_index_type": "IndexFlatIP",
        "score_type": "cosine_similarity",
    }
    EMBEDDING_CONFIG_PATH.write_text(
        json.dumps(embedding_config, indent=2),
        encoding="utf-8",
    )

    sources, source_titles = build_source_metadata(chunks)
    metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "retriever": "FAISS",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": dimension,
        "score_type": "cosine_similarity",
        "chunk_count": len(chunks),
        "source_count": len(source_titles),
        "source_titles": source_titles,
        "sources": sources,
    }
    INDEX_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class LocalVectorStore:
    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        faiss_index_path: Path = FAISS_INDEX_PATH,
        embedding_config_path: Path = EMBEDDING_CONFIG_PATH,
    ) -> None:
        if (
            not chunks_path.exists()
            or not faiss_index_path.exists()
            or not embedding_config_path.exists()
        ):
            raise FileNotFoundError("FAISS index not found. Run: python ingest.py")

        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.embedding_config = json.loads(
            embedding_config_path.read_text(encoding="utf-8")
        )
        self.model = load_embedding_model(
            self.embedding_config.get("embedding_model", EMBEDDING_MODEL_NAME)
        )
        self.index = faiss.read_index(str(faiss_index_path))
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS index and chunk metadata are out of sync. Run: python ingest.py"
            )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        # Keep this return shape stable; app.py and rag_chatbot.py depend on it.
        query_embedding = encode_texts(self.model, [query])
        search_limit = min(top_k, len(self.chunks))
        scores, indexes = self.index.search(query_embedding, search_limit)

        results = []
        for rank, (score, index) in enumerate(zip(scores[0], indexes[0]), start=1):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }
            )
        return results


def load_index_metadata() -> dict:
    if not INDEX_METADATA_PATH.exists():
        return {}
    return json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))
