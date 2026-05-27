import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CHUNKS_PATH,
    INDEX_DIR,
    INDEX_METADATA_PATH,
    MATRIX_PATH,
    VECTORIZER_PATH,
)


def build_vector_index(chunks: list[dict], index_dir: Path = INDEX_DIR) -> dict:
    if not chunks:
        raise ValueError("No chunks were created. Add readable documents first.")

    index_dir.mkdir(parents=True, exist_ok=True)
    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=60000,
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(matrix, MATRIX_PATH)

    source_titles = sorted(
        {chunk["metadata"].get("source_title", "Unknown") for chunk in chunks}
    )
    metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "source_count": len(source_titles),
        "source_titles": source_titles,
    }
    INDEX_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class LocalVectorStore:
    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        vectorizer_path: Path = VECTORIZER_PATH,
        matrix_path: Path = MATRIX_PATH,
    ) -> None:
        if not chunks_path.exists() or not vectorizer_path.exists() or not matrix_path.exists():
            raise FileNotFoundError("Vector index not found. Run: python ingest.py")
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.vectorizer = joblib.load(vectorizer_path)
        self.matrix = joblib.load(matrix_path)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indexes = scores.argsort()[::-1][:top_k]

        results = []
        for rank, index in enumerate(ranked_indexes, start=1):
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[index]),
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
