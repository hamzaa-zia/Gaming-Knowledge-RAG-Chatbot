import argparse

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.document_loader import load_documents
from src.text_processing import chunk_documents
from src.vector_store import build_vector_index
from src.wiki_fetcher import refresh_wikipedia_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local gaming RAG index.")
    parser.add_argument(
        "--refresh-wikipedia",
        action="store_true",
        help="Download updated text from the configured Wikipedia pages.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.refresh_wikipedia:
        sources = refresh_wikipedia_sources()
        print(f"Refreshed {len(sources)} Wikipedia pages.")

    documents = load_documents()
    print(f"Loaded {len(documents)} document sections/pages.")

    chunks = chunk_documents(
        documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    print(f"Created {len(chunks)} text chunks.")

    metadata = build_vector_index(chunks)
    print("Vector index built successfully.")
    print(f"Sources: {metadata['source_count']}")
    print(f"Chunks: {metadata['chunk_count']}")


if __name__ == "__main__":
    main()
