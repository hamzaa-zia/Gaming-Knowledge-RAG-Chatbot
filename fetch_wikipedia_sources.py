from src.wiki_fetcher import refresh_wikipedia_sources


def main() -> None:
    # Utility entry point for refreshing only the configured Wikipedia text
    # sources without rebuilding the FAISS index.
    sources = refresh_wikipedia_sources()
    print(f"Refreshed {len(sources)} Wikipedia pages.")
    for source in sources:
        print(f"- {source['title']} -> {source['local_path']}")


if __name__ == "__main__":
    main()
