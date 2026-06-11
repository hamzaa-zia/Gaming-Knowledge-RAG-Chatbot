import json
from pathlib import Path
from urllib.parse import quote

from pypdf import PdfReader

from src.config import (
    RAW_DATA_DIR,
    SUPPORTED_FILE_TYPES,
    WIKIPEDIA_PAGES,
    WIKIPEDIA_UPDATE_DIR,
    WIKI_SOURCE_LOG_PATH,
)


def normalize_source_title(path: Path) -> str:
    return path.stem.replace("_", " ").strip()


def title_to_wikipedia_url(title: str) -> str:
    article_path = quote(title.replace(" ", "_"), safe="()_-")
    return f"https://en.wikipedia.org/wiki/{article_path}"


def configured_wikipedia_urls() -> dict[str, str]:
    return {source["title"]: source["url"] for source in WIKIPEDIA_PAGES}


def path_keys(path: Path) -> set[str]:
    # Store both raw and resolved paths because refresh logs may contain either.
    keys = {str(path)}
    try:
        keys.add(str(path.resolve()))
    except OSError:
        pass
    return keys


def source_kind(path: Path, file_type: str) -> str:
    try:
        if WIKIPEDIA_UPDATE_DIR in path.resolve().parents:
            return "wikipedia_refresh"
    except OSError:
        pass
    if file_type == "pdf":
        return "wikipedia_pdf"
    return "local_text"


def build_source_metadata(
    path: Path,
    file_type: str,
    page: int | None,
    source_map: dict[str, dict[str, str]] | None = None,
) -> dict:
    # Every chunk carries source metadata so answers can cite Wikipedia titles,
    # page numbers, local files, and clickable URLs.
    source_info = {}
    for key in path_keys(path):
        if source_map and key in source_map:
            source_info = source_map[key]
            break

    source_title = source_info.get("title") or normalize_source_title(path)
    url_lookup = configured_wikipedia_urls()
    source_url = (
        source_info.get("url")
        or url_lookup.get(source_title)
        or title_to_wikipedia_url(source_title)
    )

    return {
        "source": str(path),
        "source_title": source_title,
        "source_url": source_url,
        "source_kind": source_kind(path, file_type),
        "file_type": file_type,
        "page": page,
    }


def load_pdf(path: Path, source_map: dict[str, dict[str, str]] | None = None) -> list[dict]:
    reader = PdfReader(str(path))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                {
                    "text": text,
                    "metadata": build_source_metadata(
                        path=path,
                        file_type="pdf",
                        page=page_number,
                        source_map=source_map,
                    ),
                }
            )
    return documents


def load_wikipedia_source_map() -> dict[str, dict[str, str]]:
    # Refreshed Wikipedia text files get exact title/URL metadata from the
    # refresh log instead of relying only on local filenames.
    if not WIKI_SOURCE_LOG_PATH.exists():
        return {}
    try:
        payload = json.loads(WIKI_SOURCE_LOG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    refreshed = payload.get("refreshed", payload if isinstance(payload, list) else [])
    source_map = {}
    for source in refreshed:
        local_path = source.get("local_path")
        title = source.get("title")
        url = source.get("url")
        if local_path and title:
            info = {"title": title, "url": url or title_to_wikipedia_url(title)}
            for key in path_keys(Path(local_path)):
                source_map[key] = info
    return source_map


def load_text_file(path: Path, source_map: dict[str, dict[str, str]] | None = None) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    return [
        {
            "text": text,
            "metadata": build_source_metadata(
                path=path,
                file_type=path.suffix.lower().lstrip("."),
                page=None,
                source_map=source_map,
            ),
        }
    ]


def load_documents(raw_dir: Path = RAW_DATA_DIR) -> list[dict]:
    # PDFs are split page-by-page; text/markdown files become one document each
    # before the shared chunking pipeline handles windowing.
    documents = []
    source_map = load_wikipedia_source_map()
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_FILE_TYPES:
            continue
        if path.suffix.lower() == ".pdf":
            documents.extend(load_pdf(path, source_map))
        else:
            documents.extend(load_text_file(path, source_map))
    return documents
