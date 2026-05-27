import json
from pathlib import Path

from pypdf import PdfReader

from src.config import RAW_DATA_DIR, SUPPORTED_FILE_TYPES, WIKI_SOURCE_LOG_PATH


def normalize_source_title(path: Path) -> str:
    return path.stem.replace("_", " ").strip()


def load_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": str(path),
                        "source_title": normalize_source_title(path),
                        "file_type": "pdf",
                        "page": page_number,
                    },
                }
            )
    return documents


def load_wikipedia_title_map() -> dict[str, str]:
    if not WIKI_SOURCE_LOG_PATH.exists():
        return {}
    try:
        payload = json.loads(WIKI_SOURCE_LOG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    refreshed = payload.get("refreshed", payload if isinstance(payload, list) else [])
    title_map = {}
    for source in refreshed:
        local_path = source.get("local_path")
        title = source.get("title")
        if local_path and title:
            title_map[str(Path(local_path))] = title
    return title_map


def load_text_file(path: Path, title_map: dict[str, str] | None = None) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    source_title = (title_map or {}).get(str(path), normalize_source_title(path))
    return [
        {
            "text": text,
            "metadata": {
                "source": str(path),
                "source_title": source_title,
                "file_type": path.suffix.lower().lstrip("."),
                "page": None,
            },
        }
    ]


def load_documents(raw_dir: Path = RAW_DATA_DIR) -> list[dict]:
    documents = []
    title_map = load_wikipedia_title_map()
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_FILE_TYPES:
            continue
        if path.suffix.lower() == ".pdf":
            documents.extend(load_pdf(path))
        else:
            documents.extend(load_text_file(path, title_map))
    return documents
