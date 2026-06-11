import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import WIKI_SOURCE_LOG_PATH, WIKIPEDIA_PAGES, WIKIPEDIA_UPDATE_DIR


API_URL = "https://en.wikipedia.org/w/api.php"


def safe_filename(title: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", title.strip())
    return name.strip("_") or "wikipedia_page"


def fetch_wikipedia_extract(title: str, retries: int = 3) -> str:
    # Use plaintext extracts so refreshed Wikipedia pages follow the same
    # cleaning/chunking path as PDF and local text sources.
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    headers = {
        "User-Agent": "GamingRAGChatbot/1.0 (student project; Wikipedia API corpus refresh)"
    }
    for attempt in range(1, retries + 1):
        response = requests.get(API_URL, params=params, headers=headers, timeout=30)
        if response.status_code == 429 and attempt < retries:
            time.sleep(4 * attempt)
            continue
        response.raise_for_status()
        break

    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {title}")
    return page.get("extract", "").strip()


def refresh_wikipedia_sources(
    output_dir: Path = WIKIPEDIA_UPDATE_DIR,
    source_log_path: Path = WIKI_SOURCE_LOG_PATH,
) -> list[dict]:
    # Refresh failures are logged and skipped so one missing/rate-limited page
    # does not block rebuilding the rest of the corpus.
    output_dir.mkdir(parents=True, exist_ok=True)
    source_log_path.parent.mkdir(parents=True, exist_ok=True)

    refreshed_sources = []
    skipped_sources = []
    for page in WIKIPEDIA_PAGES:
        title = page["title"]
        try:
            text = fetch_wikipedia_extract(title)
        except Exception as exc:
            skipped_sources.append(
                {
                    "title": title,
                    "url": page["url"],
                    "error": str(exc),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            continue
        if not text:
            continue

        file_path = output_dir / f"{safe_filename(title)}.txt"
        file_path.write_text(text, encoding="utf-8")
        refreshed_sources.append(
            {
                "title": title,
                "url": page["url"],
                "local_path": str(file_path),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "characters": len(text),
            }
        )
        time.sleep(0.8)

    source_log_path.write_text(
        json.dumps(
            {"refreshed": refreshed_sources, "skipped": skipped_sources}, indent=2
        ),
        encoding="utf-8",
    )
    return refreshed_sources
