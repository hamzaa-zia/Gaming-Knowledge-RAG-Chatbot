from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
WIKIPEDIA_UPDATE_DIR = RAW_DATA_DIR / "wikipedia_updates"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = PROCESSED_DIR / "vector_index"

CHUNKS_PATH = INDEX_DIR / "chunks.json"
VECTORIZER_PATH = INDEX_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = INDEX_DIR / "tfidf_matrix.joblib"
INDEX_METADATA_PATH = INDEX_DIR / "index_metadata.json"
WIKI_SOURCE_LOG_PATH = PROCESSED_DIR / "wikipedia_sources.json"

DEFAULT_CHUNK_SIZE = 1100
DEFAULT_CHUNK_OVERLAP = 180
DEFAULT_TOP_K = 8
DEFAULT_SENTENCE_LIMIT = 5

SUPPORTED_FILE_TYPES = {".pdf", ".txt", ".md"}

WIKIPEDIA_PAGES = [
    {
        "title": "Gaming",
        "url": "https://en.wikipedia.org/wiki/Gaming",
    },
    {
        "title": "Video game",
        "url": "https://en.wikipedia.org/wiki/Video_game",
    },
    {
        "title": "History of video games",
        "url": "https://en.wikipedia.org/wiki/History_of_video_games",
    },
    {
        "title": "Early history of video games",
        "url": "https://en.wikipedia.org/wiki/Early_history_of_video_games",
    },
    {
        "title": "AAA (video game industry)",
        "url": "https://en.wikipedia.org/wiki/AAA_(video_game_industry)",
    },
    {
        "title": "Open world",
        "url": "https://en.wikipedia.org/wiki/Open_world",
    },
    {
        "title": "Video game industry",
        "url": "https://en.wikipedia.org/wiki/Video_game_industry",
    },
    {
        "title": "Indie game",
        "url": "https://en.wikipedia.org/wiki/Indie_game",
    },
    {
        "title": "Live service game",
        "url": "https://en.wikipedia.org/wiki/Live_service_game",
    },
    {
        "title": "List of largest video game companies by revenue",
        "url": "https://en.wikipedia.org/wiki/List_of_largest_video_game_companies_by_revenue",
    },
    {
        "title": "List of best-selling video games",
        "url": "https://en.wikipedia.org/wiki/List_of_best-selling_video_games",
    },
]
