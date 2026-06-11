from pathlib import Path


# Centralized paths keep ingestion, retrieval, and Streamlit using the same
# corpus and generated FAISS index locations.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
WIKIPEDIA_UPDATE_DIR = RAW_DATA_DIR / "wikipedia_updates"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = PROCESSED_DIR / "vector_index"

CHUNKS_PATH = INDEX_DIR / "chunks.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
EMBEDDING_CONFIG_PATH = INDEX_DIR / "embedding_config.json"
INDEX_METADATA_PATH = INDEX_DIR / "index_metadata.json"
WIKI_SOURCE_LOG_PATH = PROCESSED_DIR / "wikipedia_sources.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1100
DEFAULT_CHUNK_OVERLAP = 180
DEFAULT_TOP_K = 8
DEFAULT_SENTENCE_LIMIT = 5

SUPPORTED_FILE_TYPES = {".pdf", ".txt", ".md"}

# Wikipedia pages that can be refreshed into data/raw/wikipedia_updates.
# Local PDFs with matching titles are also mapped to these source URLs.
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
        "title": "Video game genre",
        "url": "https://en.wikipedia.org/wiki/Video_game_genre",
    },
    {
        "title": "Action-adventure game",
        "url": "https://en.wikipedia.org/wiki/Action-adventure_game",
    },
    {
        "title": "Cloud gaming",
        "url": "https://en.wikipedia.org/wiki/Cloud_gaming",
    },
    {
        "title": "Horror game",
        "url": "https://en.wikipedia.org/wiki/Horror_game",
    },
    {
        "title": "Racing video game",
        "url": "https://en.wikipedia.org/wiki/Racing_video_game",
    },
    {
        "title": "Sports video game",
        "url": "https://en.wikipedia.org/wiki/Sports_video_game",
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
    {
        "title": "Esports",
        "url": "https://en.wikipedia.org/wiki/Esports",
    },
    {
        "title": "List of esports games",
        "url": "https://en.wikipedia.org/wiki/List_of_esports_games",
    },
    {
        "title": "Multiplayer online battle arena",
        "url": "https://en.wikipedia.org/wiki/Multiplayer_online_battle_arena",
    },
    {
        "title": "Battle royale game",
        "url": "https://en.wikipedia.org/wiki/Battle_royale_game",
    },
    {
        "title": "Speedrunning",
        "url": "https://en.wikipedia.org/wiki/Speedrunning",
    },
    {
        "title": "Game design",
        "url": "https://en.wikipedia.org/wiki/Game_design",
    },
    {
        "title": "Game mechanics",
        "url": "https://en.wikipedia.org/wiki/Game_mechanics",
    },
    {
        "title": "Level (video games)",
        "url": "https://en.wikipedia.org/wiki/Level_(video_games)",
    },
    {
        "title": "Artificial intelligence in video games",
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence_in_video_games",
    },
]
