import subprocess
import sys
from html import escape
from pathlib import Path

import streamlit as st

from src.config import INDEX_METADATA_PATH, ROOT_DIR
from src.llm_client import (
    get_gemini_model_name,
    get_gemini_warning,
    has_gemini_key,
)
from src.rag_chatbot import RetrievalChatbot
from src.vector_store import LocalVectorStore, load_index_metadata


st.set_page_config(
    page_title="Arcade RAG Console",
    layout="wide",
    initial_sidebar_state="collapsed",
)


TOPIC_BUTTONS = [
    ("History", "How has gaming evolved?", ":material/history:"),
    ("AAA", "Why do AAA games sell so much?", ":material/diamond:"),
    ("Open World", "Why are open world games popular?", ":material/travel_explore:"),
    ("Genres", "What are the main video game genres?", ":material/category:"),
    ("Cloud", "What is cloud gaming?", ":material/cloud:"),
    ("Mobile", "How are mobile games different from console games?", ":material/smartphone:"),
    ("RPG", "What are role-playing games?", ":material/auto_stories:"),
    ("Market", "What is the total market of gaming?", ":material/query_stats:"),
]

QUICK_LAUNCH_COLUMNS = 4

# Streamlit is styled through one CSS block so the app feels like a custom
# desktop console while still using Streamlit's runtime controls underneath.
CUSTOM_CSS = """
<style>
    :root {
        --bg: #090a12;
        --panel: #111421;
        --panel-soft: #171b2b;
        --line: rgba(0, 245, 255, 0.32);
        --line-magenta: rgba(255, 47, 146, 0.36);
        --cyan: #00f5ff;
        --magenta: #ff2f92;
        --lime: #b8ff4d;
        --amber: #ffcc33;
        --text: #edf7ff;
        --muted: #9aa7bd;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:
            linear-gradient(rgba(0, 245, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 47, 146, 0.035) 1px, transparent 1px),
            var(--bg);
        background-size: 34px 34px;
        color: var(--text);
    }

    #MainMenu,
    footer,
    header,
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    .block-container {
        padding-top: 0.85rem;
        max-width: 1280px;
        padding-bottom: 4.5rem;
    }

    .arcade-topbar {
        border: 1px solid var(--line);
        border-left: 4px solid var(--cyan);
        background: linear-gradient(135deg, rgba(17, 20, 33, 0.96), rgba(23, 27, 43, 0.92));
        padding: 0.68rem 0.85rem 0.75rem;
        border-radius: 8px;
        box-shadow: 0 0 28px rgba(0, 245, 255, 0.1);
        margin-bottom: 0.7rem;
    }

    .title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .title-copy {
        min-width: 260px;
        flex: 1 1 420px;
    }

    .main-title {
        font-size: 1.42rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: 0;
        margin: 0;
        color: var(--text);
        text-shadow: 0 0 16px rgba(0, 245, 255, 0.34);
    }

    .subtitle {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: 0.24rem;
    }

    .header-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(105px, 1fr));
        gap: 0.55rem;
        flex: 1 1 390px;
        max-width: 520px;
    }

    .header-stat {
        border: 1px solid rgba(255, 204, 51, 0.26);
        border-radius: 8px;
        background: rgba(9, 10, 18, 0.45);
        padding: 0.38rem 0.5rem;
        min-height: 48px;
    }

    .header-stat-label {
        color: var(--muted);
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .header-stat-value {
        color: var(--amber);
        font-size: 0.94rem;
        font-weight: 780;
        line-height: 1.15;
        margin-top: 0.16rem;
        white-space: nowrap;
    }

    .status-strip {
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
        justify-content: flex-start;
        margin-top: 0.58rem;
    }

    .status-pill {
        border: 1px solid rgba(184, 255, 77, 0.38);
        color: var(--lime);
        background: rgba(184, 255, 77, 0.08);
        border-radius: 999px;
        padding: 0.22rem 0.5rem;
        font-size: 0.7rem;
        white-space: nowrap;
    }

    .section-label {
        color: var(--cyan);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0.75rem 0 0.4rem;
    }

    .dock-panel {
        border: 1px solid rgba(255, 47, 146, 0.28);
        border-radius: 8px;
        background: rgba(17, 20, 33, 0.84);
        padding: 0.85rem;
        margin-top: 0;
        min-height: 210px;
        box-shadow: 0 0 24px rgba(255, 47, 146, 0.08);
    }

    .command-panel {
        border-color: rgba(0, 245, 255, 0.25);
        box-shadow: 0 0 22px rgba(0, 245, 255, 0.07);
        margin-bottom: 0.75rem;
    }

    .retrieval-panel {
        border-color: rgba(0, 245, 255, 0.28);
        box-shadow: 0 0 22px rgba(0, 245, 255, 0.07);
        position: sticky;
        top: 0.75rem;
        max-height: calc(100vh - 1.5rem);
        overflow-y: auto;
    }

    .panel-subcopy {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.45;
        margin-bottom: 0.8rem;
    }

    .prompt-card {
        border: 1px solid rgba(0, 245, 255, 0.18);
        border-radius: 8px;
        background: rgba(0, 245, 255, 0.05);
        padding: 0.58rem 0.65rem;
        margin-bottom: 0.7rem;
    }

    .prompt-label {
        color: var(--cyan);
        font-size: 0.64rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.28rem;
    }

    .prompt-text {
        color: var(--text);
        font-size: 0.82rem;
        line-height: 1.38;
        overflow-wrap: anywhere;
    }

    .query-text {
        color: var(--muted);
        font-size: 0.72rem;
        line-height: 1.35;
        margin-top: 0.45rem;
        overflow-wrap: anywhere;
    }

    .dock-title {
        color: var(--magenta);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.65rem;
    }

    .topic-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.75rem;
    }

    .topic-chip {
        border: 1px solid rgba(0, 245, 255, 0.3);
        color: var(--cyan);
        border-radius: 999px;
        padding: 0.22rem 0.52rem;
        font-size: 0.72rem;
        background: rgba(0, 245, 255, 0.06);
    }

    .empty-source {
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
        line-height: 1.45;
    }

    .chat-frame {
        border-top: 1px solid var(--line);
        padding-top: 0.75rem;
    }

    .custom-chat {
        display: flex;
        width: 100%;
        margin: 0 0 0.7rem;
    }

    .custom-chat.assistant {
        justify-content: flex-start;
    }

    .custom-chat.user {
        justify-content: flex-end;
    }

    .custom-chat-bubble {
        max-width: 88%;
        border-radius: 10px;
        padding: 0.82rem 0.95rem;
        background: linear-gradient(135deg, rgba(17, 20, 33, 0.96), rgba(9, 12, 21, 0.94));
        border: 1px solid rgba(0, 245, 255, 0.18);
        box-shadow: 0 0 18px rgba(0, 245, 255, 0.06);
    }

    .custom-chat.assistant .custom-chat-bubble {
        border-left: 3px solid var(--cyan);
        margin-right: 7%;
    }

    .custom-chat.user .custom-chat-bubble {
        border-right: 3px solid var(--magenta);
        background: linear-gradient(135deg, rgba(255, 47, 146, 0.13), rgba(17, 20, 33, 0.92));
        margin-left: 14%;
    }

    .chat-label {
        color: var(--cyan);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.38rem;
    }

    .custom-chat.user .chat-label {
        color: #ff9ac9;
        text-align: right;
    }

    .chat-body {
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .chat-body p {
        margin: 0 0 0.45rem;
    }

    .chat-body p:last-child {
        margin-bottom: 0;
    }

    .chat-body ul {
        margin: 0.3rem 0 0;
        padding-left: 1.1rem;
    }

    .chat-body li {
        margin-bottom: 0.4rem;
    }

    .message-source-strip {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.38rem;
        margin-top: 0.72rem;
        padding-top: 0.62rem;
        border-top: 1px solid rgba(0, 245, 255, 0.12);
    }

    .message-source-label {
        color: var(--muted);
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .message-source-chip {
        border: 1px solid rgba(184, 255, 77, 0.28);
        border-radius: 999px;
        padding: 0.18rem 0.48rem;
        color: var(--lime);
        background: rgba(184, 255, 77, 0.07);
        font-size: 0.72rem;
        line-height: 1.2;
        text-decoration: none;
        max-width: 100%;
        overflow-wrap: anywhere;
    }

    .message-source-chip:hover {
        color: #ffffff;
        border-color: rgba(184, 255, 77, 0.48);
        text-decoration: none;
    }

    .custom-chat.loading .custom-chat-bubble {
        border-left-color: var(--lime);
        background: linear-gradient(135deg, rgba(17, 20, 33, 0.98), rgba(14, 19, 27, 0.96));
    }

    .loading-title {
        color: var(--lime);
        font-weight: 760;
        font-size: 0.92rem;
    }

    .loading-detail {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }

    .typing-dots {
        display: inline-flex;
        gap: 0.22rem;
        margin-left: 0.35rem;
        vertical-align: middle;
    }

    .typing-dots span {
        width: 0.34rem;
        height: 0.34rem;
        border-radius: 999px;
        background: var(--lime);
        opacity: 0.35;
        animation: pulseDot 1.1s infinite ease-in-out;
    }

    .typing-dots span:nth-child(2) {
        animation-delay: 0.16s;
    }

    .typing-dots span:nth-child(3) {
        animation-delay: 0.32s;
    }

    @keyframes pulseDot {
        0%, 80%, 100% {
            opacity: 0.25;
            transform: translateY(0);
        }
        40% {
            opacity: 1;
            transform: translateY(-2px);
        }
    }

    [data-testid="stChatMessage"] {
        background: linear-gradient(135deg, rgba(17, 20, 33, 0.95), rgba(11, 14, 24, 0.92));
        border: 1px solid rgba(0, 245, 255, 0.18);
        border-radius: 10px;
        box-shadow: 0 0 18px rgba(0, 245, 255, 0.06);
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.65rem;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-color: rgba(0, 245, 255, 0.28);
        border-left: 3px solid var(--cyan);
        margin-right: 7%;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-color: rgba(255, 47, 146, 0.3);
        border-right: 3px solid var(--magenta);
        background: linear-gradient(135deg, rgba(255, 47, 146, 0.12), rgba(17, 20, 33, 0.92));
        margin-left: 14%;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        line-height: 1.55;
        font-size: 0.95rem;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.45rem;
    }

    [data-testid="stChatInput"] {
        background: rgba(9, 10, 18, 0.86);
        border-top: 1px solid rgba(0, 245, 255, 0.14);
        padding: 0.38rem 0.75rem 0.45rem !important;
    }

    [data-testid="stChatInput"] > div {
        max-width: 720px;
        margin: 0 auto;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        min-height: 2.25rem !important;
        padding: 0.42rem 0.72rem !important;
        font-size: 0.9rem !important;
        line-height: 1.25 !important;
    }

    textarea,
    input {
        color: var(--text) !important;
    }

    .stButton > button {
        background: rgba(17, 20, 33, 0.92);
        color: var(--text);
        border: 1px solid rgba(0, 245, 255, 0.42);
        border-radius: 8px;
        min-height: 2.25rem;
        width: 100%;
        padding: 0.35rem 0.52rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.35rem;
        box-shadow: inset 0 0 0 1px rgba(255, 47, 146, 0.08);
    }

    .stButton > button * {
        white-space: nowrap !important;
        overflow-wrap: normal !important;
        word-break: keep-all !important;
    }

    .stButton > button p {
        font-size: 0.88rem;
        line-height: 1;
    }

    .stButton > button:hover {
        color: #ffffff;
        border-color: var(--magenta);
        box-shadow: 0 0 18px rgba(255, 47, 146, 0.18);
    }

    .stButton > button:focus:not(:active) {
        border-color: var(--lime);
        box-shadow: 0 0 0 0.12rem rgba(184, 255, 77, 0.22);
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stCheckbox"] label {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    div[role="radiogroup"] {
        gap: 0.15rem;
    }

    div[role="radiogroup"] label {
        background: rgba(0, 245, 255, 0.05);
        border: 1px solid rgba(0, 245, 255, 0.14);
        border-radius: 8px;
        padding: 0.2rem 0.35rem;
        margin-bottom: 0.22rem;
    }

    .source-box {
        border: 1px solid rgba(0, 245, 255, 0.24);
        border-left: 3px solid var(--magenta);
        border-radius: 8px;
        padding: 0.7rem 0.78rem;
        margin-bottom: 0.55rem;
        background: rgba(17, 20, 33, 0.92);
        font-size: 0.86rem;
        box-shadow: 0 0 18px rgba(255, 47, 146, 0.06);
    }

    .source-box strong {
        color: var(--cyan);
    }

    .source-box span {
        color: var(--muted);
        overflow-wrap: anywhere;
    }

    .source-meta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin: 0.38rem 0 0.28rem;
    }

    .confidence-pill {
        border-radius: 999px;
        padding: 0.12rem 0.45rem;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border: 1px solid rgba(154, 167, 189, 0.32);
        color: var(--muted);
        background: rgba(154, 167, 189, 0.08);
    }

    .confidence-high {
        border-color: rgba(184, 255, 77, 0.42);
        color: var(--lime);
        background: rgba(184, 255, 77, 0.08);
    }

    .confidence-medium {
        border-color: rgba(255, 204, 51, 0.42);
        color: var(--amber);
        background: rgba(255, 204, 51, 0.08);
    }

    .confidence-low {
        border-color: rgba(255, 47, 146, 0.38);
        color: #ff9ac9;
        background: rgba(255, 47, 146, 0.08);
    }

    .source-link {
        color: var(--lime);
        text-decoration: none;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .source-link:hover {
        color: #ffffff;
        text-decoration: underline;
    }

    .source-detail {
        color: var(--muted);
        font-size: 0.76rem;
        overflow-wrap: anywhere;
    }

    .stExpander {
        border: 1px solid rgba(0, 245, 255, 0.2) !important;
        border-radius: 8px !important;
        background: rgba(17, 20, 33, 0.88) !important;
    }

    div[data-testid="stAlert"] {
        background: rgba(255, 204, 51, 0.08);
        border: 1px solid rgba(255, 204, 51, 0.35);
        color: var(--text);
    }

    hr {
        border-color: rgba(0, 245, 255, 0.18);
    }

    @media (max-width: 820px) {
        .header-stats {
            grid-template-columns: 1fr;
            max-width: none;
        }
        .main-title {
            font-size: 1.35rem;
        }
        .status-strip {
            justify-content: flex-start;
        }
    }
</style>
"""


def initialize_state() -> None:
    st.session_state.pop("answer_style", None)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Arcade index loaded. I can answer from the indexed Wikipedia gaming corpus."
                ),
            }
        ]
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""
    if "last_search_query" not in st.session_state:
        st.session_state.last_search_query = ""
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


@st.cache_resource
def load_chatbot() -> RetrievalChatbot:
    vector_store = LocalVectorStore()
    return RetrievalChatbot(vector_store)


def run_ingestion(refresh_wikipedia: bool) -> tuple[bool, str]:
    command = [sys.executable, "ingest.py"]
    if refresh_wikipedia:
        command.append("--refresh-wikipedia")
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output


def source_confidence(score: float) -> tuple[str, str]:
    if score >= 0.55:
        return "High", "confidence-high"
    if score >= 0.35:
        return "Medium", "confidence-medium"
    if score >= 0.18:
        return "Low", "confidence-low"
    return "Trace", ""


def format_chat_body(content: str) -> str:
    # Keep answer text safe inside custom HTML and preserve simple markdown lists.
    lines = str(content).splitlines()
    html_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{escape(stripped[2:])}</li>")
            continue
        if in_list:
            html_parts.append("</ul>")
            in_list = False
        html_parts.append(f"<p>{escape(stripped)}</p>")

    if in_list:
        html_parts.append("</ul>")
    return "\n".join(html_parts)


def build_message_sources_html(sources: list[dict] | None) -> str:
    if not sources:
        return ""

    chips = ['<span class="message-source-label">Sources</span>']
    seen = set()
    for source in sources:
        title = str(source.get("title", "Wikipedia source")).strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        url = str(source.get("source_url", "")).strip()
        safe_title = escape(title)
        if url:
            chips.append(
                f'<a class="message-source-chip" href="{escape(url, quote=True)}" '
                f'target="_blank" rel="noopener noreferrer">{safe_title}</a>'
            )
        else:
            chips.append(f'<span class="message-source-chip">{safe_title}</span>')
        if len(chips) >= 4:
            break

    return f'<div class="message-source-strip">{"".join(chips)}</div>'


def render_chat_message(
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> None:
    safe_role = "user" if role == "user" else "assistant"
    label = "Player" if safe_role == "user" else "Arcade RAG"
    sources_html = build_message_sources_html(sources) if safe_role == "assistant" else ""
    message_html = (
        f'<div class="custom-chat {safe_role}">'
        '<div class="custom-chat-bubble">'
        f'<div class="chat-label">{label}</div>'
        f'<div class="chat-body">{format_chat_body(content)}</div>'
        f"{sources_html}"
        "</div>"
        "</div>"
    )
    st.markdown(message_html, unsafe_allow_html=True)


def build_loading_html(title: str, detail: str) -> str:
    return (
        '<div class="custom-chat assistant loading">'
        '<div class="custom-chat-bubble">'
        '<div class="chat-label">Arcade RAG</div>'
        f'<div class="loading-title">{escape(title)}'
        '<span class="typing-dots"><span></span><span></span><span></span></span>'
        "</div>"
        f'<div class="loading-detail">{escape(detail)}</div>'
        "</div>"
        "</div>"
    )


def build_sources_html(sources: list[dict]) -> str:
    if not sources:
        return (
            '<div class="empty-source">'
            "Ask a question to see ranked Wikipedia sources, confidence, and links."
            "</div>"
        )

    items = []
    for source in sources:
        page = f"Page {escape(str(source['page']))}" if source.get("page") else ""
        chunk = f"Chunk {escape(str(source['chunk']))}" if source.get("chunk") is not None else ""
        location = " | ".join(item for item in (page, chunk) if item)
        title = escape(str(source.get("title", "Unknown")))
        raw_score = source.get("score", 0)
        try:
            score_value = float(raw_score)
        except (TypeError, ValueError):
            score_value = 0.0
        confidence, confidence_class = source_confidence(score_value)
        score = escape(f"{score_value:.4f}")
        source_path = source.get("source", "")
        source_file = escape(Path(source_path).name if source_path else "Indexed corpus")
        url = escape(str(source.get("source_url", "")), quote=True)
        link = (
            f'<a class="source-link" href="{url}" target="_blank" '
            f'rel="noopener noreferrer">Open Wikipedia</a>'
            if url
            else '<span class="source-detail">No URL</span>'
        )
        location_html = f'<div class="source-detail">{location}</div>' if location else ""
        items.append(
            f'<div class="source-box">'
            f"<strong>{title}</strong>"
            f"{location_html}"
            f'<div class="source-meta">'
            f'<span class="confidence-pill {confidence_class}">{confidence}</span>'
            f"<span>Score {score}</span>"
            f"{link}"
            f"</div>"
            f'<div class="source-detail">{source_file}</div>'
            f"</div>"
        )
    return "\n".join(items)


def build_retrieval_panel_html(
    sources: list[dict],
    prompt: str = "",
    search_query: str = "",
) -> str:
    prompt_html = ""
    if prompt:
        prompt_html = (
            '<div class="prompt-card">'
            '<div class="prompt-label">Current Prompt</div>'
            f'<div class="prompt-text">{escape(prompt)}</div>'
        )
        if search_query:
            prompt_html += (
                '<div class="query-text">'
                f"Query: {escape(search_query)}"
                "</div>"
            )
        prompt_html += "</div>"
    else:
        prompt_html = (
            '<div class="prompt-card">'
            '<div class="prompt-label">Current Prompt</div>'
            '<div class="prompt-text">Ask a question to populate this retrieval rail.</div>'
            "</div>"
        )

    return (
        '<div class="dock-panel retrieval-panel">'
        '<div class="dock-title">Retrieval Feed</div>'
        '<div class="panel-subcopy">Conversation-aware query and ranked sources for the latest prompt.</div>'
        f"{prompt_html}"
        f"{build_sources_html(sources)}"
        "</div>"
    )


initialize_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
metadata = load_index_metadata()

gemini_ready = has_gemini_key()
gemini_warning = get_gemini_warning() if gemini_ready else ""
answer_engine = st.session_state.get("answer_engine", "Extractive")

llm_provider = "Extractive"
if answer_engine == "Gemini API" and gemini_ready and not gemini_warning:
    llm_provider = "Gemini"

source_count = metadata.get("source_count", 0) if metadata else 0
chunk_count = metadata.get("chunk_count", 0) if metadata else 0
retriever_label = metadata.get("retriever", "FAISS") if metadata else "FAISS"
mode_label = llm_provider if llm_provider != "Extractive" else retriever_label

st.markdown(
    f"""
    <div class="arcade-topbar">
        <div class="title-row">
            <div class="title-copy">
                <div class="main-title">Arcade RAG Console</div>
                <div class="subtitle">Gaming knowledge retrieval from the local Wikipedia corpus.</div>
                <div class="status-strip">
                    <span class="status-pill">FAISS INDEX</span>
                    <span class="status-pill">MEMORY ON</span>
                    <span class="status-pill">WIKI CORPUS</span>
                    <span class="status-pill">SOURCE LINKS</span>
                </div>
            </div>
            <div class="header-stats">
                <div class="header-stat">
                    <div class="header-stat-label">Sources</div>
                    <div class="header-stat-value">{source_count}</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-label">Chunks</div>
                    <div class="header-stat-value">{chunk_count}</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-label">Mode</div>
                    <div class="header-stat-value">{mode_label}</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

feed_col, main_col, command_col = st.columns([1.12, 2.78, 1.1], gap="large")

with feed_col:
    st.markdown(
        build_retrieval_panel_html(
            sources=st.session_state.last_sources,
            prompt=st.session_state.last_prompt,
            search_query=st.session_state.last_search_query,
        ),
        unsafe_allow_html=True,
    )

with main_col:

    st.markdown('<div class="section-label">Quick Launch</div>', unsafe_allow_html=True)
    for row_start in range(0, len(TOPIC_BUTTONS), QUICK_LAUNCH_COLUMNS):
        quick_cols = st.columns(QUICK_LAUNCH_COLUMNS, gap="small")
        row_buttons = TOPIC_BUTTONS[row_start : row_start + QUICK_LAUNCH_COLUMNS]
        for column, (label, question, icon) in zip(quick_cols, row_buttons):
            with column:
                if st.button(label, icon=icon, use_container_width=True):
                    st.session_state.pending_prompt = question
                    st.rerun()

    if not INDEX_METADATA_PATH.exists():
        st.info("Build the index from the command deck before asking questions.")

    st.markdown('<div class="chat-frame"></div>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        render_chat_message(
            message["role"],
            message["content"],
            message.get("sources"),
        )

with command_col:
    st.markdown(
        f"""
        <div class="dock-panel command-panel">
            <div class="dock-title">Command Deck</div>
            <div class="panel-subcopy">Choose the answer engine and rebuild the local corpus index from here.</div>
            <div class="topic-list">
                <span class="topic-chip">Sources {source_count}</span>
                <span class="topic-chip">Chunks {chunk_count}</span>
                <span class="topic-chip">{mode_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode_options = ["Extractive", "Gemini API"]
    selected_mode_index = mode_options.index(answer_engine) if answer_engine in mode_options else 0
    st.radio(
        "Answer Engine",
        mode_options,
        index=selected_mode_index,
        key="answer_engine",
        help="Extractive is local. Gemini API uses retrieved context plus your Gemini key.",
    )
    if st.session_state.get("answer_engine") == "Gemini API" and gemini_ready:
        st.caption(f"Gemini model: {get_gemini_model_name()}")
        if gemini_warning:
            st.warning(gemini_warning)
    elif st.session_state.get("answer_engine") == "Gemini API":
        st.warning("GEMINI_API_KEY was not found in .env or .env.txt.")
    if st.session_state.get("answer_engine") == "Gemini API" and gemini_warning:
        st.warning("Gemini API mode is disabled until the model configuration is compatible.")

    refresh = st.checkbox("Refresh Wikipedia text", value=False, key="refresh_wikipedia")
    if st.button("Build / Rebuild Index", icon=":material/sync:", use_container_width=True):
        with st.status("Building FAISS vector index...", expanded=True) as status:
            ok, output = run_ingestion(refresh)
            if ok:
                load_chatbot.clear()
                status.update(label="Index built successfully", state="complete")
                st.code(output)
            else:
                status.update(label="Index build failed", state="error")
                st.code(output)

    if st.button("Reset Run", icon=":material/restart_alt:", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_prompt = ""
        st.session_state.last_search_query = ""
        st.session_state.pending_prompt = None
        st.rerun()

prompt = st.session_state.pending_prompt or st.chat_input("Ask about gaming...")
if prompt:
    st.session_state.pending_prompt = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    with main_col:
        render_chat_message("user", prompt)

    try:
        chatbot = load_chatbot()
        with main_col:
            loading_slot = st.empty()
            loading_slot.markdown(
                build_loading_html(
                    "Starting retrieval",
                    "Preparing the local gaming corpus search.",
                ),
                unsafe_allow_html=True,
            )
            with st.status("Preparing source-aware answer...", expanded=True) as status:
                def update_progress(title: str, detail: str) -> None:
                    loading_slot.markdown(
                        build_loading_html(title, detail),
                        unsafe_allow_html=True,
                    )
                    status.write(f"{title}: {detail}")

                result = chatbot.answer(
                    prompt,
                    history=st.session_state.messages[:-1],
                    llm_provider=llm_provider,
                    progress=update_progress,
                )
                status.update(label="Answer ready", state="complete", expanded=False)
            loading_slot.empty()
            render_chat_message("assistant", result["answer"], result["sources"])
            with st.expander("Retrieved context"):
                st.caption(f"Search query: {result['search_query']}")
                for chunk in result["retrieved_chunks"]:
                    title = chunk["metadata"].get("source_title", "Unknown")
                    source_url = chunk["metadata"].get("source_url")
                    source_line = f"**{title}** | score `{chunk['score']:.4f}`"
                    if source_url:
                        source_line += f" | [Wikipedia source]({source_url})"
                    st.markdown(source_line)
                    st.write(
                        chunk["text"][:900]
                        + ("..." if len(chunk["text"]) > 900 else "")
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )
        st.session_state.last_sources = result["sources"]
        st.session_state.last_prompt = prompt
        st.session_state.last_search_query = result["search_query"]
        st.rerun()
    except FileNotFoundError:
        with main_col:
            st.error("FAISS index not found. Build the index from the command deck first.")
    except Exception as exc:
        with main_col:
            st.error(f"Answer generation failed: {exc}")
