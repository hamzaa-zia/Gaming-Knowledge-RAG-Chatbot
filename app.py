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
    initial_sidebar_state="expanded",
)


ASSISTANT_AVATAR = ":material/sports_esports:"
USER_AVATAR = ":material/terminal:"


TOPIC_BUTTONS = [
    ("History", "How has gaming evolved?", ":material/history:"),
    ("AAA", "Why do AAA games sell so much?", ":material/diamond:"),
    ("Open World", "Why are open world games popular?", ":material/travel_explore:"),
    ("Mobile", "How are mobile games different from console games?", ":material/smartphone:"),
    ("RPG", "What are role-playing games?", ":material/auto_stories:"),
    ("Market", "What is the total market of gaming?", ":material/query_stats:"),
]


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

    [data-testid="stHeader"] {
        background: rgba(9, 10, 18, 0.86);
        border-bottom: 1px solid rgba(0, 245, 255, 0.18);
    }

    .block-container {
        padding-top: 1.1rem;
        max-width: 1280px;
        padding-bottom: 5rem;
    }

    div[data-testid="stSidebar"] {
        background: #0d101b;
        border-right: 1px solid var(--line);
    }

    div[data-testid="stSidebar"] * {
        color: var(--text);
    }

    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stSidebar"] .stCaptionContainer {
        color: var(--muted);
    }

    .arcade-topbar {
        border: 1px solid var(--line);
        border-left: 4px solid var(--cyan);
        background: linear-gradient(135deg, rgba(17, 20, 33, 0.96), rgba(23, 27, 43, 0.92));
        padding: 1rem 1.1rem;
        border-radius: 8px;
        box-shadow: 0 0 28px rgba(0, 245, 255, 0.1);
        margin-bottom: 0.95rem;
    }

    .title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .main-title {
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: 0;
        margin: 0;
        color: var(--text);
        text-shadow: 0 0 16px rgba(0, 245, 255, 0.34);
    }

    .subtitle {
        color: var(--muted);
        font-size: 0.95rem;
        margin-top: 0.35rem;
    }

    .status-strip {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    .status-pill {
        border: 1px solid rgba(184, 255, 77, 0.38);
        color: var(--lime);
        background: rgba(184, 255, 77, 0.08);
        border-radius: 999px;
        padding: 0.35rem 0.65rem;
        font-size: 0.8rem;
        white-space: nowrap;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.6rem;
        margin: 0.9rem 0 1rem;
    }

    .metric-row {
        border: 1px solid rgba(255, 204, 51, 0.28);
        border-radius: 8px;
        padding: 0.72rem 0.8rem;
        background: rgba(17, 20, 33, 0.92);
        min-height: 72px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        color: var(--amber);
        font-size: 1.35rem;
        font-weight: 760;
        margin-top: 0.18rem;
    }

    .section-label {
        color: var(--cyan);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0.9rem 0 0.45rem;
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
    }

    .chat-frame {
        border-top: 1px solid var(--line);
        padding-top: 0.75rem;
    }

    [data-testid="stChatMessage"] {
        background: rgba(17, 20, 33, 0.9);
        border: 1px solid rgba(0, 245, 255, 0.18);
        border-radius: 8px;
        box-shadow: 0 0 18px rgba(0, 245, 255, 0.06);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-color: rgba(0, 245, 255, 0.28);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-color: rgba(255, 47, 146, 0.3);
    }

    [data-testid="stChatInput"] {
        background: rgba(9, 10, 18, 0.94);
        border-top: 1px solid rgba(0, 245, 255, 0.2);
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
        min-height: 2.35rem;
        box-shadow: inset 0 0 0 1px rgba(255, 47, 146, 0.08);
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
        .metric-grid {
            grid-template-columns: 1fr;
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
        timeout=180,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output


def build_sources_html(sources: list[dict]) -> str:
    if not sources:
        return '<div class="empty-source">No sources retrieved yet.</div>'

    items = []
    for source in sources:
        page = f", page {escape(str(source['page']))}" if source.get("page") else ""
        title = escape(str(source.get("title", "Unknown")))
        score = escape(str(source.get("score", "")))
        source_file = escape(Path(source.get("source", "")).name)
        items.append(
            f'<div class="source-box">'
            f"<strong>{title}</strong>{page}<br>"
            f"Score: {score}<br>"
            f"<span>{source_file}</span>"
            f"</div>"
        )
    return "\n".join(items)


def render_sources(sources: list[dict]) -> None:
    st.markdown(build_sources_html(sources), unsafe_allow_html=True)


initialize_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
metadata = load_index_metadata()

with st.sidebar:
    st.markdown('<div class="section-label">Knowledge Core</div>', unsafe_allow_html=True)
    if metadata:
        st.success("Vector index ready")
        st.write(f"Sources: {metadata.get('source_count', 0)}")
        st.write(f"Chunks: {metadata.get('chunk_count', 0)}")
        st.caption(f"Built: {metadata.get('built_at_utc', 'Unknown')}")
    else:
        st.warning("Index not built yet")

    st.divider()
    st.markdown('<div class="section-label">Answer Engine</div>', unsafe_allow_html=True)
    gemini_ready = has_gemini_key()
    gemini_warning = get_gemini_warning() if gemini_ready else ""
    default_engine = 1 if gemini_ready and not gemini_warning else 0
    answer_engine = st.radio(
        "Mode",
        ["Extractive", "Gemini API"],
        index=default_engine,
        help="Extractive is local. Gemini API uses retrieved context plus your Gemini key.",
    )
    answer_style = st.selectbox("Answer Style", ["Concise", "Detailed"], index=0)
    if answer_engine == "Gemini API" and gemini_ready:
        st.caption(f"Gemini model: {get_gemini_model_name()}")
        if gemini_warning:
            st.warning(gemini_warning)
    elif answer_engine == "Gemini API":
        st.warning("GEMINI_API_KEY was not found in .env or .env.txt.")

    llm_provider = "Extractive"
    if answer_engine == "Gemini API" and gemini_ready and not gemini_warning:
        llm_provider = "Gemini"

    if answer_engine == "Gemini API" and gemini_warning:
        st.warning("Gemini API mode is disabled until the model configuration is compatible.")

    st.divider()
    refresh = st.checkbox("Refresh Wikipedia text before indexing", value=False)
    if st.button("Build / Rebuild Index", use_container_width=True):
        with st.status("Building local vector index...", expanded=True) as status:
            ok, output = run_ingestion(refresh)
            if ok:
                load_chatbot.clear()
                status.update(label="Index built successfully", state="complete")
                st.code(output)
            else:
                status.update(label="Index build failed", state="error")
                st.code(output)

    if st.button("Reset Run", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.divider()
    st.markdown('<div class="section-label">Source Feed</div>', unsafe_allow_html=True)
    render_sources(st.session_state.last_sources)

main_col, source_col = st.columns([3.25, 1], gap="large")

with main_col:
    st.markdown(
        f"""
        <div class="arcade-topbar">
            <div class="title-row">
                <div>
                    <div class="main-title">Arcade RAG Console</div>
                    <div class="subtitle">Gaming knowledge retrieval from the local Wikipedia corpus.</div>
                </div>
                <div class="status-strip">
                    <span class="status-pill">LOCAL INDEX</span>
                    <span class="status-pill">MEMORY ON</span>
                    <span class="status-pill">WIKI CORPUS</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metadata:
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric-row">
                    <div class="metric-label">Sources</div>
                    <div class="metric-value">{metadata.get("source_count", 0)}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">Chunks</div>
                    <div class="metric-value">{metadata.get("chunk_count", 0)}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">Mode</div>
                    <div class="metric-value">{llm_provider if llm_provider != "Extractive" else "RAG"}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Quick Launch</div>', unsafe_allow_html=True)
    quick_cols = st.columns(len(TOPIC_BUTTONS))
    for column, (label, question, icon) in zip(quick_cols, TOPIC_BUTTONS):
        with column:
            if st.button(label, icon=icon, use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()

    if not INDEX_METADATA_PATH.exists():
        st.info("Build the index from the sidebar before asking questions.")

    st.markdown('<div class="chat-frame">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    st.markdown("</div>", unsafe_allow_html=True)

with source_col:
    source_panel_html = (
        '<div class="dock-panel">'
        '<div class="dock-title">Retrieval Feed</div>'
        f"{build_sources_html(st.session_state.last_sources)}"
        '<div class="dock-title">Corpus Channels</div>'
        '<div class="topic-list">'
        '<span class="topic-chip">AAA</span>'
        '<span class="topic-chip">Open World</span>'
        '<span class="topic-chip">Mobile</span>'
        '<span class="topic-chip">RPG</span>'
        '<span class="topic-chip">Action</span>'
        '<span class="topic-chip">Market</span>'
        "</div>"
        "</div>"
    )
    st.markdown(
        source_panel_html,
        unsafe_allow_html=True,
    )

prompt = st.session_state.pending_prompt or st.chat_input(
    "Enter command: ask about gaming history, genres, market, or design."
)
if prompt:
    st.session_state.pending_prompt = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    with main_col:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

    try:
        chatbot = load_chatbot()
        with main_col:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                with st.spinner("Retrieving relevant Wikipedia context..."):
                    result = chatbot.answer(
                        prompt,
                        history=st.session_state.messages[:-1],
                        llm_provider=llm_provider,
                        answer_style=answer_style,
                    )
                st.markdown(result["answer"])
                with st.expander("Retrieved context"):
                    st.caption(f"Search query: {result['search_query']}")
                    for chunk in result["retrieved_chunks"]:
                        title = chunk["metadata"].get("source_title", "Unknown")
                        st.markdown(f"**{title}** | score `{chunk['score']:.4f}`")
                        st.write(
                            chunk["text"][:900]
                            + ("..." if len(chunk["text"]) > 900 else "")
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
        st.session_state.last_sources = result["sources"]
        st.rerun()
    except FileNotFoundError:
        with main_col:
            st.error("Vector index not found. Build the index from the sidebar first.")
    except Exception as exc:
        with main_col:
            st.error(f"API answer mode failed: {exc}")
