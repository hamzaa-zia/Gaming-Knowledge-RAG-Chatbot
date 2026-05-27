import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.config import INDEX_METADATA_PATH, ROOT_DIR
from src.rag_chatbot import RetrievalChatbot
from src.vector_store import LocalVectorStore, load_index_metadata


st.set_page_config(
    page_title="Gaming Knowledge RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.5rem;
        max-width: 1180px;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #56616f;
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stSidebar"] {
        background: #f6f7f9;
    }
    .source-box {
        border: 1px solid #d9dee7;
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        margin-bottom: 0.5rem;
        background: #ffffff;
        font-size: 0.88rem;
    }
    .metric-row {
        border: 1px solid #e0e5ee;
        border-radius: 8px;
        padding: 0.7rem;
        background: white;
    }
</style>
"""


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask me about gaming history, AAA games, open-world games, "
                    "live-service games, indie games, or the video game industry."
                ),
            }
        ]
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


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


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("No sources retrieved yet.")
        return
    for source in sources:
        page = f", page {source['page']}" if source.get("page") else ""
        st.markdown(
            f"""
            <div class="source-box">
                <strong>{source['title']}</strong>{page}<br>
                Score: {source['score']}<br>
                <span>{Path(source['source']).name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


initialize_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Knowledge Base")
    metadata = load_index_metadata()
    if metadata:
        st.success("Vector index ready")
        st.write(f"Sources: {metadata.get('source_count', 0)}")
        st.write(f"Chunks: {metadata.get('chunk_count', 0)}")
        st.caption(f"Built: {metadata.get('built_at_utc', 'Unknown')}")
    else:
        st.warning("Index not built yet")

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

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    st.divider()
    st.subheader("Latest Sources")
    render_sources(st.session_state.last_sources)

st.markdown('<div class="main-title">Gaming Knowledge RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Local retrieval-based chatbot using Wikipedia PDFs and refreshed Wikipedia text.</div>',
    unsafe_allow_html=True,
)

if not INDEX_METADATA_PATH.exists():
    st.info("Build the index from the sidebar before asking questions.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about gaming...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        chatbot = load_chatbot()
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant Wikipedia context..."):
                result = chatbot.answer(prompt, history=st.session_state.messages[:-1])
            st.markdown(result["answer"])
            with st.expander("Retrieved context"):
                st.caption(f"Search query: {result['search_query']}")
                for chunk in result["retrieved_chunks"]:
                    title = chunk["metadata"].get("source_title", "Unknown")
                    st.markdown(f"**{title}** | score `{chunk['score']:.4f}`")
                    st.write(chunk["text"][:900] + ("..." if len(chunk["text"]) > 900 else ""))

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
        st.session_state.last_sources = result["sources"]
    except FileNotFoundError:
        st.error("Vector index not found. Build the index from the sidebar first.")
