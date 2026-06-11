# 🎮 Gaming Knowledge RAG Chatbot

A conversational Retrieval-Augmented Generation (RAG) chatbot for gaming knowledge. It answers questions about gaming history, AAA games, open-world games, genres, esports, game design, cloud gaming, mobile games, and the gaming market using a local Wikipedia-based corpus.

The core system works locally with FAISS retrieval and extractive answer generation. Gemini API mode is optional and is used only after FAISS retrieves relevant context.

---

## 🚀 Project Overview

This project demonstrates how a chatbot can remember conversation context, retrieve external knowledge from a vectorized document store, and answer with source-aware responses.

Main capabilities:

- 📚 Wikipedia-only gaming knowledge base
- 🧹 PDF/API text cleaning and chunking
- 🔎 Local embeddings with `sentence-transformers`
- ⚡ FAISS semantic search over the local vector index
- 🧠 Conversation-aware query building for follow-up prompts
- 💬 Concise conversational paragraph answers
- 🤖 Optional Gemini API generation over retrieved context
- 🛡️ Gemini retry handling with local FAISS fallback
- 🕹️ Custom arcade-style Streamlit dashboard interface
- 🧾 Left retrieval feed with current prompt, ranked sources, scores, chunks, and links

---

## 🆕 Latest Project Updates

| Area | What Changed |
|---|---|
| Corpus depth | The knowledge base was expanded and rebuilt with deeper gaming topics such as esports, game design, game mechanics, AI in games, battle royale, MOBA, speedrunning, and level design. |
| Retrieval engine | The project now uses local embeddings plus FAISS instead of keyword-only retrieval. |
| Current index | The latest index contains `35` sources and `2307` chunks using `sentence-transformers/all-MiniLM-L6-v2`. |
| Gemini mode | Gemini is now the only API provider. It receives only retrieved Wikipedia context, retries transient failures, and falls back to the local FAISS answer engine if the API fails. |
| UI | The Streamlit app was redesigned into a custom arcade dashboard with hidden Streamlit chrome, improved chat bubbles, a left retrieval rail, a right command deck, and loading/status states. |
| Answer style | Answers are formatted as concise conversational paragraphs instead of bullet-heavy responses. |
| Cleanup | Unused dependency noise was removed, `numpy` was added explicitly, runtime logs/cache files were cleaned, and comments were added around important RAG sections. |
| Diagram | The Excalidraw workflow diagram and PNG were rebuilt to show the current FAISS + Gemini + Streamlit workflow clearly. |

---

## 🧭 Project Workflow Diagram

![Gaming RAG workflow](docs/rag_workflow.png)

---

## 🧰 Tools and Libraries Used

| Library / Tool | Function in This Project |
|---|---|
| `streamlit` | Builds the desktop-style chatbot interface. Key functions used include `st.set_page_config()`, `st.columns()`, `st.chat_input()`, `st.session_state`, `st.button()`, `st.radio()`, `st.checkbox()`, `st.status()`, `st.expander()`, and custom `st.markdown()` HTML/CSS. |
| `pypdf` | Extracts text from Wikipedia PDF files. `PdfReader` opens PDFs and `page.extract_text()` reads each page. |
| `sentence-transformers` | Loads the free local embedding model. `SentenceTransformer` converts document chunks and user questions into semantic vectors. |
| `faiss-cpu` | Stores and searches the local vector index. `IndexFlatIP`, `faiss.write_index()`, `faiss.read_index()`, and `index.search()` are used for cosine-style semantic retrieval. |
| `numpy` | Converts embedding arrays into `float32` format required by FAISS. |
| `requests` | Fetches optional refreshed Wikipedia article text through API requests. |
| `google-genai` | Connects to Gemini. `genai.Client()` and `client.models.generate_content()` generate a conversational answer from retrieved context. |
| `python-dotenv` | Loads `GEMINI_API_KEY` and `GEMINI_MODEL` from `.env` or `.env.txt` with `load_dotenv()`. |
| `json` | Saves chunks, source logs, embedding config, and index metadata in readable files. |
| `pathlib` | Handles project paths safely across folders. |
| `urllib.parse` | Builds browser-safe Wikipedia URLs from article titles. |
| `re` | Cleans text, removes citation noise, splits sentences, extracts keywords, and applies retrieval heuristics. |
| `argparse` | Adds indexing command options such as `--refresh-wikipedia`, `--chunk-size`, and `--chunk-overlap`. |
| `os` | Reads Gemini-related environment variables. |
| `datetime` and `time` | Create UTC metadata timestamps and add small delays between Wikipedia API requests. |
| `html.escape` | Safely renders prompts, source names, scores, and links inside custom Streamlit HTML. |
| `sys` | Uses the active Python executable when the app launches indexing through `subprocess`. |
| `subprocess` | Lets the Streamlit command deck run `ingest.py` from inside the app. |

---

## 🧠 Techniques Used

### 1. Retrieval-Augmented Generation

The chatbot does not answer from model memory alone. It first retrieves relevant chunks from the local Wikipedia corpus, then builds an answer from that evidence.

### 2. Local Embeddings + FAISS

Each cleaned chunk is converted into a dense vector using `sentence-transformers/all-MiniLM-L6-v2`. FAISS stores those vectors locally and searches for the chunks most similar to the user question.

### 3. Source-Aware Gemini Generation

Gemini does not search the documents. FAISS searches first, then Gemini receives the question, recent conversation, and retrieved Wikipedia chunks. Gemini's job is to rewrite the retrieved evidence into a smoother conversational answer.

### 4. Local Fallback Answering

If Gemini is unavailable, overloaded, rate-limited, or misconfigured, the project still answers using the local FAISS retrieval path. This keeps the demo reliable for internship and university submission.

### 5. Corpus Cleaning

The cleaning pipeline removes common Wikipedia/PDF noise before indexing:

- Citation markers such as `[12]`
- Reference and external-link sections
- Raw URLs, DOI fragments, ISBN fragments, and category footer text
- Broken PDF line breaks and hyphenated words
- Extra whitespace and punctuation spacing issues

### 6. Overlapping Text Chunking

Long documents are split into manageable chunks with overlap. This keeps each chunk small enough for retrieval while reducing the chance that useful information is cut off between chunks.

### 7. Source Metadata Tracking

Each chunk keeps metadata:

- `source_title`
- `source_url`
- `source_kind`
- `page`
- `chunk_index`

This lets the UI show ranked Wikipedia sources, confidence labels, chunk numbers, and clickable source links.

### 8. Query Expansion

`src/rag_chatbot.py` expands short gaming questions with useful related terms. For example, open-world questions can include terms like autonomy, freedom, exploration, nonlinear, and sandbox.

### 9. Conversational Context Memory

The app stores chat history in `st.session_state`. For follow-up prompts, the latest question can be combined with recent conversation context before retrieval.

### 10. Sentence Scoring and Answer Selection

The local extractive mode ranks answer sentences using keyword overlap, topic-specific boosts, source relevance, and filters for noisy sentences. The final local answer is formatted as one concise conversational paragraph.

### 11. Custom Streamlit Interface

The interface is customized beyond a default chatbot:

- Hidden Streamlit chrome
- Arcade neon dashboard theme
- Left retrieval rail for the current prompt
- Central chat workspace
- Right command deck for answer engine and index rebuild
- Custom chat bubbles
- Compact prompt bar
- Source confidence labels and Wikipedia links
- Loading/status cards during retrieval and generation

---

## ⚙️ How the Code Works

1. `ingest.py` starts the indexing workflow.
2. `src/wiki_fetcher.py` optionally downloads updated Wikipedia article text when `--refresh-wikipedia` is used.
3. `src/document_loader.py` loads PDFs and refreshed text files from `data/raw/`, extracts text, and attaches source metadata.
4. `src/text_processing.py` cleans the text and creates overlapping chunks.
5. `src/vector_store.py` embeds chunks with `SentenceTransformer`, builds the FAISS index, and saves `faiss.index`, `chunks.json`, `embedding_config.json`, and `index_metadata.json`.
6. `app.py` loads the saved FAISS index through `LocalVectorStore`.
7. The user asks a question through the Streamlit chat input or a quick-launch button.
8. `src/rag_chatbot.py` builds a conversation-aware search query, expands it when useful, retrieves top chunks from FAISS, ranks answer-ready sentences, and creates a local answer.
9. If Gemini mode is selected, `src/llm_client.py` sends only the retrieved Wikipedia context to Gemini.
10. If Gemini fails after retrying, `src/rag_chatbot.py` falls back to the local FAISS answer engine.
11. `app.py` displays the answer, current retrieval query, ranked sources, confidence labels, source links, and retrieved context.

---

## 🔎 FAISS vs Gemini in This Project

| Component | Role |
|---|---|
| FAISS | Searches the local vector index and finds the most relevant chunks from the gaming corpus. |
| Gemini | Writes a smoother conversational answer using only the chunks retrieved by FAISS. |
| Local fallback | Builds an answer directly from retrieved chunks when Gemini is unavailable. |
| Streamlit | Presents the chat interface, command deck, loading states, and retrieval feed. |

Simple flow:

```text
User question
-> Sentence-transformer query embedding
-> FAISS semantic search
-> Ranked Wikipedia chunks
-> Gemini generation or local extractive fallback
-> Conversational answer + source links
```

---

## 📁 Project Structure

```text
.
|-- app.py                         # Streamlit chatbot interface
|-- ingest.py                      # Builds/rebuilds the FAISS index
|-- fetch_wikipedia_sources.py     # Refreshes configured Wikipedia text only
|-- requirements.txt               # Python dependencies
|-- run_app.bat                    # Windows shortcut to run the app
|-- .env.example                   # Gemini environment variable template
|-- .streamlit/
|   `-- config.toml                # Streamlit local settings
|-- data/
|   |-- raw/                       # Wikipedia PDFs and optional refreshed text
|   |   `-- wikipedia_updates/     # generated by refresh, ignored by Git
|   `-- processed/
|       |-- wikipedia_sources.json
|       `-- vector_index/          # generated FAISS files, ignored by Git
|-- docs/
|   |-- rag_workflow.excalidraw
|   |-- rag_workflow.png
|   `-- project_post_visual.png
`-- src/
    |-- config.py
    |-- document_loader.py
    |-- llm_client.py
    |-- rag_chatbot.py
    |-- text_processing.py
    |-- vector_store.py
    `-- wiki_fetcher.py
```

---

## 📚 Knowledge Base

The corpus uses Wikipedia-only gaming sources. Current indexed topics include:

- Gaming and video games
- History of video games and early video game history
- AAA games
- Open-world games
- Video game genres
- Action, action-adventure, adventure, racing, horror, sports, RPG, shooter, strategy, puzzle, platformer, and fighting games
- Cloud gaming
- Mobile games
- Indie games and live-service games
- Esports and esports game lists
- Battle royale and MOBA games
- Artificial intelligence in video games
- Game design, game mechanics, level design, and speedrunning
- Video game industry and market-related pages
- Best-selling games and largest game companies by revenue

Current index snapshot:

| Item | Value |
|---|---:|
| Indexed source titles | `35` |
| Created text chunks | `2307` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimensions | `384` |
| Vector index | `FAISS IndexFlatIP` |
| Score type | `cosine_similarity` |

---

## ▶️ How to Run Locally

Recommended setup:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Build the local vector index:

```powershell
python ingest.py
```

Optionally refresh configured Wikipedia text before indexing:

```powershell
python ingest.py --refresh-wikipedia
```

Run the Streamlit app:

```powershell
python -m streamlit run app.py
```

On Windows, you can also run:

```text
run_app.bat
```

Notes:

- The first run may download the local sentence-transformer model.
- The project can run without Gemini in local extractive mode.
- If the FAISS files are already present in `data/processed/vector_index/`, the app can load them directly.

---

## 🔐 Optional Gemini Setup

Gemini mode is optional. Local extractive mode works without any API key.

Create `.env` from the example:

```powershell
Copy-Item .env.example .env
```

Add your Gemini settings:

```env
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-3.5-flash
```

Secrets are ignored by Git through `.gitignore`.

---

## 💬 Example Questions

- What is gaming?
- How has gaming evolved?
- Why are open-world games popular?
- Why do AAA games sell so much?
- What is the total market of gaming?
- What are the main video game genres?
- What is cloud gaming?
- What are racing video games?
- How are mobile games different from console games?
- What genre in gaming requires the most skill?
- How does AI affect video games?
- Why are esports games so competitive?

---

## 📦 Generated Output Files

After indexing, the project creates:

```text
data/processed/vector_index/chunks.json
data/processed/vector_index/faiss.index
data/processed/vector_index/embedding_config.json
data/processed/vector_index/index_metadata.json
data/processed/wikipedia_sources.json
```

These files are generated artifacts. They are useful for running the app locally, but `data/processed/` is ignored by Git to avoid committing large rebuildable files.

---

## ✅ What This Project Demonstrates

- How RAG improves chatbot answers by grounding them in retrieved documents
- How to build a free local semantic retrieval system without paid embedding APIs
- How to clean and chunk a custom PDF/text corpus
- How to persist and reuse a local FAISS vector index
- How to add conversational memory with Streamlit session state
- How to add source transparency with links, scores, and retrieved chunks
- How Gemini can improve answer wording without replacing retrieval
- How to keep the app usable when the API fails by falling back to local retrieval
- How to build a more polished Streamlit interface for an AI course or internship submission

---

## ⚠️ Limitations

- Local extractive mode is reliable and free, but less fluent than a strong LLM.
- Gemini mode requires a valid API key and can be rate-limited.
- Market-size answers are limited to figures present in the indexed Wikipedia text.
- The first FAISS index build downloads the embedding model if it is not already cached.
- Wikipedia refresh requests can be rate-limited; skipped pages are logged and the index still builds from available sources.
- The project uses a direct Python RAG pipeline rather than LangChain wrappers.

---

## 🔮 Future Improvements

- Add reranking on top of FAISS retrieval for stronger source selection.
- Add prepared evaluation questions for retrieval and answer quality testing.
- Add optional ChromaDB support for richer metadata filtering.
- Add a stronger LLM or local LLM backend after the core RAG pipeline is stable.
- Package the Streamlit app into a desktop-style executable after the RAG workflow is finalized.
