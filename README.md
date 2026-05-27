# Gaming Knowledge RAG Chatbot

A local retrieval-based chatbot that answers questions about gaming history, AAA games, open-world games, indie games, live-service games, and the video game industry using Wikipedia PDFs and refreshed Wikipedia text.

This first version focuses on the core RAG workflow without an LLM: it retrieves relevant document chunks from a local vector index and builds concise extractive answers with source references.

---

## Overview

The project objective is to build a conversational chatbot that can remember chat context and retrieve external information during conversation.

The knowledge base uses Wikipedia-only sources:

- Local Wikipedia PDFs in `data/raw/`
- Optional refreshed Wikipedia text downloaded by `ingest.py --refresh-wikipedia`

The chatbot is deployed with Streamlit and runs locally.

---

## Tools and Libraries

- `streamlit`: builds the chatbot interface. Main functions used are `st.chat_input()` for user input, `st.chat_message()` for chat bubbles, `st.session_state` for memory, `st.sidebar` for controls, and `st.status()` for indexing feedback.
- `pypdf`: reads text from Wikipedia PDF files. `PdfReader` opens each PDF and `page.extract_text()` extracts page text.
- `scikit-learn`: creates the local vectorized document store. `TfidfVectorizer` converts text chunks into vectors, and `cosine_similarity` compares the user query vector with stored document vectors.
- `joblib`: saves and loads the vectorizer and TF-IDF matrix so the index can be reused without rebuilding every time.
- `requests`: fetches updated plain-text extracts from the Wikipedia API.
- `json`: stores chunk metadata, source logs, and index metadata in readable files.
- `pathlib`: handles project paths safely across Windows folders.
- `re`: cleans text, splits sentences, extracts keywords, and normalizes answer text.
- `subprocess`: lets the Streamlit sidebar run the ingestion script from inside the app.

---

## Workflow

1. `fetch_wikipedia_sources.py` or `python ingest.py --refresh-wikipedia` downloads updated text from configured Wikipedia pages.
2. `src/document_loader.py` loads PDF, TXT, and Markdown files from `data/raw/`.
3. `src/text_processing.py` cleans text and splits it into overlapping chunks.
4. `src/vector_store.py` converts chunks into TF-IDF vectors and saves the local vector index in `data/processed/vector_index/`.
5. `src/rag_chatbot.py` receives a user question, expands the search query when useful, retrieves the most relevant chunks, selects useful sentences, and returns an answer with source names.
6. `app.py` provides the Streamlit chat interface and stores conversation history in `st.session_state`.

---

## Project Structure

```text
.
├── app.py
├── ingest.py
├── fetch_wikipedia_sources.py
├── requirements.txt
├── run_app.bat
├── data/
│   ├── raw/
│   │   ├── AAA_(video_game_industry).pdf
│   │   ├── Early_history_of_video_games.pdf
│   │   ├── Gaming.pdf
│   │   ├── Indie_game.pdf
│   │   ├── Live_service_game.pdf
│   │   ├── Open_world.pdf
│   │   ├── Video_game.pdf
│   │   └── Video_game_industry.pdf
│   └── processed/
│       └── vector_index/
└── src/
    ├── config.py
    ├── document_loader.py
    ├── rag_chatbot.py
    ├── text_processing.py
    ├── vector_store.py
    └── wiki_fetcher.py
```

---

## How to Run

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Build the vector index from local PDFs:

```powershell
python ingest.py
```

Build the vector index and refresh Wikipedia text first:

```powershell
python ingest.py --refresh-wikipedia
```

Run the Streamlit chatbot:

```powershell
python -m streamlit run app.py
```

On Windows, you can also double-click:

```text
run_app.bat
```

---

## Current Output

The current index was built from the provided PDFs and refreshed Wikipedia text.

- Loaded document sections/pages: `172`
- Created text chunks: `1131`
- Indexed source titles: `11`

Generated files:

- `data/processed/vector_index/chunks.json`
- `data/processed/vector_index/tfidf_vectorizer.joblib`
- `data/processed/vector_index/tfidf_matrix.joblib`
- `data/processed/vector_index/index_metadata.json`
- `data/processed/wikipedia_sources.json`

---

## Example Questions

- What is gaming?
- How has gaming evolved?
- Why are open-world games popular?
- Why do AAA games sell so much?
- What is the total market of gaming?

---

## Limitations

- This version is retrieval-based and does not use an LLM.
- Answers are extractive, so they are built from retrieved Wikipedia sentences instead of generated from a language model.
- If Wikipedia rate-limits a refresh request, the script skips that page and continues building the index from available sources.
- Market-size answers are limited to what is present in the indexed Wikipedia text.

---

## Future Improvements

- Replace extractive answering with a local or API-based LLM.
- Add HuggingFace sentence-transformer embeddings or ChromaDB for semantic vector search.
- Add answer quality evaluation with prepared test questions.
- Package the Streamlit app into a desktop-style executable after the RAG workflow is finalized.
