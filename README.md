# Gaming Knowledge RAG Chatbot

A gaming RAG chatbot that answers questions about gaming history, AAA games, open-world games, genres, and the video game industry using Wikipedia PDFs and refreshed Wikipedia text.

The app supports two answer modes: a free local extractive mode and a Gemini API mode that generates cleaner answers from the retrieved Wikipedia context.

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
- `google-genai`: calls Gemini models with the official Google GenAI SDK when Gemini mode is enabled.
- `python-dotenv`: loads API settings from `.env` or `.env.txt` without hardcoding secrets in the code.
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
5. `src/rag_chatbot.py` receives a user question, expands the search query when useful, retrieves the most relevant chunks, and either builds an extractive answer or sends the retrieved context to Gemini.
6. `src/llm_client.py` loads API settings, formats chat history and retrieved context, and asks the selected model to answer only from the provided sources.
7. `app.py` provides the Streamlit chat interface and stores conversation history in `st.session_state`.

---

## Project Structure

```text
.
|-- app.py
|-- ingest.py
|-- fetch_wikipedia_sources.py
|-- requirements.txt
|-- run_app.bat
|-- .env.example
|-- data/
|   |-- raw/
|   |   |-- AAA_(video_game_industry).pdf
|   |   |-- Action_game.pdf
|   |   |-- Adventure_game.pdf
|   |   |-- Early_history_of_video_games.pdf
|   |   |-- Fighting_game.pdf
|   |   |-- Gaming.pdf
|   |   |-- Indie_game.pdf
|   |   |-- Live_service_game.pdf
|   |   |-- Mobile_game.pdf
|   |   |-- Open_world.pdf
|   |   |-- Platformer.pdf
|   |   |-- Puzzle_video_game.pdf
|   |   |-- Role-playing_video_game.pdf
|   |   |-- Shooter_game.pdf
|   |   |-- Strategy_video_game.pdf
|   |   |-- Video_game.pdf
|   |   `-- Video_game_industry.pdf
|   `-- processed/
|       `-- vector_index/
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

Optional LLM API mode:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` with your API settings:

```env
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-3.5-flash
```

For Gemini, `GEMINI_API_KEY` and `GEMINI_MODEL` are preferred.

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

- Loaded document sections/pages: `359`
- Created text chunks: `1938`
- Indexed source titles: `20`

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
- What are role-playing games?
- How are mobile games different from console games?

---

## Limitations

- Extractive mode is free and local but less fluent than LLM mode.
- Gemini mode requires a valid API key and compatible model configuration.
- If Wikipedia rate-limits a refresh request, the script skips that page and continues building the index from available sources.
- Market-size answers are limited to what is present in the indexed Wikipedia text.

---

## Future Improvements

- Add HuggingFace sentence-transformer embeddings or ChromaDB for semantic vector search.
- Add answer quality evaluation with prepared test questions.
- Package the Streamlit app into a desktop-style executable after the RAG workflow is finalized.
