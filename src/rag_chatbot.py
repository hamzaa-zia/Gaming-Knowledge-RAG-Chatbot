import re
from collections.abc import Callable

from src.config import DEFAULT_SENTENCE_LIMIT, DEFAULT_TOP_K
from src.llm_client import generate_gemini_answer
from src.text_processing import split_sentences
from src.vector_store import LocalVectorStore


GENERIC_TERMS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "who",
    "is",
    "are",
    "was",
    "were",
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "about",
    "games",
    "game",
    "gaming",
    "popular",
    "most",
    "much",
    "many",
    "total",
    "sell",
    "sold",
}

REFERENCE_MARKERS = (
    "http",
    "htt ps",
    "web.archive",
    "retrieved from",
    "retrieved ",
    "archived ",
    "isbn",
    "doi:",
    "video:",
    "gamasutra",
    "external links",
    "references",
    "further reading",
)

MIN_RETRIEVAL_SCORE = 0.18


def keywords(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    return {token for token in tokens if token not in GENERIC_TERMS}


def is_skill_or_competitive_query(text: str) -> bool:
    # Skill questions are usually comparative/opinion-like, so retrieval needs
    # competitive gaming terms instead of only generic "genre" terms.
    return bool(
        re.search(
            r"\b(skill|skilled|mastery|mechanic|mechanical|competitive|competition|"
            r"esports?|tournament|hard|difficult|difficulty|challenging|ranked)\b",
            text.lower(),
        )
    )


def expand_query(question: str) -> str:
    lower_question = question.lower()
    expansions = []
    if "what is gaming" in lower_question or lower_question.strip() == "gaming":
        expansions.append("definition video game electronic game player interaction gameplay")
    if "evolved" in lower_question or "history" in lower_question:
        expansions.append("history early arcade console computer mobile online esports")
    if "open world" in lower_question:
        expansions.append("player autonomy freedom exploration nonlinear sandbox objectives")
    if is_skill_or_competitive_query(question):
        expansions.append(
            "competitive video game genres esports tournament professional players skill mastery "
            "mechanical skill reflexes strategy teamwork fighting game shooter multiplayer "
            "online battle arena moba battle royale speedrunning game mechanics difficulty"
        )
    elif "genre" in lower_question or "genres" in lower_question:
        expansions.append("video game genre action adventure role-playing shooter racing sports puzzle strategy")
    if "action-adventure" in lower_question or "action adventure" in lower_question:
        expansions.append("action-adventure game exploration combat puzzles story")
    if "cloud" in lower_question or "streaming" in lower_question:
        expansions.append("cloud gaming streaming remote server latency subscription")
    if "horror" in lower_question:
        expansions.append("horror game survival horror fear atmosphere tension")
    if "racing" in lower_question:
        expansions.append("racing video game vehicle driving simulation arcade racing")
    if "sports" in lower_question:
        expansions.append("sports video game simulation team sports competition")
    if "aaa" in lower_question or "triple-a" in lower_question:
        expansions.append("high budget marketing publisher franchise blockbuster revenue")
    if "market" in lower_question or "revenue" in lower_question or "industry" in lower_question:
        expansions.append(
            "video game industry global sales revenue market billion international world trends 2025 Newzoo"
        )
    return f"{question} {' '.join(expansions)}".strip()


def build_search_query(question: str, history: list[dict]) -> str:
    expanded_question = expand_query(question)
    if not history:
        return expanded_question

    pronoun_pattern = r"\b(it|they|them|that|these|those|their|its)\b"
    follow_up_pattern = r"\b(tell me more|explain more|go deeper|what about|how about)\b"
    is_follow_up = bool(re.search(pronoun_pattern, question.lower())) or bool(
        re.search(follow_up_pattern, question.lower())
    )
    if not is_follow_up:
        return expanded_question

    previous_user_questions = [
        message["content"]
        for message in history
        if message.get("role") == "user" and message.get("content")
    ]
    if not previous_user_questions:
        return expanded_question

    return expand_query(f"{previous_user_questions[-1]} {question}")


def sentence_score(
    sentence: str, query_terms: set[str], base_score: float, search_query: str
) -> float:
    # FAISS ranks chunks; this second pass ranks individual sentences inside
    # those chunks so local extractive answers stay concise.
    sentence_terms = keywords(sentence)
    overlap = len(query_terms & sentence_terms)
    density = overlap / max(len(sentence_terms), 1)
    score = base_score + overlap * 0.4 + density

    lower_sentence = sentence.lower()
    lower_query = search_query.lower()
    if sentence and sentence[0].islower():
        score -= 0.9
    if "what is" in lower_query:
        if lower_sentence.startswith("a video game") or lower_sentence.startswith("video games are"):
            score += 3.0
        if " is an " in lower_sentence or " is a " in lower_sentence:
            score += 1.2
    if "what are" in lower_query:
        if " is an " in lower_sentence or " is a " in lower_sentence:
            score += 1.4
    if "evolved" in lower_query or "history" in lower_query:
        if re.search(r"\b(19|20)\d{2}\b", sentence):
            score += 0.8
        if any(
            term in lower_sentence
            for term in ("first", "grew", "crash", "arcade", "console", "mobile")
        ):
            score += 0.7
        if any(
            term in lower_sentence
            for term in ("taiwan", "germany", "canada", "china", "india", "south korea")
        ):
            score -= 1.4
    if "open world" in lower_query:
        if any(term in lower_sentence for term in ("autonomy", "freedom", "exploration", "nonlinear", "sandbox")):
            score += 1.4
    skill_query = is_skill_or_competitive_query(lower_query)
    if "genre" in lower_query:
        if not skill_query and any(
            term in lower_sentence
            for term in ("classification", "how it is played", "gameplay interaction")
        ):
            score += 2.0
        if any(term in lower_sentence for term in ("computer gaming world", "freedom planet", "mini metro")):
            score -= 1.2
    if skill_query:
        if any(
            term in lower_sentence
            for term in (
                "esports",
                "competitive",
                "competition",
                "tournament",
                "professional",
                "skill",
                "mastery",
                "mechanics",
                "difficulty",
                "speedrunning",
                "players compete",
                "ranked",
            )
        ):
            score += 2.0
        if any(
            term in lower_sentence
            for term in (
                "fighting game",
                "shooter",
                "multiplayer online battle arena",
                "moba",
                "battle royale",
                "strategy",
                "sports video game",
            )
        ):
            score += 1.0
        if any(term in lower_sentence for term in ("classification", "informal classification")):
            score -= 1.6
    if "cloud" in lower_query or "streaming" in lower_query:
        if any(term in lower_sentence for term in ("remote servers", "streams", "cloud computing", "latency")):
            score += 1.8
    if "horror" in lower_query:
        if any(term in lower_sentence for term in ("survival horror", "fear", "horror themes", "atmosphere")):
            score += 1.4
    if "racing" in lower_query:
        if any(term in lower_sentence for term in ("racing game", "racing video game", "driving", "vehicle")):
            score += 1.5
        if "racing games are a video game genre" in lower_sentence:
            score += 2.2
        if lower_sentence.startswith(("list of", "formula one video games", "ascar video games")):
            score -= 3.0
    if "sports" in lower_query:
        if any(term in lower_sentence for term in ("sports video game is", "simulates", "team sports", "competition")):
            score += 1.7
        if "racing" in lower_sentence:
            score -= 2.5
    if "action-adventure" in lower_query or "action adventure" in lower_query:
        if any(term in lower_sentence for term in ("action/adventure", "action-adventure", "puzzles", "problem-solving")):
            score += 1.5
    if "aaa" in lower_query or "triple-a" in lower_query:
        if any(term in lower_sentence for term in ("budget", "marketing", "publisher", "franchise", "revenue")):
            score += 1.3
    if "market" in lower_query or "revenue" in lower_query:
        if any(term in lower_sentence for term in ("revenue", "sales", "market", "billion", "global")):
            score += 1.5
        if any(
            phrase in lower_sentence
            for phrase in (
                "international video game revenue",
                "global revenue",
                "global video game market",
                "worldwide sales",
                "largest video game markets",
            )
        ):
            score += 2.0
        if re.search(r"\b20(1[8-9]|2[0-6])\b", sentence) and "equivalent to" not in lower_sentence:
            score += 0.8
        if re.search(r"\b(19[5-9]\d|200\d)\b", sentence):
            score -= 1.2
        if "equivalent to" in lower_sentence:
            score -= 0.8
        if "following countries" in lower_sentence:
            score -= 1.0
        if "arcade" in lower_sentence and re.search(r"\b198\d\b", sentence):
            score -= 1.0
    return score


def is_answer_sentence(sentence: str) -> bool:
    # Filter noisy PDF/Wikipedia fragments before they can become final answers.
    lower_sentence = sentence.lower()
    if lower_sentence.startswith(
        ("both of which", "of which", "these bugs", "first-generation pong console")
    ):
        return False
    if not sentence.endswith((".", "!", "?")):
        return False
    if any(marker in lower_sentence for marker in REFERENCE_MARKERS):
        return False
    if "computerspielemuseum" in lower_sentence:
        return False
    if "may refer to:" in lower_sentence:
        return False
    if lower_sentence.startswith(("list of", "formula one video games", "ascar video games")):
        return False
    if lower_sentence.count(" list of ") >= 2:
        return False
    if re.search(r"\b\d+/\S", lower_sentence):
        return False
    if "following countries" in lower_sentence:
        return False
    if "museum" in lower_sentence and "history" in lower_sentence:
        return False
    if "skill gallery" in lower_sentence:
        return False
    if lower_sentence.startswith("competitions in the genre"):
        return False
    if len(sentence.split()) < 8:
        return False
    if sentence and sentence[0].islower():
        return False
    return True


def clean_answer_sentence(sentence: str) -> str:
    sentence = re.sub(r"=+\s*([^=]{1,80})\s*=+", r"\1", sentence)
    sentence = re.sub(r"\[[^\]]{1,10}\]", "", sentence)
    sentence = re.sub(
        r"^(AAA \(video game industry\)|Open world|Video game|Gaming|Cloud gaming|Horror game|Racing video game|Sports video game)\s+(?=(A|An|The|In|It|This|These)\b)",
        "",
        sentence,
        flags=re.IGNORECASE,
    )
    sentence = re.sub(
        r"^(Cloud gaming|Horror game|Racing video game)\s+\1\b",
        r"\1",
        sentence,
        flags=re.IGNORECASE,
    )
    sentence = re.sub(
        r"^Sports video game\s+(A sports video game is\b)",
        r"\1",
        sentence,
        flags=re.IGNORECASE,
    )
    sentence = re.sub(
        r"^.*?\b(A video game genre is an informal classification\b)",
        r"\1",
        sentence,
        flags=re.IGNORECASE,
    )
    sentence = re.sub(
        r"^.*?\b(Racing games are a video game genre\b)",
        r"\1",
        sentence,
        flags=re.IGNORECASE,
    )
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if len(sentence) > 620:
        sentence = sentence[:620].rsplit(" ", 1)[0].rstrip(" ,;:") + "."
    return sentence


class RetrievalChatbot:
    def __init__(self, vector_store: LocalVectorStore | None = None) -> None:
        self.vector_store = vector_store or LocalVectorStore()

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = DEFAULT_TOP_K,
        sentence_limit: int = DEFAULT_SENTENCE_LIMIT,
        llm_provider: str = "Extractive",
        progress: Callable[[str, str], None] | None = None,
    ) -> dict:
        def report(title: str, detail: str) -> None:
            if progress:
                progress(title, detail)

        history = history or []
        report(
            "Reading the prompt",
            "Building a conversation-aware query from the latest question.",
        )
        search_query = build_search_query(question, history)
        sentence_limit = min(sentence_limit, 3)
        if "what is gaming" in question.lower():
            top_k = max(top_k, 12)
            sentence_limit = min(sentence_limit, 3)
        if is_skill_or_competitive_query(question):
            top_k = max(top_k, 14)
            sentence_limit = min(sentence_limit, 3)
        report(
            "Retrieving source context",
            "Searching the local Wikipedia chunks and ranking the strongest matches.",
        )
        retrieved_chunks = self.vector_store.search(search_query, top_k=top_k)

        report(
            "Checking source evidence",
            "Filtering retrieved chunks into answer-ready sentences.",
        )
        query_terms = keywords(search_query)
        scored_sentences = []
        for result in retrieved_chunks:
            for sentence in split_sentences(result["text"]):
                if not is_answer_sentence(sentence):
                    continue
                score = sentence_score(sentence, query_terms, result["score"], search_query)
                scored_sentences.append((score, sentence, result))

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        selected_sentences = []
        seen = set()
        for _, sentence, result in scored_sentences:
            cleaned_sentence = clean_answer_sentence(sentence)
            normalized = re.sub(r"[^a-z0-9]+", " ", cleaned_sentence.lower()).strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            selected_sentences.append((cleaned_sentence, result))
            if len(selected_sentences) >= sentence_limit:
                break

        top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0
        no_retrieval_match = not selected_sentences or top_score < MIN_RETRIEVAL_SCORE
        fallback_reason = ""
        if no_retrieval_match:
            answer = (
                "I could not find a solid match in the indexed gaming corpus yet. "
                "Try asking with a more specific gaming term, or rebuild the index with more sources."
            )
            mode = "extractive"
        elif llm_provider == "Gemini":
            report(
                "Generating source-aware answer",
                "Sending only the retrieved Wikipedia context to Gemini.",
            )
            fallback_reason = ""
            try:
                # If Gemini fails after retries, the same retrieved evidence is
                # still used to produce a local answer for stable demos.
                answer = generate_gemini_answer(
                    question=question,
                    history=history,
                    retrieved_chunks=retrieved_chunks,
                )
                answer = self._ensure_source_line(
                    answer,
                    [result for _, result in selected_sentences] or retrieved_chunks,
                )
                mode = "gemini"
            except Exception as exc:
                fallback_reason = str(exc)
                report(
                    "Using local FAISS fallback",
                    "Gemini did not respond reliably, so the local answer engine is taking over.",
                )
                local_answer = self._compose_answer(question, selected_sentences)
                answer = (
                    "Gemini could not complete this request, so I answered using local FAISS retrieval. "
                    f"{local_answer}"
                )
                mode = "extractive_fallback"
        else:
            report(
                "Composing source-aware answer",
                "Using the best retrieved sentences and attaching source titles.",
            )
            answer = self._compose_answer(question, selected_sentences)
            mode = "extractive"

        return {
            "answer": answer,
            "search_query": search_query,
            "mode": mode,
            "fallback_reason": fallback_reason,
            "sources": self._format_sources(
                [result for _, result in selected_sentences] or retrieved_chunks
            ),
            "retrieved_chunks": retrieved_chunks,
        }

    def _compose_answer(self, question: str, selected_sentences: list[tuple[str, dict]]) -> str:
        lower_question = question.lower()
        if "why" in lower_question:
            if "open world" in lower_question:
                opening = (
                    "Short answer: the corpus does not prove open-world games are the most popular, "
                    "but it does show why players are drawn to them."
                )
            else:
                opening = "Here is the quick breakdown from the gaming corpus."
        elif "market" in lower_question or "total" in lower_question:
            opening = "Here is the market picture the corpus gives."
        elif "evolved" in lower_question or "history" in lower_question:
            opening = "Here is the gaming history arc in simple terms."
        elif is_skill_or_competitive_query(question):
            opening = (
                "The corpus does not rank one genre as objectively requiring the most skill, "
                "but it points toward competitive genres and esports-heavy games."
            )
        else:
            opening = "Here is the quick answer from the gaming corpus."

        cited_sentences = []
        for sentence, result in selected_sentences:
            cited_sentences.append(sentence)
        answer = f"{opening} {' '.join(cited_sentences)}"
        return self._ensure_source_line(
            answer,
            [result for _, result in selected_sentences],
        )

    def _source_titles(self, retrieved_chunks: list[dict], limit: int = 3) -> list[str]:
        titles = []
        seen = set()
        for result in retrieved_chunks:
            title = result["metadata"].get("source_title", "Wikipedia source")
            normalized = title.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            titles.append(title)
            if len(titles) >= limit:
                break
        return titles

    def _ensure_source_line(self, answer: str, retrieved_chunks: list[dict]) -> str:
        answer = answer.strip()
        if re.search(r"(?im)^sources\s*:", answer):
            return answer
        titles = self._source_titles(retrieved_chunks)
        if not titles:
            return answer
        return f"{answer}\n\nSources: {'; '.join(titles)}"

    def _format_sources(self, retrieved_chunks: list[dict]) -> list[dict]:
        sources = []
        seen = set()
        for result in retrieved_chunks:
            metadata = result["metadata"]
            key = (
                metadata.get("source_title"),
                metadata.get("page"),
                metadata.get("chunk_index"),
            )
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "title": metadata.get("source_title", "Unknown"),
                    "source_url": metadata.get("source_url", ""),
                    "source_kind": metadata.get("source_kind", ""),
                    "source": metadata.get("source", ""),
                    "page": metadata.get("page"),
                    "chunk": metadata.get("chunk_index"),
                    "score": round(result["score"], 4),
                }
            )
        return sources
