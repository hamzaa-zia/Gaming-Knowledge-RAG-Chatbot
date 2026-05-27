import re

from src.config import DEFAULT_SENTENCE_LIMIT, DEFAULT_TOP_K
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
    "retrieved from",
    "retrieved ",
    "archived ",
    "isbn",
    "doi:",
    "external links",
    "references",
    "further reading",
)


def keywords(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    return {token for token in tokens if token not in GENERIC_TERMS}


def expand_query(question: str) -> str:
    lower_question = question.lower()
    expansions = []
    if "what is gaming" in lower_question or lower_question.strip() == "gaming":
        expansions.append("definition video game electronic game player interaction gameplay")
    if "evolved" in lower_question or "history" in lower_question:
        expansions.append("history early arcade console computer mobile online esports")
    if "open world" in lower_question:
        expansions.append("player autonomy freedom exploration nonlinear sandbox objectives")
    if "aaa" in lower_question or "triple-a" in lower_question:
        expansions.append("high budget marketing publisher franchise blockbuster revenue")
    if "market" in lower_question or "revenue" in lower_question or "industry" in lower_question:
        expansions.append("video game industry global sales revenue market billion")
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
    if "aaa" in lower_query or "triple-a" in lower_query:
        if any(term in lower_sentence for term in ("budget", "marketing", "publisher", "franchise", "revenue")):
            score += 1.3
    if "market" in lower_query or "revenue" in lower_query:
        if any(term in lower_sentence for term in ("revenue", "sales", "market", "billion", "global")):
            score += 1.5
    return score


def is_answer_sentence(sentence: str) -> bool:
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
    if "museum" in lower_sentence and "history" in lower_sentence:
        return False
    if len(sentence.split()) < 8:
        return False
    if sentence and sentence[0].islower():
        return False
    return True


def clean_answer_sentence(sentence: str) -> str:
    sentence = re.sub(r"=+\s*([^=]{1,80})\s*=+", r"\1", sentence)
    sentence = re.sub(r"\[[^\]]{1,10}\]", "", sentence)
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
    ) -> dict:
        history = history or []
        search_query = build_search_query(question, history)
        if "what is gaming" in question.lower():
            top_k = max(top_k, 12)
            sentence_limit = min(sentence_limit, 3)
        retrieved_chunks = self.vector_store.search(search_query, top_k=top_k)

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

        if not selected_sentences or all(result["score"] <= 0 for result in retrieved_chunks):
            answer = (
                "I could not find a direct answer in the indexed Wikipedia corpus. "
                "Try asking with more specific gaming terms or rebuild the index with more sources."
            )
        else:
            answer = self._compose_answer(question, selected_sentences)

        return {
            "answer": answer,
            "search_query": search_query,
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
                    "The sources do not prove that open-world games are the most popular, "
                    "but they explain why the format is appealing:"
                )
            else:
                opening = "The retrieved Wikipedia context points to these main reasons:"
        elif "market" in lower_question or "total" in lower_question:
            opening = "The indexed Wikipedia material describes the gaming market like this:"
        elif "evolved" in lower_question or "history" in lower_question:
            opening = "The retrieved history-focused sources show this evolution:"
        else:
            opening = "Based on the indexed Wikipedia sources:"

        lines = [opening]
        for sentence, result in selected_sentences:
            title = result["metadata"].get("source_title", "Wikipedia source")
            lines.append(f"- {sentence} [{title}]")
        return "\n".join(lines)

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
                    "source": metadata.get("source", ""),
                    "page": metadata.get("page"),
                    "chunk": metadata.get("chunk_index"),
                    "score": round(result["score"], 4),
                }
            )
        return sources
