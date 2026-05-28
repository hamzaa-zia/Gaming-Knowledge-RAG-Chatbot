import os
import re

from dotenv import load_dotenv
from google import genai

from src.config import ROOT_DIR


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
MAX_CONTEXT_CHARS = 9000


def load_api_environment() -> None:
    for env_name in (".env", ".env.txt"):
        env_path = ROOT_DIR / env_name
        if env_path.exists():
            load_dotenv(env_path, override=False)


def get_gemini_api_key() -> str:
    load_api_environment()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    # Compatibility fallback for earlier local setup. Prefer GEMINI_API_KEY.
    openai_model = os.getenv("OPENAI_MODEL", "").lower()
    if "gemini" in openai_model:
        return os.getenv("OPENAI_API_KEY", "").strip()
    return ""


def has_gemini_key() -> bool:
    return bool(get_gemini_api_key())


def normalize_gemini_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        return DEFAULT_GEMINI_MODEL
    if normalized.lower() in {"gemini", "gemini 3.5", "gemini-3.5"}:
        return DEFAULT_GEMINI_MODEL
    return normalized


def get_gemini_model_name() -> str:
    load_api_environment()
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not model and "gemini" in os.getenv("OPENAI_MODEL", "").lower():
        model = os.getenv("OPENAI_MODEL", "")
    return normalize_gemini_model(model)


def get_gemini_warning() -> str:
    model = get_gemini_model_name()
    if not has_gemini_key():
        return "GEMINI_API_KEY was not found. Add it to .env or .env.txt."
    if re.search(r"\s", model):
        return "The Gemini model name contains spaces. Use a model ID like gemini-3.5-flash."
    return ""


def format_history(history: list[dict], limit: int = 6) -> str:
    recent_messages = history[-limit:]
    lines = []
    for message in recent_messages:
        role = message.get("role", "user")
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "No prior conversation."


def format_context(retrieved_chunks: list[dict]) -> str:
    sections = []
    total_chars = 0
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("source_title", "Wikipedia source")
        page = metadata.get("page")
        page_label = f", page {page}" if page else ""
        text = str(chunk.get("text", "")).strip()
        block = f"[{index}] {title}{page_label}\n{text}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        sections.append(block)
        total_chars += len(block)
    return "\n\n".join(sections)


def build_prompt(
    question: str,
    history: list[dict],
    retrieved_chunks: list[dict],
    answer_style: str = "Concise",
) -> tuple[str, str]:
    style_instruction = (
        "Answer in 4-6 direct bullet points."
        if answer_style == "Detailed"
        else "Answer in 2-4 short bullet points."
    )
    context = format_context(retrieved_chunks)
    chat_history = format_history(history)

    instructions = (
        "You are a gaming knowledge assistant for an internship and university AI "
        "course project. Answer only from the retrieved Wikipedia context. Do not "
        "invent facts, numbers, dates, or market claims. If the retrieved context "
        "does not contain enough evidence, say that clearly. Keep the answer clean, "
        "direct, and related to the user's question. End with a short 'Sources:' "
        "line listing the source titles used."
    )
    prompt = f"""
Question:
{question}

Recent conversation:
{chat_history}

Retrieved Wikipedia context:
{context}

Answer style:
{style_instruction}
"""
    return instructions, prompt


def generate_gemini_answer(
    question: str,
    history: list[dict],
    retrieved_chunks: list[dict],
    answer_style: str = "Concise",
) -> str:
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found in .env or .env.txt.")

    instructions, prompt = build_prompt(
        question=question,
        history=history,
        retrieved_chunks=retrieved_chunks,
        answer_style=answer_style,
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=get_gemini_model_name(),
        contents=f"{instructions}\n\n{prompt}",
    )
    return response.text.strip()
