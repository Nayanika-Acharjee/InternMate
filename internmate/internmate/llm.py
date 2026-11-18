from functools import lru_cache

from internmate.config import get_settings


class MissingAPIKey(RuntimeError):
    pass


@lru_cache
def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    s = get_settings()
    if not s.google_api_key:
        raise MissingAPIKey("Set GOOGLE_API_KEY in .env to use the chat agent.")
    return ChatGoogleGenerativeAI(model=s.llm_model, google_api_key=s.google_api_key, temperature=0.2)


def message_text(msg) -> str:
    """Gemini may return content as a list of parts; flatten to plain text."""
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
