from langchain_groq import ChatGroq

from app.config import Settings, get_settings

_instance = None


def get_llm(settings: Settings | None = None) -> ChatGroq:
    global _instance
    if _instance is None:
        s = settings or get_settings()
        _instance = ChatGroq(model=s.groq_model, temperature=0, api_key=s.groq_api_key or None)
    return _instance
