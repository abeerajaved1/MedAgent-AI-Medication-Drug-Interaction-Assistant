"""
Central configuration for MedAgent.
Reads everything from environment variables (.env in local/docker runs).
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    GROK_API_KEY: str = os.getenv("GROK_API_KEY", "")
    GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-2-latest")
    GROK_BASE_URL: str = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")

    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "./data/medagent.db")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))

    EXTERNAL_SOURCES_ENABLED: bool = os.getenv("EXTERNAL_SOURCES_ENABLED", "true").lower() == "true"
    EXTERNAL_TIMEOUT_SECONDS: float = float(os.getenv("EXTERNAL_TIMEOUT_SECONDS", "6"))


settings = Settings()
