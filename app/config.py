import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

class Config:
    """System configuration and environment loader."""
    
    # Optional LLM API keys; blank values keep requests on the local fallback.
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet").strip()

    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0").strip()
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Business Logic & Guardrail Thresholds (Chapter 2, 4, 5, 8)
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    RAG_CUTOFF_THRESHOLD: float = float(os.getenv("RAG_CUTOFF_THRESHOLD", "0.15"))
    KILL_SWITCH_ACTIVE: bool = os.getenv("KILL_SWITCH_ACTIVE", "false").lower() == "true"
    
    @classmethod
    def is_live_openai_available(cls) -> bool:
        return bool(cls.OPENAI_API_KEY and len(cls.OPENAI_API_KEY) > 10 and not cls.OPENAI_API_KEY.startswith("your_"))

    @classmethod
    def is_live_gemini_available(cls) -> bool:
        return bool(cls.GEMINI_API_KEY and len(cls.GEMINI_API_KEY) > 10 and not cls.GEMINI_API_KEY.startswith("your_"))

    @classmethod
    def is_live_anthropic_available(cls) -> bool:
        return bool(cls.ANTHROPIC_API_KEY and len(cls.ANTHROPIC_API_KEY) > 10 and not cls.ANTHROPIC_API_KEY.startswith("your_"))

    @classmethod
    def active_provider_mode(cls) -> str:
        if cls.KILL_SWITCH_ACTIVE:
            return "KILL_SWITCH_ENGAGED"
        if cls.is_live_openai_available():
            return "LIVE_OPENAI"
        return "HEURISTIC_LOCAL_FALLBACK"

settings = Config()
