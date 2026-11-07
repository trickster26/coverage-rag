"""Settings, all from environment variables (a .env file is loaded if present).

Two switches decide where things run:
  LLM_PROVIDER     azure | openai | none   (none = retrieval only, no generation)
  SEARCH_BACKEND   azure | local
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "none"))
    search_backend: str = field(default_factory=lambda: _env("SEARCH_BACKEND", "local"))

    # Azure OpenAI
    azure_openai_endpoint: str = field(default_factory=lambda: _env("AZURE_OPENAI_ENDPOINT"))
    azure_openai_api_key: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_KEY"))
    azure_openai_api_version: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_VERSION", "2024-10-21"))
    azure_chat_deployment: str = field(default_factory=lambda: _env("AZURE_OPENAI_CHAT_DEPLOYMENT"))
    azure_embedding_deployment: str = field(default_factory=lambda: _env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))

    # OpenAI-compatible (local mode)
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    openai_chat_model: str = field(default_factory=lambda: _env("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_embedding_model: str = field(default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    # Azure AI Search
    azure_search_endpoint: str = field(default_factory=lambda: _env("AZURE_SEARCH_ENDPOINT"))
    azure_search_api_key: str = field(default_factory=lambda: _env("AZURE_SEARCH_API_KEY"))
    azure_search_index: str = field(default_factory=lambda: _env("AZURE_SEARCH_INDEX", "coverage-chunks"))
    azure_search_semantic_config: str = field(default_factory=lambda: _env("AZURE_SEARCH_SEMANTIC_CONFIG"))

    # Retrieval
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    candidates: int = field(default_factory=lambda: int(_env("CANDIDATES", "50")))
    # Below this best-match score the question is treated as not answerable from the documents.
    min_score: float = field(default_factory=lambda: float(_env("MIN_SCORE", "0.50")))

    @property
    def embeddings_enabled(self) -> bool:
        if self.llm_provider == "azure":
            return bool(self.azure_embedding_deployment)
        # Empty OPENAI_EMBEDDING_MODEL turns vectors off for chat-only providers (Groq, most NVIDIA chat setups).
        return self.llm_provider == "openai" and bool(self.openai_embedding_model)


settings = Settings()
