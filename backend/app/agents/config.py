from functools import lru_cache
import os
from pathlib import Path
import stat
from urllib.parse import urlsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


MAX_OAUTH_TIMEOUT_SECONDS = 300


class AgentSettings(BaseSettings):
    # Master switch. When false, the router is not mounted, MCP servers are
    # not contacted, and no background tasks run.
    enabled: bool = False

    # Built-in MCP server URL. The mcp-server container speaks JSON-RPC 2.0.
    # This is the internal address the backend uses to reach the container.
    builtin_mcp_url: str = "http://mcp-server:8765/mcp"

    # Public URL external agents use to reach the built-in MCP server, shown
    # in the UI token panel. Leave blank to have the frontend derive it from
    # the browser location (``<protocol>//<hostname>:8765/mcp``). Set this
    # when the MCP server is exposed behind an ingress/reverse proxy on a
    # custom host, subpath, or standard 80/443 port instead of ``:8765``.
    # Example: "https://securo.example.com/mcp".
    external_mcp_url: str = ""

    # Comma-separated extra MCP servers users can plug in (URL[|name]).
    # Example: "http://my-tools:9000/mcp|my-tools,http://other:9001/mcp"
    extra_mcp_servers: str = ""

    # Shared secret used to mint short-lived JWTs for MCP calls. Distinct
    # from the main app secret so revocation is independent.
    mcp_jwt_secret: str = "change-me-in-production"
    mcp_jwt_ttl_seconds: int = 600

    # TTL for long-lived tokens minted via the UI for external agents
    # (Claude Desktop, n8n, custom clients). The feature itself follows
    # `enabled` — if agents are on, the mint endpoint is mounted and the
    # mcp-server container publishes port 8765.
    mcp_external_ttl_days: int = 90

    # Embedding dimension for the knowledge_chunks vector column. Locked at
    # migration time. 1536 covers OpenAI text-embedding-3-small (default) and
    # nomic-embed-text via Matryoshka padding/truncation.
    embedding_dim: int = 1536

    # Default RAG parameters (overridable per agent in the DB row).
    default_top_n: int = 6
    default_similarity_threshold: float = 0.25

    # Embedding provider/model. Both MCP-server and Celery worker honor
    # these. Switching requires re-embedding existing docs (their
    # embeddings are tied to the model that produced them).
    #
    # Default `native` uses fastembed with a small multilingual model
    # bundled in-process — works zero-config (no Ollama/OpenAI/etc).
    # Users can switch to ollama/openai/openai_compatible for higher
    # quality or faster inference.
    embedding_provider: str = "native"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_native_cache_dir: str = "/app/data/embedding_models"
    embedding_ollama_base_url: str = "http://ollama:11434"
    embedding_openai_base_url: str = "https://api.openai.com/v1"
    embedding_openai_api_key: str = ""

    # Sign in with ChatGPT OAuth egress is isolated from ambient proxy/CA
    # environment variables. Deployments must opt in here instead.
    siwc_oauth_proxy_url: str = ""
    siwc_oauth_ca_bundle: str = ""
    siwc_oauth_timeout_seconds: float = 30.0
    siwc_oauth_connect_timeout_seconds: float = 5.0

    @field_validator("siwc_oauth_proxy_url")
    @classmethod
    def validate_siwc_oauth_proxy_url(cls, value: str) -> str:
        if not value:
            return value
        if value != value.strip() or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ValueError("SIWC OAuth proxy URL contains whitespace or control characters")
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("SIWC OAuth proxy URL is invalid") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or "@" in parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or port == 0
        ):
            raise ValueError("SIWC OAuth proxy URL must be an HTTP(S) origin without credentials")
        return value

    @field_validator("siwc_oauth_ca_bundle")
    @classmethod
    def validate_siwc_oauth_ca_bundle(cls, value: str) -> str:
        if not value:
            return value
        if value != value.strip() or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        ):
            raise ValueError("SIWC OAuth CA bundle path contains whitespace or control characters")
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("SIWC OAuth CA bundle path must be absolute")
        try:
            details = path.lstat()
        except (OSError, ValueError) as exc:
            raise ValueError("SIWC OAuth CA bundle is not readable") from exc
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise ValueError("SIWC OAuth CA bundle must be a regular non-symlink file")
        if details.st_mode & 0o444 == 0 or not os.access(path, os.R_OK):
            raise ValueError("SIWC OAuth CA bundle is not readable")
        try:
            with path.open("rb") as bundle:
                prefix = bundle.read(64)
        except OSError as exc:
            raise ValueError("SIWC OAuth CA bundle is not readable") from exc
        if b"-----BEGIN CERTIFICATE-----" not in prefix:
            raise ValueError("SIWC OAuth CA bundle is not PEM encoded")
        return value

    @field_validator("siwc_oauth_timeout_seconds", "siwc_oauth_connect_timeout_seconds")
    @classmethod
    def validate_siwc_oauth_timeout(cls, value: float) -> float:
        if not 0 < value <= MAX_OAUTH_TIMEOUT_SECONDS:
            raise ValueError("SIWC OAuth timeout must be between 0 and 300 seconds")
        return value

    # Where uploaded knowledge files live on disk (per-instance).
    knowledge_storage_path: str = "/app/data/agent_knowledge"
    knowledge_max_file_size_mb: int = 25

    # Same env_file pair as the main Settings: the CWD-relative ".env" for
    # backward compatibility plus the anchored backend/.env, so the API and the
    # Celery worker/beat read the same file whatever directory they start from.
    model_config = SettingsConfigDict(
        env_file=(".env", Path(__file__).resolve().parents[2] / ".env"),
        env_prefix="AGENTS_",
        extra="ignore",
    )


@lru_cache
def get_agent_settings() -> AgentSettings:
    return AgentSettings()
