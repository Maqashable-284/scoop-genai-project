"""
Configuration for Scoop GenAI - Google Gemini SDK Implementation
Answers Question #5: Production Considerations & #6: Security
"""
import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    """Application settings with production defaults"""

    # Google Gemini API
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # MongoDB
    mongodb_uri: str = Field(default_factory=lambda: os.getenv("MONGODB_URI", ""))
    mongodb_database: str = Field(default_factory=lambda: os.getenv("MONGODB_DATABASE", "scoop_db"))

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Model Configuration
    # Question #5: Rate Limits for Gemini 2.5 Flash:
    # - Free tier: 15 RPM, 1M TPM, 1500 RPD
    # - Paid tier: 2000 RPM, 4M TPM (standard), scales with billing
    model_name: str = "gemini-2.5-flash"  # Production stable (Gemini 3 is too slow)

    # Session & Memory
    # Question #1: Memory Persistence - Session TTL
    session_ttl_seconds: int = 3600  # 1 hour (longer than Claude version)

    # Question #1: Token Limit Management
    # Gemini 2.5 Flash context: 1M tokens input, but recommend limiting for cost
    max_history_messages: int = 100  # Sliding window trigger
    max_history_tokens: int = 50000  # When to summarize

    # Catalog
    # Question #3: 315 products ~60k tokens
    catalog_cache_ttl_seconds: int = 3600  # 1 hour cache

    # Rate Limiting
    rate_limit_per_minute: int = 30

    # CORS - Use env var for production restriction, default "*" for dev
    allowed_origins: str = Field(default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "*"))

    # Question #6: Security - Content filtering
    enable_safety_settings: bool = True

    # Security: Admin token for protected endpoints
    admin_token: Optional[str] = Field(default_factory=lambda: os.getenv("ADMIN_TOKEN"))

    class Config:
        env_file = ".env"


# System Prompt - Imported from prompts module
# Full style guide: docs/RESPONSE_STYLE_GUIDE.md
from prompts import SYSTEM_PROMPT


settings = Settings()
