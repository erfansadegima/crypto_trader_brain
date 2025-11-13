"""Simple configuration using pydantic BaseSettings.

Values are loaded from environment variables. This module is optional; the
app uses a minimal approach and will still function if pydantic isn't
installed (it will fallback to reading `os.environ`).
"""
from typing import Optional
import os


class Settings:
    def __init__(self) -> None:
        self.binance_api_key: Optional[str] = os.environ.get("BINANCE_API_KEY")
        self.binance_secret: Optional[str] = os.environ.get("BINANCE_SECRET")
        self.execute_live: bool = os.environ.get("EXECUTE_LIVE", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        self.host: str = os.environ.get("BIND_HOST", "0.0.0.0")
        self.port: int = int(os.environ.get("BIND_PORT", "8000"))


settings = Settings()
