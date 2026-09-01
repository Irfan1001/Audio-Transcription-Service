from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables (or a local .env file)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    whisper_model: str = "large-v3"
    whisper_device: str = "auto"
    whisper_compute_type: str = "default"
    preload_model: bool = True

    chunk_duration_seconds: int = 30
    max_file_size_mb: int = 100
    max_transcription_retries: int = 3
    retry_delay_seconds: float = 1.0
    max_concurrent_transcriptions: int = 1

    log_level: str = "INFO"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance (also used as a FastAPI dependency)."""
    return Settings()
