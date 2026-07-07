from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_extraction_model: str = Field(
        default="llama-3.1-8b-instant",
        validation_alias="GROQ_EXTRACTION_MODEL",
    )
    groq_planner_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="GROQ_PLANNER_MODEL",
    )
    groq_interview_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="GROQ_INTERVIEW_MODEL",
    )
    groq_evaluation_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="GROQ_EVALUATION_MODEL",
    )

    voice_record_seconds: int = Field(default=5, ge=1, le=60, validation_alias="VOICE_RECORD_SECONDS")
    voice_max_record_seconds: int = Field(
        default=45,
        ge=3,
        le=180,
        validation_alias="VOICE_MAX_RECORD_SECONDS",
    )
    voice_silence_seconds: float = Field(
        default=1.4,
        ge=0.3,
        le=5.0,
        validation_alias="VOICE_SILENCE_SECONDS",
    )
    voice_vad_threshold: float = Field(
        default=0.015,
        ge=0.001,
        le=0.5,
        validation_alias="VOICE_VAD_THRESHOLD",
    )
    voice_sample_rate: int = Field(default=16000, ge=8000, validation_alias="VOICE_SAMPLE_RATE")
    voice_tts_voice: str = Field(default="en-US-JennyNeural", validation_alias="VOICE_TTS_VOICE")

    whisper_model_size: str = Field(default="base", validation_alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="cpu", validation_alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", validation_alias="WHISPER_COMPUTE_TYPE")
    whisper_language: str = Field(default="en", validation_alias="WHISPER_LANGUAGE")


@lru_cache
def get_settings() -> Settings:
    return Settings()
