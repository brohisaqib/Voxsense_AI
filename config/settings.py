# ============================================================
# VoxSense - Settings & Configuration Loader
# ============================================================

import os
from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "VoxSense Online AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # API Keys
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    deepgram_api_key: str = Field(default="", env="DEEPGRAM_API_KEY")

    # Backend
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")
    backend_url: str = Field(default="http://localhost:8000", env="BACKEND_URL")

    # Security
    secret_key: str = Field(default="changeme", env="SECRET_KEY")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")

    # Storage
    chroma_persist_dir: str = Field(default="./data/chroma", env="CHROMA_PERSIST_DIR")

    # Vision
    yolo_model: str = Field(default="yolov8n.pt", env="YOLO_MODEL")
    yolo_confidence: float = Field(default=0.5, env="YOLO_CONFIDENCE")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/voxsense.log", env="LOG_FILE")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"          # ✅ Yeh line fix karti hai — extra fields ignore hongi


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def load_config() -> dict:
    """Load YAML configuration file."""
    config_path = BASE_DIR / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r",encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# Global instances
settings = get_settings()
config = load_config()