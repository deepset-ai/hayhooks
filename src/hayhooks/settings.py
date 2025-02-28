from typing import Union
from warnings import warn
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

load_dotenv(dotenv_path=find_dotenv(usecwd=True))


class AppSettings(BaseSettings):
    # Root path for the FastAPI app
    root_path: str = ""

    # Path to the directory containing the pipelines
    # Default to project root / pipelines
    pipelines_dir: str = str(Path(__file__).parent.parent.parent / "pipelines")

    # Additional Python path to be added to the Python path
    additional_python_path: str = ""

    # Host for the FastAPI app
    host: str = "localhost"

    # Port for the FastAPI app
    port: int = 1416

    # Disable SSL verification when making requests from the CLI
    disable_ssl: bool = False

    # Files to ignore when reading pipeline files from a directory
    files_to_ignore_patterns: list[str] = ["*.pyc", "*.pyo", "*.pyd", "__pycache__", "*.so", "*.egg", "*.egg-info"]

    # Show tracebacks on errors during pipeline execution and deployment
    show_tracebacks: bool = False

    # CORS Settings
    cors_allow_origins: list[str] = ["*"]
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    cors_allow_credentials: bool = False
    cors_allow_origin_regex: Union[str, None] = None
    cors_expose_headers: list[str] = []
    cors_max_age: int = 600

    # Prefix for the environment variables to avoid conflicts
    # with other similar environment variables
    model_config = SettingsConfigDict(env_prefix='hayhooks_')


settings = AppSettings()


def check_cors_settings():
    if (
        settings.cors_allow_origins == ["*"]
        and settings.cors_allow_methods == ["*"]
        and settings.cors_allow_headers == ["*"]
    ):
        warn("Using default CORS settings - All origins, methods, and headers are allowed.")
