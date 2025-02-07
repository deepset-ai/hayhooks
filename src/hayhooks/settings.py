from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


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


settings = AppSettings()
