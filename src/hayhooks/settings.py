from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class AppSettings(BaseSettings):
    # Root path for the FastAPI app
    root_path: str = ""

    # Path to the folder containing the pipelines
    pipelines_dir: str = "pipelines"

    # Additional Python path to be added to the Python path
    additional_python_path: str = ""

    # Host for the FastAPI app
    host: str = "localhost"

    # Port for the FastAPI app
    port: int = 1416

    @field_validator("pipelines_dir")
    def validate_pipelines_dir(cls, v):
        path = Path(v)

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        if not path.is_dir():
            raise ValueError(f"pipelines_dir '{v}' exists but is not a directory")

        return str(path)


settings = AppSettings()
