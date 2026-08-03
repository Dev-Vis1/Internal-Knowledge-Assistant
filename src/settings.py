from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class AppSettings:
    project_name: str
    data_dir: Path
    persist_dir: Path
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    mlflow_enabled: bool
    mlflow_tracking_uri: str
    mlflow_experiment_name: str


def _resolve_path(value: str, default: str) -> Path:
    path_value = Path(value or default)
    if path_value.is_absolute():
        return path_value
    return (REPO_ROOT / path_value).resolve()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return default


def load_settings(config_path: Path | None = None) -> AppSettings:
    path = config_path or CONFIG_PATH
    raw_settings: dict[str, Any] = {}

    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            raw_settings = loaded

    return AppSettings(
        project_name=str(raw_settings.get("project_name", "internal-knowledge-assistant")),
        data_dir=_resolve_path(str(raw_settings.get("data_dir", "./data")), "./data"),
        persist_dir=_resolve_path(str(raw_settings.get("persist_dir", "./faiss_store")), "./faiss_store"),
        embedding_model=str(raw_settings.get("embedding_model", "all-MiniLM-L6-v2")),
        llm_model=str(raw_settings.get("llm_model", "llama-3.1-8b-instant")),
        chunk_size=_as_int(raw_settings.get("chunk_size"), 500),
        chunk_overlap=_as_int(raw_settings.get("chunk_overlap"), 20),
        mlflow_enabled=_as_bool(raw_settings.get("mlflow_enabled"), True),
        mlflow_tracking_uri=str(raw_settings.get("mlflow_tracking_uri", "sqlite:///mlflow.db")),
        mlflow_experiment_name=str(raw_settings.get("mlflow_experiment_name", "internal-knowledge-assistant")),
    )