# Configuration and Why It Matters in This Project

Configuration makes this project easier to run, tune, and maintain across environments. Instead of hardcoding paths and model settings inside Python files, the app reads a central YAML config so both local development and future deployment can follow the same setup pattern.

## How configuration fits this project

This project uses `config/settings.yaml` as the primary runtime configuration for the RAG pipeline.

The flow is simple:

1. The app loads settings from `config/settings.yaml`.
2. Paths, model options, and chunking values are applied to the RAG service.
3. MLflow settings are read from the same file.
4. The app runs with one consistent set of values.

## Fields used in this config

Current fields in `config/settings.yaml` include:

- `project_name`: logical project identifier used in the app and tracking logs.
- `data_dir`: source knowledge-base folder used for ingestion.
- `persist_dir`: local folder for persisted FAISS index files.
- `embedding_model`: sentence-transformer model used for vector embeddings.
- `llm_model`: model name used for answer generation via Groq.
- `chunk_size`: number of characters per text chunk.
- `chunk_overlap`: overlap between adjacent chunks.
- `mlflow_enabled`: enables or disables MLflow tracking.
- `mlflow_tracking_uri`: MLflow backend location (local `sqlite:///mlflow.db` by default).
- `mlflow_experiment_name`: experiment name used for logged runs.

## Files added and updated for config

These files now support configuration-driven behavior:

- `config/settings.yaml`: central config file for paths, models, chunking, and MLflow.
- `src/settings.py`: settings loader and typed settings object.
- `app/main.py`: uses config values when creating the RAG service.
- `src/search.py`: uses config defaults and MLflow config during runtime.

## Benefits for an MLOps workflow

For MLOps, central configuration helps teams keep behavior consistent while still allowing controlled changes. It becomes easier to:

- change model or chunking strategy without rewriting code
- align app behavior across different machines
- connect observability tools like MLflow through config
- reduce hidden environment-specific differences
