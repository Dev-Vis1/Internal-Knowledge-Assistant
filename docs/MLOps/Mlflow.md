# MLflow and Why It Matters in This Project

MLflow helps make this RAG project observable and easier to improve over time. It records run metadata, parameters, and metrics so you can compare retrieval and answer behavior across configuration changes.

In a project like this internal knowledge assistant, that is important because quality depends on multiple moving parts such as chunk settings, embedding model, and retrieval behavior.

## How MLflow fits this project

In this project, MLflow is integrated into the search service and controlled through `config/settings.yaml`.

The flow is simple:

1. App settings are loaded from `config/settings.yaml`.
2. MLflow tracking URI and experiment name are configured.
3. The RAG pipeline logs runs for indexing, retrieval, and answer generation.
4. Results can be inspected in the MLflow tracking UI.

By default, this project uses a local SQLite tracking backend (`sqlite:///mlflow.db`) to stay compatible with current MLflow behavior.

## Files added and updated for MLflow

These files were added or updated to support MLflow tracking:

- `src/search.py`: logs MLflow runs for index build, retrieval, and answer stages.
- `src/settings.py`: exposes MLflow settings from YAML.
- `config/settings.yaml`: stores `mlflow_enabled`, `mlflow_tracking_uri`, and `mlflow_experiment_name`.
- `pyproject.toml`: includes `mlflow` as a project dependency.
- `requirements.txt`: includes `mlflow` for requirements-based installs.

## What is logged right now

The current implementation logs practical pipeline signals including:

- index build metrics: document count, chunk count, elapsed time
- retrieval metrics: query length, number of returned chunks, average score, elapsed time
- answer metrics: query length, context size, answer size, elapsed time
- core parameters: project name, top_k, model names, chunk settings

## Benefits for an MLOps workflow

For MLOps, MLflow improves traceability and iteration quality. It becomes easier to:

- compare retrieval behavior after config changes
- detect regressions in chunking or indexing strategy
- keep a history of how answers were produced
- support evidence-based tuning of RAG settings
