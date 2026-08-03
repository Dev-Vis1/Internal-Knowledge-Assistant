# Internal Knowledge Assistant

A lightweight Retrieval-Augmented Generation project for querying internal documents.

## What it does

- Loads company documents from local folders
- Splits text into chunks
- Generates embeddings using Sentence Transformers
- Stores vectors in Faiss & ChromaDB
- Retrieves relevant context for LLM responses

## Tech stack

- Python 3.12+
- LangChain
- Sentence Transformers
- ChromaDB, Faiss
- Groq API for LLM inference

## Project structure

- app/main.py: Streamlit UI entry point
- src/data_loader.py: Multi-format document loading
- src/embedding.py: Chunking and embedding generation
- src/search.py: Retrieval and answer generation
- config/: Basic runtime configuration
- tests/: Basic project checks
- docker/: Docker build files
- data/: Source documents and vector store data

## Quick start

1. Install dependencies:

```bash
uv sync
```

2. Add your API key to a .env file:

```env
GROQ_API_KEY=your_key_here
```

3. Run locally:

```bash
streamlit run app/main.py
```

## Run with Docker

Build and run the app in a container:

```bash
docker build -t internal-knowledge-assistant .
docker run -p 8501:8501 internal-knowledge-assistant
```

Or with Docker Compose:

```bash
docker compose up --build
```

The app will be available at `http://localhost:8501` and MLflow UI at `http://localhost:5000`.

## Notes

- Keep input files inside folders like data/pdf_files and data/txt_files.
- Vector store files are persisted under faiss_store by default.
