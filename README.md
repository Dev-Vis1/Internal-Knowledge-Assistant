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

- app.py: End-to-end pipeline run
- src/data_loader.py: Multi-format document loading
- src/embedding.py: Chunking and embedding generation
- notebook/rag_pipeline.ipynb: Interactive RAG workflow
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

3. Run the pipeline:

```bash
python app.py
```

## Notes

- Keep input files inside folders like data/pdf_files and data/txt_files.
- Vector store files are persisted under data/vector_store.
