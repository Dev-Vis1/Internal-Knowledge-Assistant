import os
import pickle
import uuid
from typing import Any, Dict, List, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FaissVectorStore:
    """Simple FAISS-backed vector store for local RAG retrieval."""

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = int(self.model.get_embedding_dimension())

        self.index: faiss.Index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        print(f"[INFO] Loaded embedding model: {embedding_model}")
        print(f"[INFO] Embedding dimension: {self.embedding_dim}")

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine-similarity search with IndexFlatIP."""
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        normalized = embeddings.astype(np.float32)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return normalized / norms

    def _flatten_documents(self, documents: Any) -> List[Any]:
        """Accept either a flat list of Document objects or a dict of lists."""
        if isinstance(documents, dict):
            return [doc for docs_list in documents.values() for doc in docs_list]
        if isinstance(documents, list):
            return documents
        raise TypeError("documents must be a list or dict of lists.")

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> None:
        """Embed and add raw texts to the FAISS index."""
        if not texts:
            print("[WARN] No texts provided. Skipping add_texts.")
            return

        embeddings = self.model.encode(
            list(texts), convert_to_numpy=True, show_progress_bar=True
        )
        self.add_embeddings(embeddings, texts=texts, metadatas=metadatas, ids=ids)

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> None:
        """Add precomputed embeddings and related metadata to the index."""
        if len(texts) == 0:
            print("[WARN] No texts provided. Skipping add_embeddings.")
            return

        if embeddings.shape[0] != len(texts):
            raise ValueError("Embeddings and texts size mismatch.")

        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("Metadatas and texts size mismatch.")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if len(ids) != len(texts):
            raise ValueError("IDs and texts size mismatch.")

        normalized = self._normalize(embeddings)
        self.index.add(normalized)

        for i, text in enumerate(texts):
            metadata = dict(metadatas[i]) if metadatas is not None else {}
            metadata.setdefault("text", text)
            self.metadata.append(metadata)
            self.ids.append(ids[i])

        print(f"[INFO] Added {len(texts)} vectors. Total vectors: {self.index.ntotal}")

    def build_from_documents(self, documents: Any) -> None:
        """Build vector index directly from Document objects."""
        docs = self._flatten_documents(documents)
        if not docs:
            print("[WARN] No documents found. Nothing to index.")
            return

        texts = [doc.page_content for doc in docs]
        metadatas = [dict(getattr(doc, "metadata", {})) for doc in docs]

        self.add_texts(texts=texts, metadatas=metadatas)

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        state_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)
        with open(state_path, "wb") as f:
            pickle.dump({"metadata": self.metadata, "ids": self.ids}, f)

        print(f"[INFO] Saved index to: {faiss_path}")
        print(f"[INFO] Saved metadata to: {state_path}")

    def load(self) -> None:
        """Load FAISS index and metadata from disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        state_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(state_path)):
            raise FileNotFoundError(
                f"Missing persisted files in {self.persist_dir}. "
                "Expected faiss.index and metadata.pkl"
            )

        self.index = faiss.read_index(faiss_path)
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        self.metadata = state.get("metadata", [])
        self.ids = state.get("ids", [])

        print(f"[INFO] Loaded index from: {faiss_path}")
        print(f"[INFO] Loaded {self.index.ntotal} vectors.")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search top-k similar entries for the given query text."""
        if self.index.ntotal == 0:
            print("[WARN] Index is empty. Returning no results.")
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_embedding = self.model.encode(
            [query_text], convert_to_numpy=True, show_progress_bar=False
        )
        query_embedding = self._normalize(query_embedding)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append(
                {
                    "id": self.ids[idx] if idx < len(self.ids) else str(idx),
                    "score": float(score),
                    "metadata": self.metadata[idx],
                }
            )

        return results

