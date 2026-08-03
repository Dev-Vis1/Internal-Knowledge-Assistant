import os
import time
from contextlib import nullcontext
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Any, Dict, List

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None

try:
    from src.vectorstore import FaissVectorStore
    from src.data_loader import load_all_documents_from_directory
    from src.embedding import EmbeddingPipeline
    from src.settings import load_settings
except ModuleNotFoundError:
    from vectorstore import FaissVectorStore
    from data_loader import load_all_documents_from_directory
    from embedding import EmbeddingPipeline
    from settings import load_settings

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        data_dir: str | None = None,
        persist_dir: str | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        settings = load_settings()
        self.project_name = settings.project_name

        self.data_dir = data_dir or str(settings.data_dir)
        self.persist_dir = persist_dir or str(settings.persist_dir)
        self.embedding_model = embedding_model or settings.embedding_model
        self.llm_model = llm_model or settings.llm_model
        resolved_chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        resolved_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.mlflow_enabled = bool(settings.mlflow_enabled and mlflow is not None)
        self.mlflow_tracking_uri = settings.mlflow_tracking_uri
        self.mlflow_experiment_name = settings.mlflow_experiment_name

        self.vectorstore = FaissVectorStore(
            persist_dir=self.persist_dir,
            embedding_model=self.embedding_model,
        )
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
        )

        if self.mlflow_enabled:
            try:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                mlflow.set_experiment(self.mlflow_experiment_name)
            except Exception:
                self.mlflow_enabled = False
                print("[WARN] MLflow setup failed. Tracking is disabled.")
        elif mlflow is None:
            print("[WARN] MLflow is not installed. Tracking is disabled.")

        self.llm = None
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.llm = ChatGroq(
                api_key=groq_api_key,
                model=self.llm_model,
                temperature=0.1,
                max_tokens=1024,
            )
            print(f"[INFO] Groq LLM initialized: {self.llm_model}")
        else:
            print("[WARN] GROQ_API_KEY not found. LLM answering is disabled.")

    def _mlflow_run(self, run_name: str):
        if not self.mlflow_enabled:
            return nullcontext()
        return mlflow.start_run(run_name=run_name)

    def _index_paths(self) -> tuple[str, str]:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        return faiss_path, meta_path

    def index_exists(self) -> bool:
        faiss_path, meta_path = self._index_paths()
        return os.path.exists(faiss_path) and os.path.exists(meta_path)

    def build_index(self, save: bool = True) -> None:
        """Load documents, split into chunks, and build a persisted FAISS index."""
        started = time.perf_counter()
        docs_by_type = load_all_documents_from_directory(self.data_dir)
        chunks = self.embedding_pipeline.split_documents(docs_by_type)
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [dict(getattr(chunk, "metadata", {})) for chunk in chunks]

        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        if save:
            self.vectorstore.save()

        if self.mlflow_enabled:
            total_documents = sum(len(items) for items in docs_by_type.values())
            elapsed = time.perf_counter() - started
            try:
                with self._mlflow_run("build-index"):
                    mlflow.set_tag("stage", "index_build")
                    mlflow.log_param("project_name", self.project_name)
                    mlflow.log_param("data_dir", self.data_dir)
                    mlflow.log_param("persist_dir", self.persist_dir)
                    mlflow.log_param("embedding_model", self.embedding_model)
                    mlflow.log_param("chunk_size", self.embedding_pipeline.chunk_size)
                    mlflow.log_param("chunk_overlap", self.embedding_pipeline.chunk_overlap)
                    mlflow.log_metric("documents_loaded", total_documents)
                    mlflow.log_metric("chunks_created", len(chunks))
                    mlflow.log_metric("elapsed_seconds", elapsed)
            except Exception:
                print("[WARN] MLflow logging failed during index build.")

    def load_or_build_index(self, force_rebuild: bool = False) -> None:
        """Load persisted index if available, otherwise build and save it."""
        if force_rebuild:
            self.build_index(save=True)
            return

        if self.index_exists():
            self.vectorstore.load()
        else:
            self.build_index(save=True)

    def retrieve(self, query: str, top_k: int = 5, log_to_mlflow: bool = True) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar chunks for the query."""
        started = time.perf_counter()
        results = self.vectorstore.query(query_text=query, top_k=top_k)

        if self.mlflow_enabled and log_to_mlflow:
            elapsed = time.perf_counter() - started
            scores = [float(item.get("score", 0.0)) for item in results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            try:
                with self._mlflow_run("retrieve"):
                    mlflow.set_tag("stage", "retrieve")
                    mlflow.log_param("project_name", self.project_name)
                    mlflow.log_param("top_k", top_k)
                    mlflow.log_metric("query_length", len(query))
                    mlflow.log_metric("results_returned", len(results))
                    mlflow.log_metric("avg_score", avg_score)
                    mlflow.log_metric("elapsed_seconds", elapsed)
            except Exception:
                print("[WARN] MLflow logging failed during retrieval.")

        return results

    def answer(self, query: str, top_k: int = 5) -> str:
        """Generate an answer from retrieved context using Groq, if configured."""
        started = time.perf_counter()
        results = self.retrieve(query=query, top_k=top_k, log_to_mlflow=False)
        context_chunks = [r.get("metadata", {}).get("text", "") for r in results]
        context = "\n\n".join([chunk for chunk in context_chunks if chunk])

        if not context:
            return "No relevant documents found."

        if self.llm is None:
            return (
                "GROQ_API_KEY is not configured. Retrieved context:\n\n"
                f"{context[:2000]}"
            )

        prompt = (
            "You are a helpful assistant. Answer the question using only the given context. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        response = self.llm.invoke(prompt)
        answer_text = response.content

        if self.mlflow_enabled:
            elapsed = time.perf_counter() - started
            try:
                with self._mlflow_run("answer"):
                    mlflow.set_tag("stage", "answer")
                    mlflow.log_param("project_name", self.project_name)
                    mlflow.log_param("llm_model", self.llm_model)
                    mlflow.log_param("top_k", top_k)
                    mlflow.log_metric("query_length", len(query))
                    mlflow.log_metric("results_returned", len(results))
                    mlflow.log_metric("context_chars", len(context))
                    mlflow.log_metric("answer_chars", len(answer_text))
                    mlflow.log_metric("elapsed_seconds", elapsed)
            except Exception:
                print("[WARN] MLflow logging failed during answer generation.")

        return answer_text

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Backward-compatible wrapper for older call sites."""
        return self.answer(query=query, top_k=top_k)

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    rag_search.load_or_build_index()

    query = "What is the information security policy?"
    retrieved = rag_search.retrieve(query=query, top_k=3)
    print("Retrieved results:")
    for i, item in enumerate(retrieved, start=1):
        source = item.get("metadata", {}).get("source", "unknown")
        score = item.get("score", 0.0)
        print(f"{i}. source={source}, score={score:.4f}")

    answer = rag_search.answer(query=query, top_k=3)
    print("\nAnswer:")
    print(answer)