import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Any, Dict, List

try:
    from src.vectorstore import FaissVectorStore
    from src.data_loader import load_all_documents_from_directory
    from src.embedding import EmbeddingPipeline
except ModuleNotFoundError:
    from vectorstore import FaissVectorStore
    from data_loader import load_all_documents_from_directory
    from embedding import EmbeddingPipeline

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        data_dir: str = "./data",
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
        chunk_size: int = 500,
        chunk_overlap: int = 20,
    ):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.vectorstore = FaissVectorStore(
            persist_dir=self.persist_dir,
            embedding_model=self.embedding_model,
        )
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

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

    def _index_paths(self) -> tuple[str, str]:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        return faiss_path, meta_path

    def index_exists(self) -> bool:
        faiss_path, meta_path = self._index_paths()
        return os.path.exists(faiss_path) and os.path.exists(meta_path)

    def build_index(self, save: bool = True) -> None:
        """Load documents, split into chunks, and build a persisted FAISS index."""
        docs_by_type = load_all_documents_from_directory(self.data_dir)
        chunks = self.embedding_pipeline.split_documents(docs_by_type)
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [dict(getattr(chunk, "metadata", {})) for chunk in chunks]

        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        if save:
            self.vectorstore.save()

    def load_or_build_index(self, force_rebuild: bool = False) -> None:
        """Load persisted index if available, otherwise build and save it."""
        if force_rebuild:
            self.build_index(save=True)
            return

        if self.index_exists():
            self.vectorstore.load()
        else:
            self.build_index(save=True)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar chunks for the query."""
        return self.vectorstore.query(query_text=query, top_k=top_k)

    def answer(self, query: str, top_k: int = 5) -> str:
        """Generate an answer from retrieved context using Groq, if configured."""
        results = self.retrieve(query=query, top_k=top_k)
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
        return response.content

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Backward-compatible wrapper for older call sites."""
        return self.answer(query=query, top_k=top_k)

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch(data_dir="./data", persist_dir="faiss_store")
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