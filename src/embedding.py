from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
from data_loader import load_all_documents_from_directory

class EmbeddingPipeline:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, chunk_overlap: int = 20):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"Initialized with Embedding model: {self.model}")

    def split_documents(self, documents: Any) -> List[Any]:
        """
        Splits documents into smaller chunks based on the specified chunk size and overlap.

        Args:
            documents (List[Any]): List of document objects to be split.
        """

        if isinstance(documents, dict):
            documents = [doc for docs_list in documents.values() for doc in docs_list]

        if not isinstance(documents, list):
            raise TypeError("documents must be a list of Document objects or a dict of lists.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:

        if not self.model:
            raise ValueError("Model is not loaded. Call _load_model() first.")

        print(f"Generating embeddings for {len(texts)} texts using model: {self.model_name}.")

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            print(f"Successfully generated embeddings. Shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"Failed to generate embeddings. Error: {e}")
            raise

if __name__ == "__main__":
    all_documents = load_all_documents_from_directory("./data")
    pipeline = EmbeddingPipeline()
    chunks = pipeline.split_documents(all_documents)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = pipeline.generate_embeddings(texts)