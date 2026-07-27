from src.data_loader import load_all_documents_from_directory
from src.embedding import EmbeddingPipeline
from sentence_transformers import SentenceTransformer
import os
import pickle
import uuid
import chromadb
from typing import List, Dict, Any
import numpy as np

## Vector Store

class FaissVectorStore:

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        #self.collection_name = collection_name
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []  
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size  # or any default value you prefer
        self.chunk_overlap = chunk_overlap  # or any default value you prefer
        print(f"[INFO] Loaded embedding model: {embedding_model}")

