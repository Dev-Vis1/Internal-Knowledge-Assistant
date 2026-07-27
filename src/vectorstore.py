try:
	from src.vectore_store import FaissVectorStore
except ModuleNotFoundError:
	from vectore_store import FaissVectorStore

__all__ = ["FaissVectorStore"]
