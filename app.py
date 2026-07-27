from src.data_loader import load_all_documents_from_directory
from src.embedding import EmbeddingPipeline

##example usage
if __name__ == "__main__":
    docs_by_type = load_all_documents_from_directory("./data")
    docs = [doc for docs_list in docs_by_type.values() for doc in docs_list]
    embedding_pipeline = EmbeddingPipeline()
    chunks = embedding_pipeline.split_documents(docs)
    vectors = embedding_pipeline.generate_embeddings([chunk.page_content for chunk in chunks])

    print(docs_by_type)
    print(chunks)
    print(vectors)