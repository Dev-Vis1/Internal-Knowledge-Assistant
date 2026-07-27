from pathlib import Path
import os
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader as PPTXLoader,
)

File_type_and_loaders = {
    "txt": TextLoader,
    "pdf": PyMuPDFLoader,
    "csv": CSVLoader,
    "docx": Docx2txtLoader,
    "html": TextLoader,
    "pptx": PPTXLoader,
    "sql": TextLoader,
    "json": TextLoader,
    "yaml": TextLoader,
    "xml": TextLoader,

}

def load_all_documents_from_directory(directory_path: str) -> Dict[str, List[Any]]:

    #loaders for each file type
    loaded_documents = {}
    base_directory = Path(directory_path).resolve()

    for file_type, loader_cls in File_type_and_loaders.items():

        #if folder type does not exist, skip to next file type
        file_type_directory = base_directory / f"{file_type}_files"
        print(file_type_directory)

        if not file_type_directory.is_dir():
            print(f"Directory for {file_type} files does not exist. Skipping.")
            continue

        files = list(file_type_directory.glob(f"*.{file_type}"))
        print(f"Found {len(files)} {file_type.upper()} files in {file_type_directory}.")

        for file_path in files:

                filename = file_path.name

                print(f"\n Processing {filename}")

                try:
                    loader = loader_cls(str(file_path))
                    documents = loader.load()

                    for doc in documents:
                        doc.metadata["source"] = filename
                        doc.metadata["total_pages"] = len(documents)
                        doc.metadata["file_size"] = file_path.stat().st_size
                        doc.metadata["file_type"] = file_type

                    loaded_documents.setdefault(file_type, []).extend(documents)
                    print(f"Successfully loaded {len(documents)} page(s) from {filename} using {loader_cls.__name__}.")

                except Exception as e:
                    print(f"Failed to load {filename} with {loader_cls.__name__}: {e}. ")

    print(f"\n Total documents loaded: {sum(len(docs) for docs in loaded_documents.values())}")
        
    return loaded_documents
            
if __name__ == "__main__":
    all_documents = load_all_documents_from_directory("./data")