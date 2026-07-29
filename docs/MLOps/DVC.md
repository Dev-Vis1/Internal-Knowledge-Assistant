# DVC and Why It Matters in This Project

DVC, which stands for Data Version Control, helps manage datasets and machine-learning artifacts in a way that feels similar to Git. Git is very good at tracking source code, but it is not designed for large files such as document collections, model artifacts, or generated indexes. DVC fills that gap by letting the project track those assets cleanly without bloating the Git history.

In a project like this internal knowledge assistant, that matters because the app depends on a growing knowledge base. As more PDFs, text files, embeddings, and vector indexes are added over time, it becomes important to know which version of the data was used and how the application was built around it.

## How DVC fits this project

In this project, DVC has been initialized and the first raw knowledge-base folders are now tracked with it.

The basic idea is:

1. Git tracks the application code.
2. DVC tracks large data assets and generated artifacts.
3. The team can reproduce the same knowledge-base state across machines.
4. Future updates to the dataset can be managed more safely and more clearly.

This is especially useful for a RAG application because the system depends on more than just Python files. It also depends on the source documents that feed retrieval, and sometimes on derived outputs such as indexes or processed data.

## Files added for DVC

These files were added for the DVC setup and the first tracked data folders:

- `.dvc/`: stores DVC's internal project configuration.
- `.dvc/config`: the main DVC configuration file for the repository.
- `.dvc/.gitignore`: prevents DVC's internal cache and temporary files from being committed incorrectly.
- `.dvcignore`: tells DVC which files or folders it should ignore when scanning the project.
- `data/pdf_files.dvc`: the DVC pointer file for the PDF knowledge-base folder.
- `data/txt_files.dvc`: the DVC pointer file for the text knowledge-base folder.
- `data/.gitignore`: prevents Git from tracking the DVC-managed data folders directly.

## Why this helps for MLOps

For MLOps, DVC improves reproducibility and project organization. Instead of treating data as an afterthought, it becomes part of the workflow.

That helps the project:

- keep large data out of normal Git history
- reproduce the same document set across environments
- track changes to datasets more clearly
- prepare for remote storage later if the project grows

## What DVC tracks here right now

The raw knowledge-base folders below are now tracked through DVC:

- `data/pdf_files/`
- `data/txt_files/`

This means Git stores lightweight `.dvc` pointer files, while DVC manages the actual folder contents.

In practice, the raw source documents are usually the best first thing to track with DVC. Generated folders such as local vector indexes are often better treated as rebuildable artifacts unless the team has a specific reason to version them directly.

## What DVC would likely track next

In this project, DVC is also a good fit for assets such as:

- other large source-data folders inside `data/`
- large sample corpora added for testing
- generated retrieval artifacts if they become too large for Git




