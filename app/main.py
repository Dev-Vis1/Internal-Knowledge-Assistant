from pathlib import Path
import streamlit as st

from src.search import RAGSearch


DATA_DIR = Path("./data")
LOGO_PATH = Path("./assets/northstar_logo.svg")
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf_files",
    ".txt": "txt_files",
    ".csv": "csv_files",
    ".docx": "docx_files",
    ".html": "html_files",
    ".pptx": "pptx_files",
    ".sql": "sql_files",
    ".json": "json_files",
    ".yaml": "yaml_files",
    ".xml": "xml_files",
}


def ensure_data_folders() -> None:
    for folder in SUPPORTED_EXTENSIONS.values():
        (DATA_DIR / folder).mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_rag_service() -> RAGSearch:
    return RAGSearch(data_dir="./data", persist_dir="faiss_store")


def save_uploaded_files(files) -> tuple[int, int, list[str]]:
    saved_count = 0
    skipped_count = 0
    saved_paths: list[str] = []

    for uploaded_file in files:
        ext = Path(uploaded_file.name).suffix.lower()
        target_folder = SUPPORTED_EXTENSIONS.get(ext)
        if not target_folder:
            skipped_count += 1
            continue

        target_path = DATA_DIR / target_folder / uploaded_file.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_count += 1
        saved_paths.append(str(target_path))

    return saved_count, skipped_count, saved_paths


def list_knowledge_files() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ext, folder in SUPPORTED_EXTENSIONS.items():
        folder_path = DATA_DIR / folder
        if not folder_path.is_dir():
            continue
        for path in sorted(folder_path.glob(f"*{ext}")):
            rows.append(
                {
                    "file_name": path.name,
                    "file_type": ext.replace(".", "").upper(),
                    "folder": folder,
                    "size_kb": f"{path.stat().st_size / 1024:.1f}",
                    "updated": path.stat().st_mtime,
                    "path": str(path),
                }
            )
    rows.sort(key=lambda item: item["updated"], reverse=True)
    return rows


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top right, #E6F8FF 0%, #F8FBFF 40%, #FFFFFF 100%);
        }
        .hero {
            padding: 1.1rem 1.2rem;
            border-radius: 14px;
            border: 1px solid #d9e8f7;
            background: linear-gradient(120deg, #f2f8ff 0%, #ffffff 70%);
            margin-bottom: 0.8rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 1.85rem;
            color: #0f2a43;
        }
        .hero p {
            margin: 0.35rem 0 0;
            color: #35516b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_admin_upload_section() -> None:
    st.subheader("Knowledge admin portal")
    st.caption("Upload company documents into the correct knowledge folders.")

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "txt", "csv", "docx", "html", "pptx", "sql", "json", "yaml", "xml"],
        accept_multiple_files=True,
    )

    if st.button("Save uploaded files", width="stretch"):
        if not uploaded_files:
            st.warning("No files selected.")
        else:
            saved_count, skipped_count, saved_paths = save_uploaded_files(uploaded_files)
            st.success(f"Saved {saved_count} file(s).")
            if skipped_count:
                st.warning(f"Skipped {skipped_count} unsupported file(s).")
            if saved_paths:
                with st.expander("Saved file locations", icon=":material/folder:"):
                    for path in saved_paths:
                        st.write(path)


def render_admin_knowledge_base_section() -> None:
    st.subheader("Knowledge base viewer")
    st.caption("Browse all ingested files available to the RAG pipeline.")

    rows = list_knowledge_files()
    if not rows:
        st.info("No documents available yet.", icon=":material/info:")
        return

    type_options = ["All"] + sorted({row["file_type"] for row in rows})
    selected_type = st.selectbox("Filter by file type", options=type_options)
    filtered_rows = rows if selected_type == "All" else [row for row in rows if row["file_type"] == selected_type]

    st.caption(f"Showing {len(filtered_rows)} of {len(rows)} files")
    st.dataframe(
        [
            {
                "File": row["file_name"],
                "Type": row["file_type"],
                "Folder": row["folder"],
                "Size (KB)": row["size_kb"],
            }
            for row in filtered_rows
        ],
        width="stretch",
        hide_index=True,
    )

    selected_file = st.selectbox(
        "Preview file content",
        options=[row["path"] for row in filtered_rows],
        format_func=lambda item: Path(item).name,
    )

    preview_path = Path(selected_file)
    suffix = preview_path.suffix.lower()
    if suffix in {".txt", ".csv", ".html", ".sql", ".json", ".yaml", ".xml"}:
        preview_text = preview_path.read_text(encoding="utf-8", errors="ignore")
        st.text_area("Preview", preview_text[:4000], height=260)
    else:
        st.info("Preview is available for text-based formats. PDF/DOCX/PPTX are indexed but not previewed here.")


def render_admin_index_section() -> None:
    st.subheader("Index management")
    st.caption("Rebuild the vector index after major document updates.")

    if st.button("Rebuild knowledge index", width="stretch"):
        with st.spinner("Rebuilding vector index from all documents..."):
            rag = get_rag_service()
            rag.load_or_build_index(force_rebuild=True)
        st.success("Knowledge index rebuilt successfully.")


def render_admin_portal() -> None:
    admin_view = st.sidebar.segmented_control(
        "Admin view",
        options=["Upload", "Knowledge base", "Index"],
        default="Upload",
    )

    if admin_view == "Upload":
        render_admin_upload_section()
    elif admin_view == "Knowledge base":
        render_admin_knowledge_base_section()
    else:
        render_admin_index_section()


def render_user_portal() -> None:
    st.subheader("User query portal")
    st.caption("Ask questions and retrieve answers from your company knowledge base.")

    query = st.text_area(
        "Ask a question",
        placeholder="Example: What is our incident response escalation process?",
        height=110,
    )
    top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=10, value=4)

    if st.button("Get answer", width="stretch"):
        if not query.strip():
            st.warning("Please enter a question first.")
            return

        rag = get_rag_service()
        with st.spinner("Searching and generating response..."):
            rag.load_or_build_index(force_rebuild=False)
            answer = rag.answer(query=query, top_k=top_k)
            results = rag.retrieve(query=query, top_k=top_k)

        st.markdown("### Response")
        st.write(answer)

        with st.expander("Retrieved sources"):
            for idx, result in enumerate(results, start=1):
                metadata = result.get("metadata", {})
                st.write(
                    f"{idx}. {metadata.get('source', 'unknown')} | "
                    f"score={result.get('score', 0.0):.4f}"
                )


def main() -> None:
    st.set_page_config(page_title="Internal Knowledge Assistant", page_icon=":material/hub:", layout="wide")
    ensure_data_folders()
    apply_custom_style()

    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH), icon_image=str(LOGO_PATH))

    st.markdown(
        """
        <div class="hero">
            <h1>Internal knowledge assistant</h1>
            <p>For Northstar Meridian Group. Developed by Matthew Ayodele</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    role = st.sidebar.segmented_control(
        "Portal",
        options=["User", "Knowledge Admin"],
        default="User",
    )

    st.sidebar.caption("Northstar Meridian Group")
    st.sidebar.badge("Knowledge ready", icon=":material/database:", color="blue")

    if role == "Knowledge Admin":
        render_admin_portal()
    else:
        render_user_portal()


if __name__ == "__main__":
    main()
