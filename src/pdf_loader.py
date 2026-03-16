"""
pdf_loader.py
─────────────
Loads one or more PDF files and splits them into chunks.

Key feature: Each chunk carries metadata (source filename + page number)
so we always know WHERE each answer came from.

Interview explanation:
"I used PyPDFLoader to extract text page by page, then
RecursiveCharacterTextSplitter to break it into 500-character chunks
with 100-character overlap. Each chunk is tagged with its source filename
and page number so the UI can show exactly where the answer came from."
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tempfile
import os
from typing import List


def load_and_chunk_pdfs(uploaded_files: list) -> List[Document]:
    """
    Load multiple PDFs and split into overlapping chunks.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects

    Returns:
        List of Document chunks, each with source_file + page metadata
    """

    # Text splitter — splits on paragraphs first, then sentences, then words
    # chunk_size=500: each chunk is ~500 characters (roughly 1-2 paragraphs)
    # chunk_overlap=50: 50 chars shared between consecutive chunks
    #                   this prevents losing context at chunk boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", "!", "?", " "]
    )

    all_chunks = []

    for uploaded_file in uploaded_files:
        # PyPDFLoader needs a real file path, not a file object
        # So we save to a temp file first, then load it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()  # one Document per page
            chunks = text_splitter.split_documents(pages)  # split into smaller chunks

            # Tag every chunk with the original filename
            # This is how we track which PDF each answer came from
            for chunk in chunks:
                chunk.metadata["source_file"] = uploaded_file.name
                chunk.metadata["page"] = chunk.metadata.get("page", 0)

            all_chunks.extend(chunks)

        finally:
            os.unlink(tmp_path)  # clean up temp file — good practice

    return all_chunks


def get_doc_summary(chunks: List[Document]) -> dict:
    """
    Build a summary of how many chunks came from each file.
    Used in the sidebar to show what's loaded.

    Returns:
        {"files": {"file.pdf": 42, ...}, "total": 120}
    """
    files = {}
    for chunk in chunks:
        name = chunk.metadata.get("source_file", "unknown")
        files[name] = files.get(name, 0) + 1

    return {"files": files, "total": len(chunks)}
