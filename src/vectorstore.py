"""
vectorstore.py
──────────────
Converts document chunks into vector embeddings using HuggingFace,
stores them in FAISS, and retrieves the most relevant chunks
along with their similarity (confidence) scores.

NOW WITH HYBRID SEARCH:
- FAISS = semantic search (finds chunks by meaning)
- BM25  = keyword search (finds chunks by exact words)
- EnsembleRetriever = combines both for better accuracy

Interview explanation:
"I used the sentence-transformers/all-MiniLM-L6-v2 model from HuggingFace
to convert each text chunk into a 384-dimensional embedding vector.
FAISS stores all these vectors and finds the nearest ones using cosine
similarity. I also added BM25 keyword search and combined both using
LangChain's EnsembleRetriever — this hybrid approach significantly
improves retrieval accuracy, especially when questions use exact
keywords from the document."
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from typing import List, Tuple


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Create a FAISS vector store from document chunks.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_retriever(vectorstore: FAISS, chunks: List[Document], k: int = 5):
    """
    Build a Hybrid Retriever combining FAISS + BM25.

    Why Hybrid?
    - FAISS alone: good at meaning but misses exact keyword matches
    - BM25 alone: good at keywords but misses semantic meaning
    - Together: best of both worlds — catches more relevant chunks

    Weights: 60% FAISS semantic + 40% BM25 keyword
    """
    # FAISS retriever — semantic search
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    # BM25 retriever — keyword search
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # Combine both — weights must sum to 1.0
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=[0.6, 0.4]
    )

    return hybrid_retriever


def retrieve_with_scores(
    vectorstore: FAISS, query: str, k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k chunks with similarity scores.
    Used for displaying confidence score in UI.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def score_to_percentage(score: float) -> int:
    """
    Convert FAISS cosine similarity score to 0-100 percentage.
    """
    percentage = int(score * 100)
    return max(0, min(100, percentage))
