"""
vectorstore.py
──────────────
Converts document chunks into vector embeddings using HuggingFace,
stores them in FAISS, and retrieves the most relevant chunks
along with their similarity (confidence) scores.

NOW WITH HYBRID SEARCH (FAISS MMR + BM25):
- FAISS MMR = diverse semantic search (relevance + diversity)
- BM25      = keyword search (exact word matching)
- EnsembleRetriever = combines both for better accuracy

Why MMR over pure similarity:
- Similarity returns top-10 most similar chunks
  → often 4-5 chunks say the SAME thing from nearby pages
  → redundant context → LLM repeats information

- MMR fetches 20 candidates, picks 10 that are BOTH relevant AND different
  → diverse context → richer, non-repetitive answers
  → better RAGAS scores (context precision + recall)

Interview explanation:
"I replaced pure similarity search with MMR — Maximal Marginal Relevance.
MMR fetches 20 candidates then selects 10 that maximize both relevance
to the query AND diversity from each other. This prevents the LLM from
receiving redundant context which was causing repetitive answers.
Combined with BM25 keyword search, this gives the re-ranker a
high-quality diverse candidate pool to work with."
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

    Steps:
    1. Initialize HuggingFace embedding model
    2. Convert every chunk to a vector
    3. Store all vectors in FAISS index

    Args:
        chunks: List of Document chunks from pdf_loader

    Returns:
        FAISS vector store
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_retriever(vectorstore: FAISS, chunks: List[Document], k: int = 10):
    """
    Build a Hybrid Retriever combining FAISS MMR + BM25.

    FAISS uses MMR (Maximal Marginal Relevance) for diverse retrieval.
    fetch_k=20: fetch 20 candidates first
    k=10: from those 20, return 10 maximally diverse ones
    lambda_mult=0.7: 70% relevance weight, 30% diversity weight

    Args:
        vectorstore: FAISS index
        chunks: Original document chunks (needed for BM25)
        k: Number of chunks each retriever returns (default 10)

    Returns:
        EnsembleRetriever (FAISS MMR + BM25 combined)
    """
    # FAISS retriever — MMR for diverse semantic search
    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 2,  # fetch double, pick diverse k
            "lambda_mult": 0.7,  # 70% relevance, 30% diversity
        },
    )

    # BM25 retriever — keyword search
    # BM25 naturally returns diverse results via term frequency ranking
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # Combine both — weights must sum to 1.0
    # 60% MMR semantic + 40% BM25 keyword
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=[0.6, 0.4]
    )

    return hybrid_retriever


def retrieve_with_scores(
    vectorstore: FAISS, query: str, k: int = 10
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k chunks WITH similarity scores.
    Used for displaying confidence score in UI.

    Note: Uses standard similarity search for scoring since
    MMR doesn't return scores directly.

    Args:
        vectorstore: FAISS index
        query: User's question
        k: Number of chunks to retrieve

    Returns:
        List of (Document, similarity_score) tuples
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def score_to_percentage(score: float) -> int:
    """
    Convert FAISS cosine similarity score to 0-100 percentage.

    Args:
        score: Raw cosine similarity score from FAISS

    Returns:
        Integer percentage 0-100
    """
    percentage = int(score * 100)
    return max(0, min(100, percentage))
