"""
vectorstore.py
──────────────
Converts document chunks into vector embeddings using HuggingFace,
stores them in FAISS, and retrieves the most relevant chunks
along with their similarity (confidence) scores.

Interview explanation:
"I used the sentence-transformers/all-MiniLM-L6-v2 model from HuggingFace
to convert each text chunk into a 384-dimensional embedding vector.
FAISS stores all these vectors and finds the nearest ones when given
a query — using cosine similarity. The similarity score is a float
between 0 and 1 which I display as a confidence percentage."
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Tuple


# all-MiniLM-L6-v2:
# - 90MB download (cached after first run)
# - Runs on CPU (no GPU needed)
# - 384-dimensional embeddings
# - Great balance of speed and accuracy for semantic search
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
        # normalize_embeddings=True makes cosine similarity
        # equivalent to dot product — needed for correct scoring
    )

    # FAISS.from_documents:
    # - Embeds every chunk using the embedding model
    # - Builds an in-memory FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_with_scores(
    vectorstore: FAISS, query: str, k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k most relevant chunks WITH their similarity scores.

    This is the key function for the confidence score feature.
    FAISS returns (document, score) pairs where score is cosine similarity.

    Args:
        vectorstore: FAISS index
        query: User's question
        k: Number of chunks to retrieve (default 5)

    Returns:
        List of (Document, similarity_score) tuples, sorted by relevance
    """
    # similarity_search_with_score returns List[Tuple[Document, float]]
    # Score is cosine similarity: 1.0 = perfect match, 0.0 = no match
    results = vectorstore.similarity_search_with_score(query, k=k)

    # Sort by score descending (highest confidence first)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def score_to_percentage(score: float) -> int:
    """
    Convert a raw FAISS cosine similarity score to a 0-100 percentage.

    Args:
        score: Raw cosine similarity score from FAISS

    Returns:
        Integer percentage 0-100
    """
    percentage = int(score * 100)
    return max(0, min(100, percentage))  # clamp between 0 and 100


def get_retriever(vectorstore: FAISS, k: int = 5):
    """
    Get a standard LangChain retriever from the vector store.
    Used by the ConversationalRetrievalChain.

    Args:
        vectorstore: FAISS index
        k: Number of chunks to retrieve

    Returns:
        LangChain BaseRetriever
    """
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
