"""
reranker.py
───────────
Re-ranks retrieved chunks using a Cross Encoder model.

Two-stage retrieval pipeline:
  Stage 1 — Hybrid Retriever (FAISS + BM25)
            Fast but approximate — returns top-k candidates
  Stage 2 — Cross Encoder Re-ranker (this file)
            Slow but accurate — re-scores every candidate
            by reading question + chunk TOGETHER

Why Cross Encoder is more accurate than FAISS:
  FAISS: converts question to vector, converts chunk to vector SEPARATELY
         then measures distance between the two vectors
         → fast but loses context between question and chunk

  Cross Encoder: reads question AND chunk TOGETHER in one pass
                 → understands the relationship between them
                 → much more accurate relevance score

Interview explanation:
"I implemented a two-stage retrieval pipeline. First, the hybrid
FAISS+BM25 retriever quickly fetches the top 10 candidate chunks.
Then a Cross Encoder model re-reads each candidate together with
the question and assigns a precise relevance score. The top 4
re-ranked chunks are passed to the LLM. This is the same approach
used by Cohere Rerank and Azure AI Search in production systems.
The model I used is cross-encoder/ms-marco-MiniLM-L-6-v2 which is
free, runs locally on CPU, and is specifically trained for passage
re-ranking tasks."
"""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List, Tuple


# cross-encoder/ms-marco-MiniLM-L-6-v2:
# - Specifically trained for passage re-ranking
# - Small and fast — runs on CPU
# - ~80MB download, cached after first run
# - Industry standard for re-ranking tasks
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Singleton — load model once, reuse across requests
_reranker = None


def get_reranker() -> CrossEncoder:
    """
    Load the cross encoder model (singleton pattern).
    Downloads on first call, cached after that.

    Returns:
        CrossEncoder model instance
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank_documents(
    query: str, documents: List[Document], top_n: int = 4
) -> List[Tuple[Document, float]]:
    """
    Re-rank documents using Cross Encoder.

    How it works:
    1. For each document, create a (query, document_text) pair
    2. Feed ALL pairs to Cross Encoder in one batch
    3. Cross Encoder reads each pair together and scores relevance
    4. Sort by score descending
    5. Return top_n documents

    Args:
        query: User's question
        documents: List of Document chunks from hybrid retriever
        top_n: How many top documents to return after re-ranking

    Returns:
        List of (Document, score) tuples sorted by relevance
    """
    if not documents:
        return []

    reranker = get_reranker()

    # Create (query, chunk_text) pairs for cross encoder
    pairs = [[query, doc.page_content] for doc in documents]

    # Score all pairs in one batch
    # Returns a list of floats — higher = more relevant
    scores = reranker.predict(pairs)

    # Zip documents with their scores
    scored_docs = list(zip(documents, scores))

    # Sort by score descending — most relevant first
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return top_n
    return scored_docs[:top_n]


def get_reranked_docs_only(
    query: str, documents: List[Document], top_n: int = 4
) -> List[Document]:
    """
    Convenience function — returns just the documents without scores.
    Used when you only need the ranked docs, not the scores.

    Args:
        query: User's question
        documents: List of Document chunks
        top_n: How many to return

    Returns:
        List of Document objects sorted by relevance
    """
    scored = rerank_documents(query, documents, top_n)
    return [doc for doc, _ in scored]
