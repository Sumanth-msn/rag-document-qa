"""
rag_chain.py
────────────
Builds the Conversational RAG chain using:
- Groq LLaMA3 as the LLM (free, fast)
- FAISS retriever for document search
- ConversationBufferWindowMemory for remembering past Q&A turns
- Re-ranked documents for improved answer quality

Interview explanation:
"I used LangChain's ConversationalRetrievalChain which has two steps:
first it condenses the new question with chat history into a standalone
question, then it retrieves relevant chunks and generates an answer.
The memory buffer stores the last 5 Q&A turns so follow-up questions
like 'explain that further' work correctly. I also added a separate
answer generation function that accepts pre-ranked documents from the
cross encoder re-ranker, bypassing the chain's internal retriever."
"""

from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


ANSWER_PROMPT = PromptTemplate.from_template("""
You are a precise document assistant. Answer using ONLY the context below.

Instructions:
- Be specific — use exact names, numbers, tools, examples from the document
- Write naturally in paragraphs or bullet points as appropriate
- Do NOT use bold headers or labels like "Direct Answer:" or "Supporting Details:"
- Do NOT add generic closing lines like "It is worth noting...", "The specific requirements may vary...", or "Having a strong foundation..."
- Do NOT repeat the same point in different words
- Stop your answer when the document information ends — no padding
- If answer not in context say: "I couldn't find this in the uploaded documents."

Context:
{context}

Question: {question}

Answer:
""")


def get_llm():
    """Initialize Groq LLaMA3."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1024,
    )


def build_rag_chain(retriever) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain with memory.

    Args:
        retriever: Hybrid FAISS+BM25 retriever

    Returns:
        ConversationalRetrievalChain ready to use
    """
    llm = get_llm()

    memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", return_messages=True, output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT},
        verbose=False,
    )

    return chain


def ask_question(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Ask a question using the RAG chain (no re-ranking).

    Args:
        chain: Built ConversationalRetrievalChain
        question: User's question string

    Returns:
        Dict with "answer" and "source_documents"
    """
    result = chain.invoke({"question": question})
    return {
        "answer": result.get("answer", ""),
        "source_documents": result.get("source_documents", []),
    }


def answer_with_reranked_docs(
    question: str, reranked_docs: List[Document], chat_history: List[dict]
) -> dict:
    """
    Generate answer using pre-ranked documents from cross encoder.

    This bypasses the chain's internal retriever and uses
    the re-ranked documents directly — giving us full control
    over which chunks the LLM sees.

    How it works:
    1. Format re-ranked docs into a context string
    2. Build chat history string for memory context
    3. Send context + question + history to LLM
    4. Return answer + source documents

    Args:
        question: User's question
        reranked_docs: Top documents from cross encoder re-ranker
        chat_history: Recent Q&A turns for memory context

    Returns:
        Dict with "answer" and "source_documents"
    """
    llm = get_llm()

    # Format re-ranked docs into context string
    # Include source info so LLM can reference it
    context = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source_file', 'unknown')} "
            f"Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
            for doc in reranked_docs
        ]
    )

    # Format recent chat history for memory context
    history_text = ""
    if chat_history:
        recent = chat_history[-3:]
        lines = []
        for turn in recent:
            lines.append(f"User: {turn.get('question', '')}")
            ans = turn.get("answer", "")
            lines.append(f"Assistant: {ans[:200]}...")
        history_text = "\n".join(lines)

    # Prompt that includes memory + re-ranked context
    prompt = PromptTemplate.from_template("""
You are a precise document assistant. Answer using ONLY the context below.

Instructions:
- Be specific — use exact names, numbers, tools, examples from the document
- Write naturally in paragraphs or bullet points as appropriate
- Do NOT use bold headers or labels like "Direct Answer:" or "Supporting Details:"
- Do NOT add generic closing lines like "It is worth noting...", "The specific requirements may vary...", or "Having a strong foundation..."
- Do NOT repeat the same point in different words
- Stop your answer when the document information ends — no padding
- Use chat history only to understand references like "that" or "it"
- If answer not in context say: "I couldn't find this in the uploaded documents."

Recent conversation:
{chat_history}

Context from documents (re-ranked by relevance):
{context}

Question: {question}

Answer:
""")

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "question": question,
            "context": context,
            "chat_history": history_text or "No previous conversation.",
        }
    )

    return {"answer": answer.strip(), "source_documents": reranked_docs}
