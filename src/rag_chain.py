"""
rag_chain.py
────────────
Builds the Conversational RAG chain using:
- Groq LLaMA3 as the LLM (free, fast)
- FAISS retriever for document search
- ConversationBufferWindowMemory for remembering past Q&A turns

Interview explanation:
"I used LangChain's ConversationalRetrievalChain which has two steps:
first it condenses the new question with chat history into a standalone
question, then it retrieves relevant chunks and generates an answer.
The memory buffer stores the last 5 Q&A turns so follow-up questions
like 'explain that further' work correctly."
"""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


# Custom prompt — tells the LLM exactly how to behave
# This is prompt engineering, which WCT JD explicitly asks for
ANSWER_PROMPT = PromptTemplate.from_template("""
You are a helpful document assistant. Answer the user's question 
based only on the provided context from the uploaded documents.

Rules:
- Answer only from the context provided
- If the answer is not in the context, say: "I couldn't find this in the uploaded documents."
- Be concise and clear
- If quoting directly, mention the source

Context:
{context}

Question:
{question}

Answer:
""")


def build_rag_chain(retriever) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain with memory.

    How it works internally:
    1. User asks a question
    2. Chain condenses question + chat history → standalone question
    3. Standalone question is used to retrieve chunks from FAISS
    4. Retrieved chunks + original question → sent to LLM
    5. LLM generates answer
    6. Q&A pair saved to memory for next turn

    Args:
        retriever: FAISS retriever from vectorstore.py

    Returns:
        ConversationalRetrievalChain ready to use
    """

    # Groq LLaMA3 — free, very fast
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,  # low temperature = more factual, less creative
        max_tokens=1024,
    )

    # Sliding window memory — remembers last 5 Q&A turns
    # Why window? Unlimited memory would overflow the context window
    memory = ConversationBufferWindowMemory(
        k=5,  # remember last 5 turns
        memory_key="chat_history",
        return_messages=True,  # return as message list (not plain string)
        output_key="answer",  # store "answer" field in memory
    )

    # ConversationalRetrievalChain:
    # combines retrieval + LLM + memory into one chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # return which chunks were used
        combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT},
        verbose=False,
    )

    return chain


def ask_question(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Ask a question using the RAG chain.

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
