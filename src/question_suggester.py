"""
question_suggester.py
─────────────────────
After a PDF is processed, automatically generates 4 relevant
questions the user could ask based on document content.

How it works:
1. Take the first ~2000 characters from the document chunks
2. Send to Groq LLaMA3 with a prompt asking for questions
3. Parse and return the 4 questions

Interview explanation:
"I used LLaMA3 to auto-generate relevant questions from the document
content after upload. This reduces friction for users — they don't
have to think about what to ask. The LLM reads a sample of the document
and returns 4 contextually relevant questions in JSON format."
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List
import os
import json
from dotenv import load_dotenv

load_dotenv()


SUGGEST_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Based on the document content below,
generate exactly 4 relevant and specific questions a user might ask.

Rules:
- Questions must be answerable from the document
- Make them specific, not generic
- Vary the question types (what, how, why, who)
- Return ONLY a JSON array of 4 strings, nothing else
- No preamble, no explanation, just the JSON array

Document content:
{context}

Return format example:
["Question 1?", "Question 2?", "Question 3?", "Question 4?"]
""")


def generate_suggested_questions(chunks: List[Document]) -> List[str]:
    """
    Generate 4 relevant questions from document chunks.

    Args:
        chunks: List of document chunks from pdf_loader

    Returns:
        List of 4 suggested question strings
        Falls back to generic questions if LLM call fails
    """

    # Use first 10 chunks to get a representative sample
    # ~2000 chars is enough for LLM to understand document topic
    sample_chunks = chunks[:10]
    context = "\n\n".join([c.page_content for c in sample_chunks])
    context = context[:2000]  # limit to 2000 chars to save tokens

    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,  # slight creativity for varied questions
            max_tokens=300,  # questions are short, 300 tokens is enough
        )

        chain = SUGGEST_PROMPT | llm | StrOutputParser()
        response = chain.invoke({"context": context})

        # Parse JSON response
        # Clean up in case LLM added backticks or extra text
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        questions = json.loads(response)

        # Validate — must be list of strings
        if isinstance(questions, list) and len(questions) >= 3:
            return questions[:4]  # return max 4
        else:
            return _fallback_questions()

    except Exception:
        # If anything fails, return generic fallback questions
        return _fallback_questions()


def _fallback_questions() -> List[str]:
    """
    Generic fallback questions if LLM call fails.
    These work for almost any document.
    """
    return [
        "What is the main topic of this document?",
        "What are the key points discussed?",
        "What are the conclusions or recommendations?",
        "Who is the target audience for this document?",
    ]
