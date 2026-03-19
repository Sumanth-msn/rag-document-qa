from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import time
import re
from src.pdf_loader import load_and_chunk_pdfs, get_doc_summary
from src.vectorstore import (
    create_vectorstore,
    retrieve_with_scores,
    score_to_percentage,
    get_retriever,
)
from src.rag_chain import build_rag_chain, ask_question

st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp {
    background-color: #0a0a0f !important;
    color: #e8e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #0f0f18 !important;
    border-right: 1px solid #1e1e2e !important;
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
[data-testid="stHeader"] { background: #0a0a0f !important; border-bottom: 1px solid #1e1e2e; }

.app-header {
    background: linear-gradient(135deg, #1a0533 0%, #0d1f4e 50%, #001a2e 100%);
    border: 1px solid #2a1a4e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.app-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-header p { font-size: 0.82rem; color: #7c7c9a !important; margin: 0; font-family: 'JetBrains Mono', monospace; }
.pill {
    display: inline-block;
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.25);
    color: #a78bfa !important;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px 3px 0 0;
}
.stTextInput input {
    background: #0f0f18 !important;
    border: 1.5px solid #2a2a3e !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput input:focus { border-color: #7c3aed !important; box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important; }
.stTextInput input::placeholder { color: #4a4a6a !important; }
.stTextInput label { color: #e8e8f0 !important; }
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.2rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button:disabled { background: #1e1e2e !important; color: #4a4a6a !important; }
[data-testid="stFileUploader"] { background: #0f0f18 !important; border: 1.5px dashed #2a2a3e !important; border-radius: 12px !important; }
[data-testid="stFileUploader"] * { color: #e8e8f0 !important; }
.user-bubble {
    background: linear-gradient(135deg, #2d1b69, #1e3a8a);
    border: 1px solid #3730a3;
    color: #e8e8f0 !important;
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0 0.6rem 4rem;
    font-size: 0.93rem;
    line-height: 1.6;
}
.assistant-bubble {
    background: #0f0f18;
    border: 1px solid #1e1e2e;
    color: #e8e8f0 !important;
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 4rem 0.6rem 0;
    font-size: 0.93rem;
    line-height: 1.8;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.label-you { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #7c3aed !important; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; text-align: right; }
.label-ai { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #34d399 !important; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
.conf-badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 12px; border-radius: 20px; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; margin-bottom: 0.5rem; }
.conf-high   { background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3); color: #34d399 !important; }
.conf-medium { background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3); color: #fbbf24 !important; }
.conf-low    { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); color: #f87171 !important; }
.source-card { background: #0a0a0f; border: 1px solid #1e1e2e; border-left: 3px solid #7c3aed; border-radius: 0 10px 10px 0; padding: 0.8rem 1rem; margin: 0.5rem 0; }
.source-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #7c3aed !important; font-weight: 600; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.5px; }
.source-text { color: #9898b8 !important; font-size: 0.82rem; line-height: 1.6; }
.highlight { background: rgba(251,191,36,0.2); color: #fbbf24 !important; padding: 0 3px; border-radius: 3px; }
.chip { display: inline-block; background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.25); color: #a78bfa !important; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; margin: 2px; }
[data-testid="stMetric"] { background: #0f0f18 !important; border: 1px solid #1e1e2e !important; border-radius: 10px !important; padding: 0.8rem !important; }
[data-testid="stMetricLabel"] { color: #7c7c9a !important; }
[data-testid="stMetricValue"] { color: #a78bfa !important; }
[data-testid="stExpander"] { background: #0f0f18 !important; border: 1px solid #1e1e2e !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #9898b8 !important; font-size: 0.85rem !important; }
[data-testid="stInfo"] { background: rgba(96,165,250,0.07) !important; border: 1px solid rgba(96,165,250,0.2) !important; border-radius: 10px !important; }
[data-testid="stInfo"] * { color: #93c5fd !important; }
hr { border-color: #1e1e2e !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }
.example-card { background: #0f0f18; border: 1px solid #1e1e2e; border-radius: 10px; padding: 0.8rem 1rem; margin: 0.3rem 0; font-size: 0.85rem; color: #9898b8 !important; }
.footer-text { font-size: 0.72rem; color: #3a3a5a !important; font-family: 'JetBrains Mono', monospace; line-height: 1.8; }
p, span, div, li { color: #e8e8f0; }
h1, h2, h3, h4 { color: #e8e8f0 !important; }
.stMarkdown p { color: #e8e8f0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────

for key, default in {
    "vectorstore": None,
    "chunks": None,  # ← store chunks for BM25
    "rag_chain": None,
    "chat_history": [],
    "doc_summary": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📁 Documents")
    st.markdown(
        "<p style='color:#7c7c9a; font-size:0.82rem;'>Upload one or more PDFs to query.</p>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚙️ Process PDFs", use_container_width=True):
            with st.spinner("📄 Reading PDFs..."):
                chunks = load_and_chunk_pdfs(uploaded_files)
                st.session_state.chunks = chunks  # ← save chunks
                st.session_state.doc_summary = get_doc_summary(chunks)

            with st.spinner("🧠 Building embeddings..."):
                vectorstore = create_vectorstore(chunks)
                st.session_state.vectorstore = vectorstore

            with st.spinner("🔗 Building Hybrid RAG chain (FAISS + BM25)..."):
                retriever = get_retriever(vectorstore, chunks, k=5)  # ← pass chunks
                st.session_state.rag_chain = build_rag_chain(retriever)
                st.session_state.chat_history = []

            st.success("✅ Ready!")

    if st.session_state.doc_summary:
        st.divider()
        st.markdown("### 📊 Loaded")
        summary = st.session_state.doc_summary
        st.metric("Total Chunks", summary["total"])
        for fname, count in summary["files"].items():
            st.markdown(
                f"<span class='chip'>📄 {fname} · {count} chunks</span>",
                unsafe_allow_html=True,
            )

    st.divider()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.vectorstore and st.session_state.chunks:
                retriever = get_retriever(
                    st.session_state.vectorstore,
                    st.session_state.chunks,  # ← pass chunks
                    k=5,
                )
                st.session_state.rag_chain = build_rag_chain(retriever)
            st.rerun()

    st.divider()
    st.markdown(
        "<div class='footer-text'>LangChain · HuggingFace<br>FAISS + BM25 · Groq LLaMA3<br>Streamlit</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────

st.markdown(
    """
<div class='app-header'>
    <h1>📚 DocMind — RAG Document Q&A</h1>
    <p>Ask anything about your uploaded documents. Powered by LLaMA3 + Hybrid Search.</p>
    <div style='margin-top:0.8rem;'>
        <span class='pill'>Multi-PDF</span>
        <span class='pill'>Conversational Memory</span>
        <span class='pill'>Source Highlighting</span>
        <span class='pill'>Confidence Scores</span>
        <span class='pill'>Hybrid Search</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

docs_ready = st.session_state.rag_chain is not None

col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "question",
        placeholder="Ask a question about your uploaded documents...",
        label_visibility="collapsed",
        disabled=not docs_ready,
    )
with col2:
    ask_btn = st.button("Ask →", disabled=not docs_ready, use_container_width=True)

if not docs_ready:
    st.info("👈 Upload PDFs in the sidebar and click **Process PDFs** to get started.")

# ─────────────────────────────────────────
# Handle Question
# ─────────────────────────────────────────

if ask_btn and question.strip() and docs_ready:
    with st.spinner("Searching documents and generating answer..."):
        start = time.time()
        result = ask_question(st.session_state.rag_chain, question)
        scored_docs = retrieve_with_scores(st.session_state.vectorstore, question, k=5)
        top_score = score_to_percentage(scored_docs[0][1]) if scored_docs else 0
        elapsed = round(time.time() - start, 2)

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": result["answer"],
            "confidence": top_score,
            "source_documents": result["source_documents"],
            "elapsed": elapsed,
        }
    )

# ─────────────────────────────────────────
# Display Chat History
# ─────────────────────────────────────────

if st.session_state.chat_history:
    st.divider()
    for turn in reversed(st.session_state.chat_history):
        conf = turn["confidence"]
        if conf >= 70:
            conf_class, conf_label = "conf-high", f"✓ {conf}% match"
        elif conf >= 45:
            conf_class, conf_label = "conf-medium", f"~ {conf}% match"
        else:
            conf_class, conf_label = "conf-low", f"⚠ {conf}% match"

        st.markdown("<div class='label-you'>You</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='user-bubble'>{turn['question']}</div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<span class='conf-badge {conf_class}'>{conf_label}</span>"
            f"<span style='color:#3a3a5a; font-size:0.7rem; font-family:monospace;'> · {turn['elapsed']}s</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='label-ai'>DocMind</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='assistant-bubble'>{turn['answer']}</div>",
            unsafe_allow_html=True,
        )

        if turn["source_documents"]:
            with st.expander(
                f"📄 View sources ({len(turn['source_documents'])} chunks used)"
            ):
                for doc in turn["source_documents"]:
                    fname = doc.metadata.get("source_file", "unknown")
                    page = doc.metadata.get("page", 0)
                    text = doc.page_content.strip()
                    question_words = [
                        w.lower() for w in turn["question"].split() if len(w) > 3
                    ]
                    highlighted_text = text
                    for word in question_words:
                        highlighted_text = re.sub(
                            f"({re.escape(word)})",
                            r"<span class='highlight'>\1</span>",
                            highlighted_text,
                            flags=re.IGNORECASE,
                        )
                    st.markdown(
                        f"<div class='source-card'>"
                        f"<div class='source-label'>📄 {fname} · Page {page + 1}</div>"
                        f"<div class='source-text'>{highlighted_text[:400]}{'...' if len(text) > 400 else ''}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

else:
    if docs_ready:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💡 Try asking...")
        examples = [
            "What is the main topic of this document?",
            "Summarize the key points",
            "What are the conclusions or recommendations?",
            "Who are the authors or what organization published this?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                st.markdown(
                    f"<div class='example-card'>💬 {ex}</div>", unsafe_allow_html=True
                )
