import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG · Gemini",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --ink:     #0f1117;
    --paper:   #f5f0e8;
    --accent:  #1a5cff;
    --muted:   #6b7280;
    --border:  #d6cfc2;
    --card:    #ffffff;
    --success: #059669;
    --rain:    #3b82f6;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--paper);
    color: var(--ink);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--ink) !important;
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Serif Display', serif;
    color: white !important;
}

/* Main heading */
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin-bottom: 0;
}
.main-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.answer-card {
    background: var(--ink);
    color: white;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    line-height: 1.7;
}
.source-chip {
    display: inline-block;
    background: #f0f4ff;
    color: var(--accent);
    border: 1px solid #c7d7ff;
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    margin: 0.2rem 0.2rem 0.2rem 0;
}
.stat-box {
    text-align: center;
    padding: 0.75rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: white;
}
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: var(--accent);
}
.stat-label {
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Input */
.stTextInput > div > div > input {
    border-radius: 8px !important;
    border-color: var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(26,92,255,0.12) !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.15s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    background: white !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Chat history */
.chat-user {
    background: #f0f4ff;
    border-left: 3px solid var(--accent);
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.5rem;
    font-weight: 500;
}
.chat-bot {
    background: var(--ink);
    color: white;
    border-left: 3px solid var(--rain);
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
    font-size: 0.95rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: build RAG chain ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_chain(pdf_bytes: bytes, api_key: str):
    import tempfile
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    os.environ["GOOGLE_API_KEY"] = api_key

    # Write PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}, search_type="similarity")

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3, max_tokens=600)

    system_prompt = (
        "You are an expert assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question accurately and concisely. "
        "If you don't know the answer from the context, say so clearly. "
        "Keep your answer to 3-5 sentences.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    os.unlink(tmp_path)
    return rag_chain, len(docs), len(data)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Your Google Gemini API key"
    )

    st.markdown("---")
    st.markdown("### 📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.markdown("`gemini-2.5-flash`")
    st.markdown("### 🔢 Embeddings")
    st.markdown("`gemini-embedding-001`")
    st.markdown("### 🗄️ Vector Store")
    st.markdown("`Chroma (in-memory)`")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.rag_chain = None
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;opacity:0.5;'>Built with LangChain · Gemini · Streamlit</div>",
        unsafe_allow_html=True
    )


# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">PDF RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="main-sub">Powered by Google Gemini · Retrieval-Augmented Generation</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Init session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

# ── Index PDF ──────────────────────────────────────────────────────────────────
if uploaded_file and api_key:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("indexed_file") != file_key:
        with st.spinner("📖 Reading & indexing your PDF…"):
            try:
                chain, n_chunks, n_pages = build_rag_chain(uploaded_file.read(), api_key)
                st.session_state.rag_chain = chain
                st.session_state.doc_stats = {"pages": n_pages, "chunks": n_chunks, "name": uploaded_file.name}
                st.session_state.indexed_file = file_key
                st.session_state.chat_history = []
                st.success("✅ PDF indexed and ready!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

elif not api_key:
    st.info("👈 Enter your Google API key in the sidebar to get started.")
elif not uploaded_file:
    st.info("👈 Upload a PDF in the sidebar to begin.")

# ── Stats row ──────────────────────────────────────────────────────────────────
if st.session_state.doc_stats:
    stats = st.session_state.doc_stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["pages"]}</div><div class="stat-label">Pages</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["chunks"]}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-num">5</div><div class="stat-label">Top-K Results</div></div>', unsafe_allow_html=True)
    st.markdown(f"<br><div class='main-sub'>📄 {stats['name']}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for turn in st.session_state.chat_history:
        st.markdown(f'<div class="chat-user">🙋 {turn["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bot">🤖 {turn["answer"]}</div>', unsafe_allow_html=True)

# ── Query input ────────────────────────────────────────────────────────────────
if st.session_state.rag_chain:
    st.markdown("### 🔍 Ask a Question")

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("**Suggested questions:**")
        suggestions = [
            "What is the main contribution of this paper?",
            "Which regression model performed best?",
            "What dataset was used for training?",
            "What preprocessing steps were applied?",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if cols[i % 2].button(sug, key=f"sug_{i}"):
                st.session_state.prefill = sug
                st.rerun()

    prefill = st.session_state.pop("prefill", "")
    question = st.text_input(
        "Your question",
        value=prefill,
        placeholder="Ask anything about your PDF…",
        label_visibility="collapsed"
    )

    col_ask, col_clear = st.columns([1, 5])
    ask_clicked = col_ask.button("Ask →")

    if ask_clicked and question.strip():
        with st.spinner("🤔 Thinking…"):
            try:
                response = st.session_state.rag_chain.invoke({"input": question})
                answer = response["answer"]
                sources = response.get("context", [])

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                })

                # Display answer
                st.markdown(f'<div class="answer-card">🤖 {answer}</div>', unsafe_allow_html=True)

                # Source snippets
                if sources:
                    with st.expander(f"📚 View {len(sources)} source chunk(s)"):
                        for i, doc in enumerate(sources, 1):
                            page = doc.metadata.get("page", "?")
                            st.markdown(f'<span class="source-chip">Page {page + 1} · Chunk {i}</span>', unsafe_allow_html=True)
                            st.markdown(f"> {doc.page_content[:300]}…")
                            st.markdown("---")

            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif ask_clicked and not question.strip():
        st.warning("Please enter a question.")

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.rag_chain and not uploaded_file:
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">🌧️</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.5rem; margin-bottom:0.5rem;">
            Ask questions about any PDF
        </div>
        <div style="color:#6b7280; font-size:0.9rem; max-width:400px; margin:0 auto;">
            Upload a PDF and start a conversation. Powered by Google Gemini embeddings 
            and retrieval-augmented generation.
        </div>
    </div>
    """, unsafe_allow_html=True)