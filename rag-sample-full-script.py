# ============================
# ✅ ONE-CELL RUN (Colab)
# - Installs deps
# - Writes Streamlit app (runtime PDF upload + Build Index)
# - Launches UI and prints the public Colab URL
# ============================

!pip -q install -U streamlit pypdf faiss-cpu langchain langchain-community langchain-openai tiktoken

import os, textwrap, subprocess, time, socket
from getpass import getpass

# ----------------------------
# 0) Set API env vars (OpenRouter / OpenAI-compatible)
# ----------------------------
# Works with OpenRouter if you set:
#   OPENAI_API_KEY   = your OpenRouter key
#   OPENAI_BASE_URL  = https://openrouter.ai/api/v1
#
# If you already set these in Colab, this will NOT overwrite.
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("Enter OPENAI_API_KEY (OpenRouter key works): ")

if not os.environ.get("OPENAI_BASE_URL"):
    base = input("Enter OPENAI_BASE_URL (press Enter for OpenRouter default https://openrouter.ai/api/v1): ").strip()
    os.environ["OPENAI_BASE_URL"] = base or "https://openrouter.ai/api/v1"

# Optional: some OpenRouter setups like having this (safe to ignore if not needed)
# os.environ["OPENROUTER_SITE_URL"] = "https://your-site.com"
# os.environ["OPENROUTER_APP_NAME"] = "PDF RAG Chatbot"

# ----------------------------
# 1) Write Streamlit app
# ----------------------------
streamlit_app = r"""
import os
import io
import re
import time
import streamlit as st

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


APP_TITLE = "💬 PDF RAG Chatbot"
APP_SUBTITLE = "Upload PDF(s), build an index, and chat with your documents (FAISS + OpenRouter)."

# -------- Helpers --------
def ensure_env():
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL"):
        st.error("Missing OPENAI_API_KEY and/or OPENAI_BASE_URL. Set them before running Streamlit.")
        st.stop()

def is_overview_question(q: str) -> bool:
    ql = (q or "").strip().lower()
    patterns = [
        r"\bwhat\s+is\s+this\s+(pdf|document|policy)\s+about\b",
        r"\bsummary\b",
        r"\boverview\b",
        r"\bhigh\s+level\b",
        r"\bwhat\s+does\s+it\s+say\b",
        r"\bexplain\s+this\s+(pdf|document|policy)\b",
        r"\bwhat\s+is\s+the\s+policy\s+about\b",
    ]
    return any(re.search(p, ql) for p in patterns)

def extract_docs_from_pdfs(uploaded_files):
    docs = []
    for uf in uploaded_files:
        # uf is an UploadedFile
        data = uf.read()
        reader = PdfReader(io.BytesIO(data))
        for page_idx, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            txt = txt.strip()
            if not txt:
                continue
            docs.append(
                Document(
                    page_content=txt,
                    metadata={"source": uf.name, "page": page_idx + 1},
                )
            )
    return docs

def build_vectorstore(docs, embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return None, 0
    vs = FAISS.from_documents(chunks, embeddings)
    return vs, len(chunks)

def format_sources(docs, max_chars=800):
    out = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "N/A")
        page = meta.get("page", "N/A")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."
        out.append({"source": src, "page": page, "snippet": snippet})
    return out

def answer_with_llm(llm, question, retrieved_docs, force_overview=False):
    # Grounded prompt
    context = "\n\n".join(
        [f"[{i+1}] (source={d.metadata.get('source')}, page={d.metadata.get('page')}) {d.page_content}"
         for i, d in enumerate(retrieved_docs)]
    ).strip()

    if not context:
        return "Not found in the provided PDF pages."

    if force_overview:
        sys = (
            "You are a helpful assistant that summarizes documents from provided excerpts only. "
            "Give a high-level overview of what the document is about in 4-8 bullet points. "
            "If the excerpts are not enough to determine the overview, say that clearly."
        )
        user = f"Question: {question}\n\nExcerpts:\n{context}"
    else:
        sys = (
            "You are a helpful assistant answering questions using ONLY the provided excerpts. "
            "If the answer is not present in the excerpts, say: 'Not found in the provided PDF pages.' "
            "Be concise and accurate."
        )
        user = f"Question: {question}\n\nExcerpts:\n{context}"

    resp = llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])
    return resp.content if hasattr(resp, "content") else str(resp)

# -------- App UI --------
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

ensure_env()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Top K chunks", min_value=2, max_value=12, value=8, step=1)

    st.divider()
    st.subheader("Model settings")
    # OpenRouter commonly supports lots of models. Keep default to something safe.
    model_name = st.text_input("Model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

# Upload + Build Index
st.subheader("Upload PDF(s)")
uploaded = st.file_uploader(
    "Upload one or multiple PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

colA, colB = st.columns([1, 2])
with colA:
    build = st.button("Build Index", type="primary", use_container_width=True)
with colB:
    if st.session_state.vectorstore is not None:
        st.success(
            f"Index ready: {st.session_state.chunks_count} chunks | Files: {', '.join(st.session_state.indexed_files)}"
        )
    else:
        st.info("Upload PDFs and click **Build Index** to start chatting.")

if build:
    if not uploaded:
        st.warning("Please upload at least one PDF.")
        st.stop()

    with st.spinner("Reading PDFs and building FAISS index..."):
        # Create embeddings + llm
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )
        docs = extract_docs_from_pdfs(uploaded)
        vs, n_chunks = build_vectorstore(docs, embeddings)

        if vs is None:
            st.error("Could not extract any text from the uploaded PDFs.")
            st.stop()

        st.session_state.vectorstore = vs
        st.session_state.indexed_files = [u.name for u in uploaded]
        st.session_state.chunks_count = n_chunks

        # Optional: auto-add a helpful first message
        st.session_state.messages.append(
            {"role": "assistant", "content": "✅ Index built. Ask me anything about the uploaded PDF(s).", "sources": []}
        )
    st.rerun()

st.divider()

# Chat UI (like your previous one)
st.subheader("Chat")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        srcs = m.get("sources") or []
        if srcs:
            with st.expander("Sources (top matches)"):
                for i, s in enumerate(srcs, 1):
                    st.markdown(f"**{i}. {s['source']} | Page: {s['page']}**")
                    st.write(s["snippet"])

# Input box
user_q = st.chat_input("Ask a question about the PDF(s)...")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})

    if st.session_state.vectorstore is None:
        st.session_state.messages.append(
            {"role": "assistant", "content": "Please upload PDF(s) and click **Build Index** first.", "sources": []}
        )
        st.rerun()

    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )

        vs = st.session_state.vectorstore
        retriever = vs.as_retriever(search_kwargs={"k": int(top_k)})

        # Retrieve
        docs = retriever.invoke(user_q)
        sources = format_sources(docs[: int(top_k)])

        # If user asks overview, force summary-style answer and (optionally) pull a few extra chunks
        force_overview = is_overview_question(user_q)
        if force_overview:
            # for overview, retrieve a bit more context if available
            extra_k = min(12, max(int(top_k), 10))
            docs = vs.as_retriever(search_kwargs={"k": extra_k}).invoke(user_q)
            sources = format_sources(docs[: extra_k])

        answer = answer_with_llm(llm, user_q, docs[: (12 if force_overview else int(top_k))], force_overview)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()

    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"Error while answering: {e}", "sources": []})
        st.rerun()
"""

with open("ui_chatbot_streamlit.py", "w") as f:
    f.write(textwrap.dedent(streamlit_app))

print("✅ Wrote ui_chatbot_streamlit.py")

# ----------------------------
# 2) Start Streamlit on an available port
# ----------------------------
def find_free_port(start=8501, end=8600):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found in range.")

# stop any old streamlit
subprocess.run("pkill -f streamlit || true", shell=True)

PORT = find_free_port(8501, 8600)
print(f"✅ Using port: {PORT}")

p = subprocess.Popen(
    [
        "streamlit", "run", "ui_chatbot_streamlit.py",
        "--server.port", str(PORT),
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

time.sleep(2)
print("✅ Streamlit started.")

# ----------------------------
# 3) Print the public Colab URL
# ----------------------------
from google.colab import output
url = output.eval_js(f"google.colab.kernel.proxyPort({PORT})")
print("\n🌐 Open this URL:\n", url)
