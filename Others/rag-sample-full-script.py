# ================================
# 🚀 ONE-CELL RUN-ALL (Colab) - PDF + Website RAG Chatbot (FAISS + OpenRouter)
# UI-only secrets, MMR retrieval, and summary-intent handling
# ================================

!pip install -q streamlit langchain langchain-community langchain-openai faiss-cpu pypdf beautifulsoup4 requests==2.32.4 tiktoken

import time, subprocess

app_code = r'''
import streamlit as st
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="PDF + Website RAG Chatbot", layout="wide")

# ----------------------------
# Sidebar: controls & settings
# ----------------------------
st.sidebar.title("Controls")
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

st.sidebar.subheader("Retrieval settings")
top_k = st.sidebar.slider("Top K chunks", 1, 20, 8)

st.sidebar.subheader("Model settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

model_options = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-pro",
]
model_choice = st.sidebar.selectbox("Chat model", model_options + ["(Custom)"], index=0)
chat_model = st.sidebar.text_input("Custom chat model") if model_choice == "(Custom)" else model_choice

embed_options = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "(Custom)",
]
embed_choice = st.sidebar.selectbox("Embeddings model", embed_options, index=0)
embed_model = st.sidebar.text_input("Custom embeddings model") if embed_choice == "(Custom)" else embed_choice

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Secrets (UI only)")
st.sidebar.caption("Paste your OpenRouter key (starts with **sk-or-**).")
openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

openrouter_base_url = "https://openrouter.ai/api/v1"


if not openrouter_api_key:
    st.warning("Please paste your OpenRouter API Key in the sidebar.")
    st.stop()

# ----------------------------
# Main UI
# ----------------------------
st.title("💬 PDF + Website RAG Chatbot")
st.caption("Upload PDFs or provide a website URL, build an index, then chat. (FAISS + OpenRouter)")

st.info(
    f"Provider: OpenRouter | Chat model: {chat_model} | Embeddings: {embed_model} | "
    f"Temp: {temperature} | Top K: {top_k}"
)

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ----------------------------
# Helpers
# ----------------------------
def fetch_website_text(url: str) -> str:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")

def build_clients():
    embeddings = OpenAIEmbeddings(
        model=embed_model,
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    )
    llm = ChatOpenAI(
    model=chat_model,
    temperature=temperature,
    api_key=openrouter_api_key,
    base_url=openrouter_base_url,
    )

    return embeddings, llm

# ----------------------------
# Upload + Build Index
# ----------------------------
st.header("Upload Sources")
uploaded_files = st.file_uploader("Upload one or multiple PDFs", type=["pdf"], accept_multiple_files=True)
website_url = st.text_input("Optional: Website URL (example: https://example.com)")

if st.button("Build Index"):
    documents = []

    # PDFs
    if uploaded_files:
        for f in uploaded_files:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            text = text.strip()
            if text:
                documents.append(Document(page_content=text, metadata={"source": f.name}))
            else:
                st.warning(f"⚠️ No extractable text found in {f.name}. If it's scanned, OCR is needed.")

    # Website
    if website_url:
        try:
            text = fetch_website_text(website_url).strip()
            if text:
                documents.append(Document(page_content=text, metadata={"source": website_url}))
        except Exception as e:
            st.error(f"Website read error: {e}")

    if not documents:
        st.warning("Please upload PDFs and/or provide a website URL.")
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        split_docs = splitter.split_documents(documents)

        try:
            embeddings, _ = build_clients()
            st.session_state.vectorstore = FAISS.from_documents(split_docs, embeddings)
            st.success(f"✅ Index built: {len(split_docs)} chunks")
        except Exception as e:
            st.error(f"Index build failed: {e}")

# ----------------------------
# Chat
# ----------------------------
st.header("Chat")

if not st.session_state.vectorstore:
    st.info("Build the index first, then you can start chatting.")
else:
    # Show history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    if q := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": q})

        with st.chat_message("assistant"):
            try:
                _, llm = build_clients()

                # Better retrieval for broad questions
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": top_k, "fetch_k": max(20, top_k * 4), "lambda_mult": 0.5},
                )
                docs = retriever.invoke(q)

                # Debug: show what was retrieved
                with st.expander("🔎 Retrieved context (debug)"):
                    st.write("Retrieved chunks:", len(docs))
                    for i, d in enumerate(docs, 1):
                        st.write(f"{i}. Source: {d.metadata.get('source')}")
                        st.code((d.page_content or "")[:600])

                context = "\n\n".join([(d.page_content or "") for d in docs]).strip()

                ql = q.lower()
                summary_intent = any(
                    phrase in ql
                    for phrase in [
                        "what is this about", "what's this about", "about this",
                        "summary", "summarize", "overview", "high level",
                        "what is the policy", "what's the policy", "policy about",
                        "explain this document", "document about",
                    ]
                )

                if summary_intent:
                    if not context or len(context) < 200:
                        prompt = f"""
You do NOT have enough retrieved text to summarize confidently.

Ask up to 2 short clarifying questions to understand what part the user wants
(e.g., topic, keyword, section name), and suggest they try:
- asking a more specific question
- increasing Top K
- uploading a text-based PDF (not scanned)

User question: {q}
"""
                    else:
                        prompt = f"""
You are summarizing the provided sources.

Write:
1) A 2-3 sentence summary
2) 5-8 bullet key points
3) Mention main entities/terms you see (if any)
4) List sources used (from metadata 'source')

Sources:
{context}

User question:
{q}
"""
                else:
                    prompt = f"""
Answer using ONLY the context below.
If the answer is not in the context, say exactly:
Not found in the provided sources.

Context:
{context}

Question:
{q}
"""

                ans = llm.invoke(prompt).content
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})

            except Exception as e:
                st.error(f"Chat failed: {e}")
'''

with open("app.py", "w") as f:
    f.write(app_code)

# Restart Streamlit
subprocess.run("pkill -f streamlit", shell=True)
subprocess.Popen([
    "streamlit","run","app.py",
    "--server.port","8501",
    "--server.address","0.0.0.0",
    "--server.enableCORS","false",
    "--server.enableXsrfProtection","false",
    "--browser.gatherUsageStats","false",
    "--server.fileWatcherType","none"
])

time.sleep(2)

from google.colab import output
print("✅ Open this URL:")
print(output.eval_js("google.colab.kernel.proxyPort(8501)"))