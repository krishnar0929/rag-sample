import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
INDEX_DIR = "faiss_index"

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 PDF RAG Chatbot")
st.caption("Ask questions grounded in your PDF using FAISS + OpenRouter")

api_key = "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"
if not api_key:
    st.error("Missing OPENROUTER_API_KEY. Set it in your terminal and restart Streamlit.")
    st.stop()

# Cache vectorstore so it loads once
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vs

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    api_key=api_key,
    base_url=OPENROUTER_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Streamlit PDF RAG Chatbot",
    },
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Answer ONLY using the provided context. If not in context, say you don't know."}
    ]

# Render history (skip system message)
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input box
prompt = st.chat_input("Ask a question about the PDF...")
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve docs
    docs = retriever.invoke(prompt)
    context = "\n\n".join(
        f"[page {d.metadata.get('page', 'unknown')}] {d.page_content}"
        for d in docs
    )

    # Build LLM messages (system + last few turns)
    history_for_llm = []
    for m in st.session_state.messages[-8:]:
        if m["role"] == "system":
            history_for_llm.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            history_for_llm.append(HumanMessage(content=m["content"]))
        else:
            history_for_llm.append(AIMessage(content=m["content"]))

    # Force grounded answer using context
    history_for_llm.append(
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{prompt}")
    )

    # LLM call
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = llm.invoke(history_for_llm).content
            st.markdown(answer)

            # Sources expander
            with st.expander("Sources"):
                for i, d in enumerate(docs, 1):
                    page = d.metadata.get("page", "unknown")
                    snippet = d.page_content[:250].replace("\n", " ")
                    st.write(f"{i}. page={page}  {snippet}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})