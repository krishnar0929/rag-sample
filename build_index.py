import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
INDEX_DIR = "faiss_index"


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_index.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY")

    print("📄 Loading PDF...")
    docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    print(f"Pages: {len(docs)} | Chunks: {len(chunks)}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)

    print(f"✅ FAISS index saved to ./{INDEX_DIR}")


if __name__ == "__main__":
    main()