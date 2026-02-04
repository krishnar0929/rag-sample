RAG Application – Google Colab Execution Guide

This repository demonstrates a Retrieval-Augmented Generation (RAG) application using:

OpenRouter (LLM API)

FAISS (vector search)

Sentence Transformers (embeddings)

Streamlit (chatbot UI)

Google Colab (execution environment)

1. Prerequisites

Before you begin, ensure you have:

Google account (for Google Colab)

GitHub account

OpenRouter account (for LLM access)

2. Create OpenRouter Account & API Key

Go to:
👉 https://openrouter.ai

Sign up or log in

Navigate to API Keys

Click Create API Key

Copy the API key
⚠️ You will not be able to view it again

3. Store OpenRouter API Key in Google Colab Secrets

Open Google Colab

In the left sidebar, click 🔑 Secrets

Add a new secret:

Name: OPENROUTER_API_KEY

Value: <your_openrouter_api_key>

Click Save

This keeps your API key secure and out of source code.

4. Open the Project in Google Colab
Option A: Clone from GitHub (Recommended)

Run the following in a Colab cell:

!git clone https://github.com/krishnar0929/rag-sample.git
%cd rag-sample

Option B: Upload Files Manually

Upload these files into Colab:

build_index.py
colab_run.py
ui_chatbot_streamlit.py
requirements.txt
insurance_doc.pdf   (or any PDF)

5. Install Dependencies

Run this cell in Google Colab:

!pip install -U \
  streamlit \
  langchain \
  langchain-community \
  langchain-openai \
  langchain-text-splitters \
  pypdf \
  faiss-cpu \
  sentence-transformers


Wait until installation completes.

6. Load OpenRouter API Key in Colab

Run this Python cell:

import os
from google.colab import userdata

os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")

assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY not found"
print("OpenRouter API key loaded successfully")


Expected output:

OpenRouter API key loaded successfully

7. Verify Project Files
!ls


You should see:

build_index.py
colab_run.py
ui_chatbot_streamlit.py
insurance_doc.pdf
sample_data

8. Build FAISS Vector Index from PDF

Run:

!python build_index.py "insurance_doc.pdf"


Expected output:

Loading PDF...
Pages: X | Chunks: Y
FAISS index saved to ./faiss_index


This step:

Reads the PDF

Splits text into chunks

Creates embeddings

Stores vectors in FAISS

9. Launch the Streamlit Chatbot

Run:

!streamlit run ui_chatbot_streamlit.py \
  --server.port 8501 \
  --server.address 0.0.0.0


You should see a message like:

You can now view your Streamlit app in your browser.

10. Open the Chatbot UI in Browser

In Google Colab, click Settings (⚙️)

Go to Ports

Open Port 8501

Click Open in new tab

Your chatbot UI is now live.

11. Ask Questions

You can now ask questions such as:

What is this document about?

Summarize the insurance coverage

What are the exclusions?

Explain claim conditions

The system will:

Retrieve relevant text using FAISS

Send context to OpenRouter LLM

Generate grounded answers

12. Project Structure
rag-sample/
├── build_index.py           # Builds FAISS index
├── colab_run.py             # Optional orchestration
├── ui_chatbot_streamlit.py  # Streamlit chatbot UI
├── requirements.txt
├── insurance_doc.pdf        # Sample document
├── faiss_index/             # Generated vector store
└── sample_data/