RAG Application – Google Colab Execution Guide

This project demonstrates a Retrieval-Augmented Generation (RAG) application using:

OpenRouter (LLM API)

FAISS for vector search

Sentence Transformers for embeddings

Streamlit for chatbot UI

Google Colab as the execution environment

🔧 Prerequisites

Before starting, make sure you have:

A Google account (for Google Colab)

A GitHub account

An OpenRouter account (for LLM access)

🔑 Step 1: Create an OpenRouter Account & API Key

Go to 👉 https://openrouter.ai

Sign up / log in

Navigate to API Keys

Click Create API Key

Copy the key (you will need it once)

⚠️ Do NOT commit this key to GitHub

🔐 Step 2: Store OpenRouter API Key in Google Colab Secrets

Open Google Colab

In the left sidebar, click 🔑 Secrets

Add a new secret:

Name: OPENROUTER_API_KEY

Value: <your_openrouter_api_key>

Save it

This allows secure access without hard-coding the key.

🚀 Step 3: Open Project in Google Colab
Option A: Upload Files Manually

Open https://colab.research.google.com

Create a New Notebook

Upload the following files into Colab:

build_index.py
colab_run.py
ui_chatbot_streamlit.py
requirements.txt
insurance_doc.pdf   (or your own PDF)

Option B: Clone from GitHub (Recommended)
!git clone https://github.com/<your-username>/rag-sample.git
%cd rag-sample

📦 Step 4: Install Required Dependencies

Run this in a Colab cell:

!pip install -U \
  streamlit \
  langchain \
  langchain-community \
  langchain-openai \
  langchain-text-splitters \
  pypdf \
  faiss-cpu \
  sentence-transformers


⏳ This may take 1–2 minutes.

🔐 Step 5: Load OpenRouter API Key in Colab

Run the following Python cell:

import os
from google.colab import userdata

os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")

assert os.environ.get("OPENROUTER_API_KEY"), "❌ OPENROUTER_API_KEY not found"
print("✅ OpenRouter key loaded")


If successful, you should see:

✅ OpenRouter key loaded

📂 Step 6: Verify Project Files
!ls


Expected output (example):

build_index.py
colab_run.py
ui_chatbot_streamlit.py
insurance_doc.pdf
sample_data

🧠 Step 7: Build FAISS Vector Index from PDF

Run:

!python build_index.py "insurance_doc.pdf"


Expected output:

Loading PDF...
Pages: X | Chunks: Y
✅ FAISS index saved to ./faiss_index


This step:

Loads the PDF

Splits text into chunks

Generates embeddings

Saves a FAISS index locally

💬 Step 8: Launch Streamlit Chatbot UI

Run:

!streamlit run ui_chatbot_streamlit.py \
  --server.port 8501 \
  --server.address 0.0.0.0


You should see output like:

You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501

🌐 Step 9: Open the Chatbot in Browser

In Google Colab:

Click ⚙️ (Settings) → Ports

Open Port 8501

Click Preview / Open in new tab

🎉 Your RAG chatbot UI is now live.

❓ Step 10: Ask Questions

You can now ask questions like:

“What does this insurance document cover?”

“Summarize the policy exclusions.”

“What are the claim conditions?”

The chatbot will:

Retrieve relevant chunks using FAISS

Send context + question to OpenRouter LLM

Return an accurate, grounded answer