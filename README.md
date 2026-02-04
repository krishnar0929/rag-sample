# RAG Sample App

A simple Retrieval-Augmented Generation (RAG) application using:
- LangChain
- FAISS
- Streamlit
- OpenRouter

## Setup
```bash
pip install -r requirements.txt


## Build index
python build_index.py ./sample_data/Indian_Cricket_Players_RAG_Sample.pdf

##Run app
streamlit run ui_chatbot_streamlit.py
