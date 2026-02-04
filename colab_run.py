import os
import subprocess

# Optional: launch streamlit app
subprocess.run([
    "streamlit", "run", "ui_chatbot_streamlit.py",
    "--server.port", "8501",
    "--server.address", "0.0.0.0"
])