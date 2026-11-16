"""
STEP 3 - RAG + UI
Upload files, ask questions, and get answers from a Groq LLM using your documents.

Run:  streamlit run app.py
"""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from ingest import SUPPORTED_FILES, ingest_file
from retrieve import retrieve

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]


def ask_llm(question, chunks, model):
    """Put the retrieved chunks into the prompt and ask the LLM."""
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    system_prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}"
    )
    response = Groq().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


# ---------- Page ----------
st.set_page_config(page_title="File RAG", page_icon="📄")
st.title("📄 Chat with your files")

# ---------- Sidebar: settings + upload ----------
with st.sidebar:
    model = st.selectbox("Model", MODELS)
    top_k = st.slider("Chunks to retrieve", 1, 10, 4)

    st.divider()
    files = st.file_uploader(
        "Upload files",
        type=[ext.strip(".") for ext in SUPPORTED_FILES],
        accept_multiple_files=True,
    )
    if st.button("Ingest files", disabled=not files):
        Path(DATA_DIR).mkdir(exist_ok=True)
        for file in files:
            save_path = Path(DATA_DIR) / file.name
            save_path.write_bytes(file.getvalue())
            with st.spinner(f"Ingesting {file.name}..."):
                try:
                    count = ingest_file(str(save_path))
                    st.success(f"{file.name}: {count} chunks")
                except Exception as error:
                    st.error(f"{file.name}: {error}")

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about your files...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks = retrieve(question, top_k)
            if not chunks:
                answer = "No documents found. Upload and ingest some files first."
            else:
                try:
                    answer = ask_llm(question, chunks, model)
                except Exception as error:
                    answer = f"⚠️ LLM error with `{model}`: {error}"
        st.markdown(answer)
        if chunks:
            with st.expander("Sources"):
                for chunk in chunks:
                    st.markdown(f"**{chunk['source']}**")
                    st.caption(chunk["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
