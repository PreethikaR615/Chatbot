# AI Document Assistant

AI Document Assistant lets you upload documents and ask questions about their contents. It retrieves relevant passages from your files and uses a Groq language model to answer based on those passages.

## Features

- Chat with TXT, Markdown, PDF, and Word documents.
- Transcribe supported audio and video files with Groq Whisper.
- Describe images and extract visible text with a Groq vision model.
- Search document passages by semantic similarity using local embeddings.
- View the source passages used to answer a question.
- Choose a Groq chat model and the number of passages to retrieve.

## Technologies

- Python and Streamlit for the application and web interface.
- Groq for chat, audio transcription, and image understanding.
- ChromaDB and Sentence Transformers for local vector storage and embeddings.
- pypdf and python-docx for reading PDF and Word files.
- python-dotenv for loading local environment settings.

## Requirements

- Python 3.10 or later.
- A Groq API key for chat answers and for processing audio/video or images.
- Internet access when installing packages and calling Groq services. Text embeddings run locally after the embedding model is available.

## Installation

Open a terminal in the project folder.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and replace `your_groq_api_key_here` with your Groq API key. Keep `.env` private; it is excluded from Git.

## Run the application

From the project folder, run:

```bash
streamlit run app.py
```

On Windows, if the virtual environment is not activated, use:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Streamlit opens the app in your browser. Upload files in the sidebar, select **Ingest files**, then ask a question in the chat. The first embedding run may download the Sentence Transformers model.

## Optional command-line use

Put supported documents in the `data` folder, then ingest them:

```bash
python ingest.py data
```

Ask a question against the ingested documents:

```bash
python retrieve.py "What is this document about?"
```

On Windows without activating the virtual environment, use:

```powershell
.\.venv\Scripts\python.exe ingest.py data
.\.venv\Scripts\python.exe retrieve.py "What is this document about?"
```

## Local files and privacy

Uploaded documents are saved under `data/`, and the local ChromaDB index is stored under `chroma_db/`. Both folders are excluded from Git, along with `.env`, virtual environments, and generated Python files. Do not commit API keys, passwords, private documents, or other sensitive information.

When audio/video or image files are processed, their contents are sent to Groq for transcription or image understanding. Review Groq's terms and your organization's data policies before uploading sensitive files.
