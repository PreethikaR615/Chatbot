"""
STEP 1 - INGEST
Read text from files -> split into chunks -> embed -> store in ChromaDB.

Run from terminal:  python ingest.py data/
"""
import base64
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from docx import Document
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()

# ---- Settings (from .env, with defaults) ----
DB_PATH = os.getenv("DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")

TEXT_FILES = [".txt", ".md"]
AUDIO_VIDEO_FILES = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm", ".mpeg"]
IMAGE_FILES = [".png", ".jpg", ".jpeg", ".webp"]
SUPPORTED_FILES = TEXT_FILES + [".pdf", ".docx"] + AUDIO_VIDEO_FILES + IMAGE_FILES


def get_collection():
    """Open (or create) the ChromaDB collection. Chroma embeds text for us."""
    client = chromadb.PersistentClient(path=DB_PATH)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)


# ---------- Readers: each one returns plain text ----------

def read_txt(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(path):
    doc = Document(path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def read_audio_or_video(path):
    """Speech-to-text with Groq Whisper (file must be under 25 MB)."""
    groq = Groq()
    with open(path, "rb") as f:
        result = groq.audio.transcriptions.create(file=(Path(path).name, f.read()), model=WHISPER_MODEL)
    return result.text


def read_image(path):
    """Ask a Groq vision model to describe the image and copy any text in it."""
    groq = Groq()
    ext = Path(path).suffix.lower().replace(".", "").replace("jpg", "jpeg")
    image_b64 = base64.b64encode(Path(path).read_bytes()).decode()
    response = groq.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail and write out any text you see in it."},
                {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{image_b64}"}},
            ],
        }],
    )
    return response.choices[0].message.content


def read_file(path):
    """Pick the right reader based on the file extension."""
    ext = Path(path).suffix.lower()
    if ext in TEXT_FILES:
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext in AUDIO_VIDEO_FILES:
        return read_audio_or_video(path)
    if ext in IMAGE_FILES:
        return read_image(path)
    raise ValueError(f"Unsupported file type: {ext}")


# ---------- Chunking ----------

def split_into_chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cut text into overlapping pieces so no idea is lost at a boundary."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


# ---------- Main ingest function ----------

def ingest_file(path):
    """Read one file, chunk it, and save the chunks in the DB. Returns chunk count."""
    file_name = Path(path).name
    text = read_file(path)
    chunks = split_into_chunks(text)
    if not chunks:
        return 0

    collection = get_collection()
    collection.delete(where={"source": file_name})  # remove old version of this file
    collection.add(
        ids=[f"{file_name}-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": file_name} for _ in chunks],
    )
    return len(chunks)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    for file in sorted(Path(folder).iterdir()):
        if file.suffix.lower() in SUPPORTED_FILES:
            count = ingest_file(str(file))
            print(f"Ingested {file.name}: {count} chunks")
