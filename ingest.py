
"""
STEP 1 - INGEST
Multi-format AI Document Assistant

Read files -> Extract text -> Split into chunks
-> Generate embeddings -> Store in ChromaDB

Run:
python ingest.py data/
"""

import base64
import hashlib
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

# ---------- SETTINGS ----------

DB_PATH = os.getenv("DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL", "whisper-large-v3-turbo"
)

VISION_MODEL = os.getenv(
    "VISION_MODEL", "qwen/qwen3.8-27b"
)

TEXT_FILES = [".txt", ".md"]
AUDIO_VIDEO_FILES = [
    ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".mp4", ".webm", ".mpeg"
]
IMAGE_FILES = [".png", ".jpg", ".jpeg", ".webp"]

SUPPORTED_FILES = (
    TEXT_FILES + [".pdf", ".docx"]
    + AUDIO_VIDEO_FILES + IMAGE_FILES
)


# ---------- CHROMADB ----------

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)

    embedder = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder
    )


# ---------- FILE READERS ----------

def read_txt(path):
    return Path(path).read_text(
        encoding="utf-8",
        errors="ignore"
    )


def read_pdf(path):
    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(
                f"\n[Page {page_number}]\n{page_text}"
            )

    return "\n".join(pages)


def read_docx(path):
    doc = Document(path)

    paragraphs = [
        paragraph.text
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    ]

    # Extract table contents too
    for table in doc.tables:
        for row in table.rows:
            paragraphs.append(
                " | ".join(cell.text for cell in row.cells)
            )

    return "\n".join(paragraphs)


def read_audio_or_video(path):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY missing in .env")

    groq = Groq(api_key=api_key)

    with open(path, "rb") as file:
        result = groq.audio.transcriptions.create(
            file=(Path(path).name, file.read()),
            model=WHISPER_MODEL
        )

    return result.text or ""


def read_image(path):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY missing in .env")

    groq = Groq(api_key=api_key)

    ext = Path(path).suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"

    image_b64 = base64.b64encode(
        Path(path).read_bytes()
    ).decode()

    response = groq.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all visible text and describe "
                            "the important information in this image."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{ext};base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content or ""


def read_file(path):
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


# ---------- CHUNKING ----------

def split_into_chunks(
    text,
    size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    if not text or not text.strip():
        return []

    if size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero")

    if overlap < 0 or overlap >= size:
        raise ValueError(
            "CHUNK_OVERLAP must be between 0 and CHUNK_SIZE"
        )

    chunks = []

    start = 0
    step = size - overlap

    while start < len(text):
        chunk = text[start:start + size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# ---------- INGEST FUNCTION ----------

def ingest_file(path):
    path = Path(path)
    file_name = path.name

    print(f"\nProcessing: {file_name}")

    # 1. Extract text
    text = read_file(path)

    if not text or not text.strip():
        raise ValueError(
            "No readable text found. PDF may be scanned/image-based."
        )

    print(f"Extracted characters: {len(text)}")

    # 2. Split into chunks
    chunks = split_into_chunks(text)

    if not chunks:
        raise ValueError("No chunks generated")

    print(f"Generated chunks: {len(chunks)}")

    # 3. Connect to ChromaDB
    collection = get_collection()

    # 4. Create unique IDs using file path and content
    file_hash = hashlib.sha256(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:12]

    ids = [
        f"{file_hash}_{i}"
        for i in range(len(chunks))
    ]

    # 5. Remove previous version of this file
    collection.delete(
        where={"source": file_name}
    )

    # 6. Store chunks and metadata
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=[
            {
                "source": file_name,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
    )

    # 7. Verify storage
    stored_count = collection.count()

    print(f"Successfully stored: {len(chunks)} chunks")
    print(f"Total chunks in database: {stored_count}")

    return len(chunks)


# ---------- MAIN ----------

if __name__ == "__main__":

    folder = Path(
        sys.argv[1] if len(sys.argv) > 1 else "data"
    )

    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    files = [
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_FILES
    ]

    if not files:
        print("No supported files found.")
        sys.exit(0)

    success = 0
    failed = 0

    for file in sorted(files):

        try:
            count = ingest_file(file)
            success += 1

        except Exception as error:
            failed += 1
            print(f"FAILED: {file.name}")
            print(f"Reason: {error}")

    print("\n========== INGESTION SUMMARY ==========")
    print(f"Successful files: {success}")
    print(f"Failed files: {failed}")
    print(f"Total database chunks: {get_collection().count()}")
    print("=======================================")
