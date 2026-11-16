"""
STEP 2 - RETRIEVE
Find the chunks in ChromaDB that are most similar to the question.

Run from terminal:  python retrieve.py "your question"
"""
import os
import sys

from dotenv import load_dotenv

from ingest import get_collection

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "4"))


def retrieve(question, top_k=TOP_K):
    """Return a list of {"text", "source"} for the best matching chunks."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[question], n_results=min(top_k, collection.count()))

    chunks = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": text, "source": meta["source"]})
    return chunks


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    for chunk in retrieve(question):
        print(f"[{chunk['source']}] {chunk['text'][:200]}...\n")
