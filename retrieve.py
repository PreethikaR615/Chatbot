
"""
STEP 2 - RETRIEVE
Find relevant document chunks from ChromaDB.

Run:
    python retrieve.py "What does the document contain?"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from ingest import get_collection


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "4"))


# --------------------------------------------------
# HELPER: NORMALIZE SOURCE FILENAMES
# --------------------------------------------------

def normalize_source(source):
    """
    Convert a file path into its filename.
    Example:
    data/report.pdf -> report.pdf
    """

    if not source:
        return None

    return Path(str(source)).name


# --------------------------------------------------
# RETRIEVAL FUNCTION
# --------------------------------------------------

def retrieve(question, top_k=TOP_K, source=None):
    """
    Retrieve relevant chunks from ChromaDB.

    Parameters:
        question: User's question.
        top_k: Maximum number of chunks to retrieve.
        source: Optional filename or list of filenames.

    Returns:
        List of dictionaries containing text and source.
    """

    if not question or not question.strip():
        return []

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = TOP_K

    if top_k <= 0:
        return []

    collection = get_collection()

    # Check whether the database contains documents
    total_count = collection.count()

    if total_count == 0:
        print("ChromaDB is empty.")
        return []

    # --------------------------------------------------
    # PREPARE SOURCE FILTER
    # --------------------------------------------------

    where_filter = None

    if source:
        if isinstance(source, (list, tuple, set)):
            sources = [
                normalize_source(item)
                for item in source
                if item
            ]
            sources = list(set(sources))

            if not sources:
                return []

            if len(sources) == 1:
                where_filter = {"source": sources[0]}
            else:
                where_filter = {"source": {"$in": sources}}

        else:
            source_name = normalize_source(source)
            where_filter = {"source": source_name}

    # --------------------------------------------------
    # CHECK MATCHING DOCUMENT COUNT
    # --------------------------------------------------

    if where_filter:
        matching_results = collection.get(
            where=where_filter,
            include=["metadatas"]
        )

        matching_count = len(matching_results.get("ids", []))

    else:
        matching_count = total_count

    if matching_count == 0:
        print(f"No chunks found for source: {source}")
        return []

    # --------------------------------------------------
    # SEARCH CHROMADB
    # --------------------------------------------------

    number_of_results = min(top_k, matching_count)

    query_args = {
        "query_texts": [question.strip()],
        "n_results": number_of_results,
        "include": ["documents", "metadatas", "distances"]
    }

    if where_filter:
        query_args["where"] = where_filter

    results = collection.query(**query_args)

    # --------------------------------------------------
    # EXTRACT RESULTS
    # --------------------------------------------------

    chunks = []

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    if not documents or not documents[0]:
        return []

    retrieved_documents = documents[0]
    retrieved_metadatas = (
        metadatas[0] if metadatas else []
    )
    retrieved_distances = (
        distances[0] if distances else []
    )

    for index, document_text in enumerate(retrieved_documents):

        if not document_text or not document_text.strip():
            continue

        metadata = (
            retrieved_metadatas[index]
            if index < len(retrieved_metadatas)
            and retrieved_metadatas[index]
            else {}
        )

        distance = (
            retrieved_distances[index]
            if index < len(retrieved_distances)
            else None
        )

        chunks.append({
            "text": document_text,
            "source": metadata.get("source", "Unknown"),
            "distance": distance
        })

    return chunks


# --------------------------------------------------
# TERMINAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What is this document about?"
    )

    print("\nSearching ChromaDB...")
    print(f"Question: {question}\n")

    results = retrieve(question)

    if not results:
        print("No matching documents found.")

    else:
        print(f"Retrieved chunks: {len(results)}\n")

        for index, chunk in enumerate(results, start=1):

            print("=" * 60)
            print(f"CHUNK {index}")
            print(f"Source: {chunk['source']}")
            print(f"Distance: {chunk['distance']}")
            print("-" * 60)
            print(chunk["text"])
            print()
