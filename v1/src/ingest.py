# We need os to walk through dataset folders
import os

# We need json to write chunks line-by-line without keeping all chunks in RAM
import json

# We need numpy because FAISS expects numpy float32 arrays
import numpy as np

# We need faiss to build the vector index
import faiss

# We use tqdm for progress visibility
from tqdm import tqdm

# We use SentenceTransformer for text embeddings
from sentence_transformers import SentenceTransformer


# Dataset folder
DATA_DIR = "data/plain-text-wikipedia-simpleenglish/1of2"

# Output folder
INDEX_DIR = "indexes"

# FAISS index output path
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

# Chunk metadata output path
CHUNKS_JSONL_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")

# Small embedding model for starter RAG
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of words per chunk
CHUNK_SIZE = 200

# Overlap between chunks
CHUNK_OVERLAP = 40

# Number of chunks embedded at once
BATCH_SIZE = 64

# Total Files to Process
TOTAL_FILES = 20

# This walks through all dataset files one by one
def iter_files(data_dir):
    # Recursively walk all folders
    for root, _, files in os.walk(data_dir):
        # Loop through files in each folder
        for file_name in files:
            # Skip hidden files
            if file_name.startswith("."):
                continue

            # Build full path
            file_path = os.path.join(root, file_name)

            # Yield the file path instead of storing it
            yield file_path


# This reads one file at a time
def read_file(file_path):
    # Open file safely
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        # Return content of this one file
        return f.read()


# This creates chunks from one text document
def iter_chunks_from_text(text):
    # Split text into words
    words = text.split()

    # Start pointer
    start = 0

    # Continue until all words are processed
    while start < len(words):
        # End pointer
        end = start + CHUNK_SIZE

        # Build chunk
        chunk = " ".join(words[start:end])

        # Yield useful chunks only
        if len(chunk.strip()) > 50:
            yield chunk

        # Move forward with overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP


# This embeds one batch and adds it to FAISS
def add_batch_to_index(model, index, chunk_batch, chunk_file, start_id):
    # Create normalized embeddings
    embeddings = model.encode(
        chunk_batch,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Convert embeddings to float32 for FAISS
    embeddings = embeddings.astype("float32")

    # Add embeddings directly to FAISS
    index.add(embeddings)

    # Write chunks to disk immediately
    for offset, chunk in enumerate(chunk_batch):
        # Create chunk record
        record = {
            "id": start_id + offset,
            "text": chunk
        }

        # Write one JSON object per line
        chunk_file.write(json.dumps(record) + "\n")

    # Return next available chunk id
    return start_id + len(chunk_batch)


# Main ingestion pipeline
def main():
    # Create indexes folder
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Get embedding dimension by encoding one test string
    test_embedding = model.encode(
        ["test"],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Extract vector dimension
    dimension = test_embedding.shape[1]

    # Create FAISS index using inner product
    index = faiss.IndexFlatIP(dimension)

    # Current chunk batch
    chunk_batch = []

    # Global chunk id
    chunk_id = 0

    # Count processed files
    file_count = 0

    # Open chunks file once in write mode
    with open(CHUNKS_JSONL_PATH, "w", encoding="utf-8") as chunk_file:
        # Process files one by one
        count = 0
        for file_path in tqdm(iter_files(DATA_DIR), desc="Processing files"):
            count += 1

            if count > TOTAL_FILES:
                break

            print(f"file: {count}")

            # Read only this file into memory
            text = read_file(file_path)

            # Count file
            file_count += 1

            # Create chunks from this file lazily
            for chunk in iter_chunks_from_text(text):
                # Add chunk to small batch
                chunk_batch.append(chunk)

                # Once batch is full, embed and flush
                if len(chunk_batch) >= BATCH_SIZE:
                    # Add batch to index and disk
                    chunk_id = add_batch_to_index(
                        model=model,
                        index=index,
                        chunk_batch=chunk_batch,
                        chunk_file=chunk_file,
                        start_id=chunk_id
                    )

                    # Clear batch from RAM
                    chunk_batch = []

        # Flush remaining chunks
        if chunk_batch:
            # Add final batch
            chunk_id = add_batch_to_index(
                model=model,
                index=index,
                chunk_batch=chunk_batch,
                chunk_file=chunk_file,
                start_id=chunk_id
            )

    # Save FAISS index at the end
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Print summary
    print(f"Processed files: {file_count}")
    print(f"Total chunks indexed: {chunk_id}")
    print(f"Saved FAISS index to: {FAISS_INDEX_PATH}")
    print(f"Saved chunks to: {CHUNKS_JSONL_PATH}")


# Run script directly
if __name__ == "__main__":
    main()