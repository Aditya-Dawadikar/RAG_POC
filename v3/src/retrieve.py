import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = "indexes/faiss.index"
CHUNKS_JSONL_PATH = "indexes/chunks.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Missing FAISS Index: {FAISS_INDEX_PATH}")
    
    return faiss.read_index(FAISS_INDEX_PATH)

def load_chunks_by_ids(target_ids):
    target_id = set(target_ids)

    found_chunks = {}

    with open(CHUNKS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            chunk_id = record["id"]

            if chunk_id in target_ids:
                found_chunks[chunk_id] = record["text"]
            
            if len(found_chunks) == len(target_ids):
                break

    return found_chunks

def embed_query(model, query):
    query_embedding = model.encode(
        [query],
        convert_to_numpy = True,
        normalize_embeddings=True
    )

    return query_embedding.astype("float32")

def retrieve(query, top_k = TOP_K):
    index = load_faiss_index()

    model = SentenceTransformer(EMBEDDING_MODEL)
    
    query_embedding = embed_query(model, query)

    scores, ids = index.search(query_embedding, top_k)

    retrieved_ids = ids[0].tolist()
    retrieved_scores = scores[0].tolist()

    valid_results = [
        (chunk_id, score) for chunk_id, score in zip(retrieved_ids, retrieved_scores) if chunk_id!=-1
    ]

    chunks_by_id = load_chunks_by_ids([chunk_id for chunk_id, _ in valid_results])

    results = []

    for chunk_id, score in valid_results:
        if chunk_id in chunks_by_id:
            results.append({
                "id": chunk_id,
                "score": score,
                "text": chunks_by_id[chunk_id]
            })
    
    return results

def main():
    # Ask user for a query
    query = input("Ask a question: ")

    # Retrieve relevant chunks
    results = retrieve(query)

    # Print query
    print("\nQUERY:")
    print(query)

    # Print retrieved chunks
    print("\nTOP MATCHES:")

    # Loop over retrieved results
    for rank, result in enumerate(results, start=1):
        # Print rank, id, and score
        print(f"\n--- Result {rank} | id={result['id']} | score={result['score']:.4f} ---")

        # Print chunk text
        print(result["text"][:1000])


# Run only when executed directly
if __name__ == "__main__":
    main()