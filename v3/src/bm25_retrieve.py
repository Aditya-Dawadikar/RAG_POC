import json
import os
import re
from rank_bm25 import BM25Okapi

CHUNKS_JSONL_PATH = "indexes/chunks.jsonl"
TOP_K = 20

def tokenize(text):
    text = text.lower()

    tokens = re.findall(r"\b\w+\b", text)

    return tokens

def load_chunks():
    if not os.path.exists(CHUNKS_JSONL_PATH):
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_JSONL_PATH}")
    
    chunks = []

    with open(CHUNKS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            chunks.append({
                "id": record["id"],
                "text": record["text"]
            })
        
    return chunks

def build_bm25_index(chunks):
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25

_chunks_cache = None
_bm25_cache = None

def get_bm25():
    global _chunks_cache, _bm25_cache

    if _chunks_cache is not None and _bm25_cache is not None:
        return _chunks_cache, _bm25_cache
    
    print("Loading chunks for BM25...")

    _chunks_cache = load_chunks()

    print("Building BM25 index...")

    _bm25_cache = build_bm25_index(_chunks_cache)

    print(f"BM25 index ready. Chunks loaded: {len(_chunks_cache)}")

    # Return cached objects
    return _chunks_cache, _bm25_cache

def bm25_retrieve(query, top_k = TOP_K):
    chunks, bm25 = get_bm25()
    
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    top_indices = ranked_indices[:top_k]

    results = []

    for index in top_indices:
        if scores[index] <= 0:
            continue

        chunk = chunks[index]

        results.append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "score": float(scores[index]),
                "bm25_score": float(scores[index]),
                "retrieval_source": "bm25",
            }
        )

    return results

# CLI test
def main():
    # Ask user for query
    query = input("Ask a question: ")

    # Run BM25 retrieval
    results = bm25_retrieve(query, top_k=TOP_K)

    # Print results
    print("\nTOP BM25 MATCHES:")

    # Print each result
    for rank, result in enumerate(results, start=1):
        print(
            f"\n--- Result {rank} | "
            f"id={result['id']} | "
            f"bm25_score={result['bm25_score']:.4f} ---"
        )

        print(result["text"][:1000])


# Run directly
if __name__ == "__main__":
    main()