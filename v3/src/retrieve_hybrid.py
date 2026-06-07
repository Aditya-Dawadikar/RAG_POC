from retrieve import retrieve
from bm25_retrieve import bm25_retrieve

from reranker import rerank

FAISS_CANDIDATE_K = 20
BM25_CANDIDATE_K = 20
FINAL_TOP_K = 3

def merge_candidates(faiss_results, bm25_results):
    merged = {}

    for chunk in faiss_results:
        item = dict(chunk)

        item["faiss_score"] = chunk.get("score")
        item["bm25_score"] = None

        item["retrieval_sources"] = ["faiss"]

        merged[item["id"]] = item
    
    for chunk in bm25_results:
        chunk_id = chunk["id"]

        if chunk_id in merged:
            merged[chunk_id]["bm25_score"] = chunk.get("bm25_score", chunk.get("score"))

            merged[chunk_id]["retrieval_sources"].append("bm25")
        else:
            # Copy chunk so we do not mutate original object
            item = dict(chunk)

            # FAISS score is missing for BM25-only chunks
            item["faiss_score"] = None

            # Preserve BM25 score explicitly
            item["bm25_score"] = chunk.get("bm25_score", chunk.get("score"))

            # Track source
            item["retrieval_sources"] = ["bm25"]

            # The reranker expects a generic score field
            item["score"] = item["bm25_score"]

            # Store by chunk id
            merged[chunk_id] = item

    # Return merged candidates as list
    return list(merged.values())

def retrieve_hybrid(
    query,
    faiss_k = FAISS_CANDIDATE_K,
    bm25_k = BM25_CANDIDATE_K,
    top_k = FINAL_TOP_K,
):
    faiss_results = retrieve(query, top_k=faiss_k)

    bm25_results = bm25_retrieve(query, top_k=bm25_k)

    candidates = merge_candidates(
        faiss_results=faiss_results,
        bm25_results=bm25_results,
    )

    reranked_results = rerank(
        query=query,
        retrieved_chunks=candidates,
        top_k=top_k,
    )

    # Mark final retrieval strategy
    for result in reranked_results:
        result["retrieval_strategy"] = "hybrid_faiss_bm25_rerank"

    # Return final top-k reranked chunks
    return reranked_results

# CLI test
def main():
    # Ask user for query
    query = input("Ask a question: ")

    # Run hybrid retrieval
    results = retrieve_hybrid(query)

    # Print results
    print("\nTOP HYBRID RERANKED MATCHES:")

    # Print each result
    for rank, result in enumerate(results, start=1):
        print(
            f"\n--- Result {rank} | "
            f"id={result['id']} | "
            f"rerank_score={result.get('rerank_score'):.4f} | "
            f"faiss_score={result.get('faiss_score')} | "
            f"bm25_score={result.get('bm25_score')} | "
            f"sources={result.get('retrieval_sources')} ---"
        )

        print(result["text"][:1000])


# Run directly
if __name__ == "__main__":
    main()