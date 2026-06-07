from retrieve import retrieve
from reranker import rerank

FAISS_CANDIDATE_K = 20
FINAL_TOP_K = 3

def retrieve_reranked(query, candidate_k=FAISS_CANDIDATE_K, top_k=FINAL_TOP_K):
    """
    Retrieve many candidates using FAISS, then rerank down to top_k.
    """

    faiss_chunks = retrieve(query, top_k=candidate_k)

    reranked_chunks = rerank(
        query=query,
        retrieved_chunks=faiss_chunks,
        top_k=top_k
    )

    return reranked_chunks

def main():
    # Ask user for query
    query = input("Ask a question: ")

    # Run reranked retrieval
    results = retrieve_reranked(query)

    # Print results
    print("\nTOP RERANKED MATCHES:")

    # Print each result
    for rank, result in enumerate(results, start=1):
        print(
            f"\n--- Result {rank} | "
            f"id={result['id']} | "
            f"rerank_score={result['rerank_score']:.4f} | "
            f"faiss_score={result['faiss_score']:.4f} ---"
        )
        print(result["text"][:1000])


if __name__ == "__main__":
    main()