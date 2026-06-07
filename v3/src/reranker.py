from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker_model = CrossEncoder(RERANKER_MODEL)

def rerank(query, retrieved_chunks, top_k = 3):
    """
    Rerank FAISS-retrieved chunks using a cross-encoder.

    Args:
        query: User question
        retrieved_chunks: List of chunks from retrieve()
        top_k: Number of final chunks to return

    Returns:
        Top-k reranked chunks
    """

    if not retrieved_chunks:
        return []
    
    pairs = [
        [query, chunk["text"]] for chunk in retrieved_chunks
    ]

    rerank_scores = reranker_model.predict(pairs)

    reranked_chunks=[]

    for chunk, rerank_score in zip(retrieved_chunks, rerank_scores):
        updated_chunk = dict(chunk)

        updated_chunk["faiss_score"] = chunk.get("score")
        updated_chunk["rerank_score"] = float(rerank_score)
        updated_chunk["score"] = float(rerank_score)

        reranked_chunks.append(updated_chunk)
    
    reranked_chunks.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True
    )

    return reranked_chunks[:top_k]