from retrieve import retrieve
from retrieve_reranked import retrieve_reranked
from llm import generate_answer


# Based on our retrieval eval, top 3 is the sweet spot
TOP_K = 3


# This builds the RAG prompt using retrieved chunks
def build_prompt(question, retrieved_chunks):
    # Convert chunks into numbered context blocks
    context = "\n\n".join(
        [
            f"[Source {i + 1} | chunk_id={chunk['id']}]\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        ]
    )

    # Return final prompt sent to Llama
    return f"""
Use the context below to answer the question.

Rules:
- Answer only from the context.
- If the context does not contain the answer, say: "I don't know based on the provided context."
- Keep the answer concise.
- Mention the source chunk ids used.

Context:
{context}

Question:
{question}

Answer:
""".strip()


# This runs the full RAG pipeline
def answer_question(question):
    # Retrieve top-k chunks from FAISS
    # retrieved_chunks = retrieve(question, top_k=TOP_K)
    retrieved_chunks = retrieve_reranked(question, top_k=TOP_K)

    # Build prompt from retrieved context
    prompt = build_prompt(question, retrieved_chunks)

    # Generate final answer using Groq Llama
    answer = generate_answer(prompt)

    # Return answer and sources
    return answer, retrieved_chunks


# CLI entrypoint
def main():
    # Ask user for question
    question = input("Ask a question: ")

    # Run RAG
    answer, retrieved_chunks = answer_question(question)

    # Print final answer
    print("\n========== ANSWER ==========")
    print(answer)

    # Print retrieved source metadata
    print("\n========== SOURCES ==========")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\nSource {i}: chunk_id={chunk['id']} | score={chunk['score']:.4f}")
        print(chunk["text"][:500])


# Run only when executed directly
if __name__ == "__main__":
    main()