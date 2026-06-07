# Simple RAG V2 with FAISS + Cross-Encoder Reranking + Groq Llama

A Retrieval-Augmented Generation pipeline using:

* FAISS for fast vector candidate retrieval
* Sentence Transformers for embeddings
* Cross-Encoder reranking for improved relevance ordering
* Groq-hosted Llama model for generation
* Ragas for generation-side faithfulness evaluation
* Custom IR metrics for retrieval evaluation

V2 improves the V1 pipeline by adding a reranking stage after FAISS retrieval.

---

## Architecture

```text
User Query
   ↓
Query Embedding
   ↓
FAISS Vector Search
   ↓
Top-N Candidate Chunks
   ↓
Cross-Encoder Reranker
   ↓
Top-K Reranked Chunks
   ↓
RAG Prompt
   ↓
Groq Llama Model
   ↓
Final Answer with Source Chunk IDs
```

---

## What Changed in V2?

V1 used direct FAISS retrieval:

```text
query → FAISS top 3 → Llama
```

V2 uses FAISS as a candidate generator and then reranks the candidates:

```text
query → FAISS top 20 → cross-encoder reranker → top 3 → Llama
```

This is closer to production RAG systems because FAISS is fast but not always precise, while a cross-encoder can score the query and chunk together for better ranking quality.

---

## Project Structure

```text
RAG_POC/
│
├── data/
│   └── plain-text-wikipedia-simpleenglish/
│       ├── wiki_00
│       ├── wiki_01
│       └── ...
│
├── indexes/
│   ├── faiss.index
│   └── chunks.jsonl
│
├── evals/
│   ├── retrieval_testset.json
│   └── generation_testset.json
│
├── src/
│   ├── ingest.py
│   ├── retrieve.py
│   ├── reranker.py
│   ├── retrieve_reranked.py
│   ├── rag.py
│   ├── llm.py
│   ├── test_retrieval_metrics.py
│   └── test_generation_ragas.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
faiss-cpu
sentence-transformers
numpy
tqdm
pandas
datasets
ragas
groq
langchain-groq
langchain-huggingface
python-dotenv
```

The reranker uses:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This is loaded through `sentence-transformers`.

---

### 3. Add Groq API key

Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
GROQ_JUDGE_MODEL=llama-3.1-8b-instant
```

---

## Dataset

This project uses the Kaggle Simple English Wikipedia dataset.

Download:

```bash
kaggle datasets download -d ffatty/plain-text-wikipedia-simpleenglish
```

Unzip into:

```text
data/plain-text-wikipedia-simpleenglish/
```

Expected files:

```text
wiki_00
wiki_01
wiki_02
...
```

---

## Build FAISS Index

Run ingestion:

```bash
python src/ingest.py
```

This creates:

```text
indexes/faiss.index
indexes/chunks.jsonl
```

The ingestion pipeline:

```text
read wiki files
→ split into chunks
→ embed chunks
→ add vectors to FAISS
→ save chunk text as JSONL
```

---

## Test Base FAISS Retrieval

Run:

```bash
python src/retrieve.py
```

Example queries:

```text
Who was Albert Einstein?
What is photosynthesis?
What is Python?
```

The base retriever returns:

```text
chunk_id
FAISS similarity score
chunk text
```

---

## Test Reranked Retrieval

Run:

```bash
python src/retrieve_reranked.py
```

The V2 retriever works as:

```text
query
→ FAISS top 20 candidates
→ cross-encoder reranking
→ final top 3 chunks
```

Each result includes:

```text
chunk_id
FAISS score
reranker score
chunk text
```

---

## Retrieval Evaluation

Run:

```bash
python src/test_retrieval_metrics.py
```

This evaluates:

* Precision@K
* Recall@K
* MRR
* nDCG@K

The same retrieval testset is used for both V1 and V2.

---

## V1 Retrieval Results

```text
========== AVERAGE METRICS ==========
MRR: 0.78
K=1  | P@1=0.7000  | R@1=0.7000  | nDCG@1=0.7000
K=3  | P@3=0.2833  | R@3=0.8500  | nDCG@3=0.7946
K=5  | P@5=0.1700  | R@5=0.8500  | nDCG@5=0.7946
K=10 | P@10=0.0900 | R@10=0.9000 | nDCG@10=0.8091
```

---

## V2 Retrieval Results

```text
========== AVERAGE METRICS ==========
MRR: 0.95
K=1  | P@1=0.9500  | R@1=0.9500  | nDCG@1=0.9500
K=3  | P@3=0.3167  | R@3=0.9500  | nDCG@3=0.9500
K=5  | P@5=0.1900  | R@5=0.9500  | nDCG@5=0.9500
K=10 | P@10=0.0950 | R@10=0.9500 | nDCG@10=0.9500
```

---

## Retrieval Improvement Summary

```text
MRR:    0.78 → 0.95
P@1:    0.70 → 0.95
R@3:    0.85 → 0.95
nDCG@3: 0.79 → 0.95
```

The biggest improvement is in ranking quality. FAISS was already finding useful candidates, but the cross-encoder reranker was much better at placing the most relevant chunk at rank 1.

---

## Run Full RAG

Run:

```bash
python src/rag.py
```

Example:

```text
Ask a question: What is the capital city of Scotland?
```

The V2 RAG pipeline:

```text
question
→ retrieve FAISS top 20
→ rerank candidates
→ select top 3 chunks
→ build prompt
→ call Groq Llama
→ print answer and source chunks
```

---

## Generation Evaluation with Ragas

Run:

```bash
python src/test_generation_ragas.py
```

Currently enabled metric:

```python
faithfulness
```

This measures whether the generated answer is supported by the retrieved context.

Hallucination rate is calculated as:

```text
hallucination_rate = 1 - faithfulness
```

---

## V1 Generation Results

```text
========== RAGAS GENERATION EVAL ==========
faithfulness: 0.8579
hallucination_rate: 0.1421
```

---

## V2 Generation Results

```text
========== RAGAS GENERATION EVAL ==========
faithfulness: 0.8667
hallucination_rate: 0.1333
```

---

## Generation Improvement Summary

```text
faithfulness:        0.8579 → 0.8667
hallucination_rate:  0.1421 → 0.1333
```

The generation improvement is smaller than the retrieval improvement. This is expected because reranking improves the context order, but hallucination also depends on prompt strictness, answer length, model behavior, and whether the retrieved chunks fully contain the answer.

---

## Evaluation Philosophy

This project separates evaluation into two layers.

### Retrieval Evaluation

Retrieval is tested without an LLM.

Metrics:

```text
Precision@K
Recall@K
MRR
nDCG@K
```

This tells us whether the retriever finds and ranks the correct chunks.

### Generation Evaluation

Generation is tested with Ragas.

Metrics available:

```text
faithfulness
answer_correctness
answer_relevancy
context_precision
context_recall
```

Currently only `faithfulness` is enabled to avoid Groq free-tier rate limits.

---

## Why Reranking Helped

FAISS uses vector similarity:

```text
query embedding ↔ chunk embedding
```

This is fast and good for broad semantic search, but it may not always rank the best chunk first.

The cross-encoder reranker scores:

```text
(query, chunk)
```

together as a pair. This gives it stronger relevance judgment because it sees the full query and full candidate chunk at the same time.

This is why V2 significantly improved:

```text
MRR
Precision@1
nDCG@3
```

---

## Rate Limit Notes

Groq free tier may hit request or token limits during Ragas evaluation.

Recommended Ragas config:

```python
RunConfig(
    timeout=600,
    max_retries=3,
    max_wait=90,
    max_workers=1,
    log_tenacity=True,
)
```

Recommended first metric:

```python
metrics=[
    faithfulness,
]
```

Add other metrics only after the basic run is stable.

---

## Important Files

### `src/ingest.py`

Builds FAISS index from raw wiki files.

### `src/retrieve.py`

Loads FAISS index and performs base FAISS retrieval.

### `src/reranker.py`

Loads the cross-encoder reranker and scores query-chunk pairs.

### `src/retrieve_reranked.py`

Retrieves FAISS candidates and reranks them into the final top-k results.

### `src/llm.py`

Calls Groq-hosted Llama model.

### `src/rag.py`

Connects reranked retrieval and generation.

### `src/test_retrieval_metrics.py`

Computes classic retrieval metrics.

### `src/test_generation_ragas.py`

Runs Ragas-based generation evaluation.

---

## Current V2 Results

Retrieval:

```text
MRR: 0.95
P@1: 0.95
R@3: 0.95
nDCG@3: 0.95
```

Generation:

```text
faithfulness: 0.8667
hallucination_rate: 0.1333
```

---

## Suggested CI Gates

```text
MRR >= 0.90
Recall@3 >= 0.90
nDCG@3 >= 0.90
Faithfulness >= 0.85
Hallucination rate <= 0.15
```

These gates are stricter than V1 because V2 has a much stronger retrieval baseline.

---

## Next Improvements

Possible next steps:

```text
1. Add BM25 keyword search
2. Combine BM25 + FAISS into hybrid retrieval
3. Rerank merged candidates
4. Add answer_correctness in Ragas
5. Add refusal tests for unanswerable questions
6. Add latency and cost tracking
7. Add CI threshold gates
```

Recommended V3 architecture:

```text
Query
→ FAISS top 20
→ BM25 top 20
→ merge and deduplicate candidates
→ cross-encoder rerank
→ top 3 chunks
→ Groq Llama
→ answer
→ evaluate retrieval and generation
```

---

## Notes

This is an educational RAG system built in stages. V1 established a working FAISS-based RAG baseline. V2 added reranking and showed a clear improvement in retrieval quality.

The main lesson from V2 is that candidate retrieval and ranking are separate problems. FAISS is good at finding possible matches quickly, while reranking is better at ordering those matches by actual relevance.
