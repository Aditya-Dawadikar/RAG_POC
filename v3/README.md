# Simple RAG V3 with FAISS + BM25 + Cross-Encoder Reranking + Groq Llama

A Retrieval-Augmented Generation pipeline using:

* FAISS for semantic vector retrieval
* BM25 for keyword-based retrieval
* Cross-Encoder reranking for final relevance ordering
* Sentence Transformers for embeddings
* Groq-hosted Llama model for answer generation
* Ragas for generation-side faithfulness evaluation
* Custom IR metrics for retrieval evaluation

V3 improves the earlier RAG versions by adding **hybrid retrieval**: FAISS + BM25 + reranking.

---

## Architecture

```text
User Query
   ↓
Query Embedding
   ↓
FAISS Semantic Search
   ↓
BM25 Keyword Search
   ↓
Merge + Deduplicate Candidates
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

## Version Progression

### V1: FAISS-Only RAG

```text
query → FAISS top 3 → Llama
```

V1 established the baseline retrieval and generation pipeline.

### V2: FAISS + Reranker

```text
query → FAISS top 20 → cross-encoder reranker → top 3 → Llama
```

V2 improved ranking quality by using a cross-encoder reranker after FAISS retrieval.

### V3: Hybrid Retrieval + Reranker

```text
query
→ FAISS top 20
→ BM25 top 20
→ merge + deduplicate
→ cross-encoder reranker
→ top 3
→ Llama
```

V3 improves the candidate pool itself by combining semantic and lexical retrieval before reranking.

---

## Why V3 Works Better

FAISS is good at semantic similarity:

```text
query embedding ↔ chunk embedding
```

BM25 is good at exact keyword matching:

```text
names
dates
acronyms
titles
rare entities
specific phrases
```

The cross-encoder reranker then scores:

```text
(query, chunk)
```

as a pair, which gives stronger final relevance ranking.

This combination is much closer to a production-style RAG retrieval stack.

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
│   ├── bm25_retrieve.py
│   ├── reranker.py
│   ├── retrieve_reranked.py
│   ├── retrieve_hybrid.py
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
rank-bm25
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

BM25 uses the same `chunks.jsonl` file at runtime.

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

The FAISS retriever returns:

```text
chunk_id
FAISS similarity score
chunk text
```

---

## Test BM25 Retrieval

Run:

```bash
python src/bm25_retrieve.py
```

BM25 is useful for exact-match-heavy queries such as:

```text
What is the U.S. 1st Infantry Division also called?
What do Maxwell's equations describe?
What is Serial ATA used for?
When was the People's Republic of China founded?
```

The BM25 retriever returns:

```text
chunk_id
BM25 score
chunk text
```

---

## Test Reranked Retrieval

Run:

```bash
python src/retrieve_reranked.py
```

The V2 reranked retriever works as:

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

## Test Hybrid Retrieval

Run:

```bash
python src/retrieve_hybrid.py
```

The V3 hybrid retriever works as:

```text
query
→ FAISS top 20
→ BM25 top 20
→ merge candidates by chunk_id
→ cross-encoder reranking
→ final top 3 chunks
```

Each result includes:

```text
chunk_id
FAISS score
BM25 score
reranker score
retrieval sources
chunk text
```

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

The V3 RAG pipeline:

```text
question
→ retrieve FAISS candidates
→ retrieve BM25 candidates
→ merge and deduplicate candidates
→ rerank merged candidates
→ select top 3 chunks
→ build prompt
→ call Groq Llama
→ print answer and source chunks
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

The same retrieval testset is used across V1, V2, and V3 for fair comparison.

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

## V3 Retrieval Results

```text
========== AVERAGE METRICS ==========
MRR: 1.0
K=1  | P@1=1.0000  | R@1=1.0000  | nDCG@1=1.0000
K=3  | P@3=0.3333  | R@3=1.0000  | nDCG@3=1.0000
K=5  | P@5=0.2000  | R@5=1.0000  | nDCG@5=1.0000
K=10 | P@10=0.1000 | R@10=1.0000 | nDCG@10=1.0000
```

---

## Retrieval Improvement Summary

```text
V1 → V2:
MRR:    0.78 → 0.95
P@1:    0.70 → 0.95
R@3:    0.85 → 0.95
nDCG@3: 0.79 → 0.95

V2 → V3:
MRR:    0.95 → 1.00
P@1:    0.95 → 1.00
R@3:    0.95 → 1.00
nDCG@3: 0.95 → 1.00
```

The V2 reranker improved ranking quality. V3 hybrid retrieval improved the candidate pool before reranking, which pushed retrieval metrics to perfect scores on the current evaluation set.

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

## V3 Generation Results

```text
========== RAGAS GENERATION EVAL ==========
faithfulness: 0.9292
hallucination_rate: 0.0708
```

---

## Generation Improvement Summary

```text
V1 → V2:
faithfulness:        0.8579 → 0.8667
hallucination_rate:  0.1421 → 0.1333

V2 → V3:
faithfulness:        0.8667 → 0.9292
hallucination_rate:  0.1333 → 0.0708
```

The V3 hybrid retrieval pipeline significantly improved grounding quality. Better retrieval context led to higher answer faithfulness and lower hallucination rate.

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

These metrics show whether the retriever finds and ranks the correct chunks.

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

## Rate Limit Notes

Groq free tier may hit request or token limits during Ragas evaluation.

Recommended Ragas config:

```python
RunConfig(
    timeout=600,
    max_retries=5,
    max_wait=120,
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

For Groq free tier, `llama-3.1-8b-instant` may have a low tokens-per-minute limit. If Ragas fails with token or timeout errors, use:

```python
max_tokens=2048
```

and keep the evaluator serial:

```python
max_workers=1
```

---

## Important Files

### `src/ingest.py`

Builds the FAISS index from raw wiki files.

### `src/retrieve.py`

Loads FAISS index and performs base semantic retrieval.

### `src/bm25_retrieve.py`

Loads chunks and performs keyword-based BM25 retrieval.

### `src/reranker.py`

Loads the cross-encoder reranker and scores query-chunk pairs.

### `src/retrieve_reranked.py`

Runs FAISS candidate retrieval followed by cross-encoder reranking.

### `src/retrieve_hybrid.py`

Runs FAISS + BM25 candidate retrieval, merges results, deduplicates by chunk ID, and reranks the combined candidate pool.

### `src/llm.py`

Calls the Groq-hosted Llama model.

### `src/rag.py`

Connects hybrid retrieval and generation.

### `src/test_retrieval_metrics.py`

Computes classic retrieval metrics.

### `src/test_generation_ragas.py`

Runs Ragas-based generation faithfulness evaluation.

---

## Current V3 Results

Retrieval:

```text
MRR: 1.00
P@1: 1.00
R@3: 1.00
nDCG@3: 1.00
```

Generation:

```text
faithfulness: 0.9292
hallucination_rate: 0.0708
```

---

## Suggested CI Gates

```text
MRR >= 0.95
Recall@3 >= 0.95
nDCG@3 >= 0.95
Faithfulness >= 0.90
Hallucination rate <= 0.10
```

These gates are stricter than V1 and V2 because V3 has a much stronger retrieval and grounding baseline.

---

## Current Limitations

The current evaluation set is small and mostly answerable. Retrieval scores are perfect on this testset, but this does not mean the system is production-ready.

Missing robustness tests:

```text
unanswerable questions
ambiguous questions
conversational follow-up questions
adversarial prompt-injection questions
citation correctness checks
latency and cost tracking
larger evaluation set
```

---

## Next Improvements

Since V3 retrieval is already strong on the current benchmark, the next focus should be robustness and reliability.

Recommended next steps:

```text
1. Add unanswerable-question tests
2. Add refusal accuracy evaluation
3. Add ambiguous-query tests
4. Add conversational follow-up query rewriting
5. Add adversarial prompt-injection tests
6. Add citation correctness validation
7. Add latency and cost metrics
8. Add CI threshold gates
```

Recommended V4 architecture:

```text
User Query
→ query rewriting or clarification detection
→ FAISS + BM25 hybrid retrieval
→ merge and deduplicate candidates
→ cross-encoder rerank
→ context selection
→ Groq Llama answer
→ faithfulness + refusal + citation validation
```

---

## Notes

This is an educational RAG system built in stages.

V1 established the baseline FAISS RAG pipeline.

V2 added cross-encoder reranking and showed that ranking quality matters.

V3 added BM25 keyword retrieval and showed that candidate quality matters before reranking.

The main lesson from V3 is that production RAG is not just vector search. Strong RAG systems usually combine semantic retrieval, lexical retrieval, reranking, strict prompting, and evaluation-driven iteration.
