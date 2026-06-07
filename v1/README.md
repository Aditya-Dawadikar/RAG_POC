# Simple RAG with FAISS + Groq Llama

A simple Retrieval-Augmented Generation pipeline using:

- FAISS for vector search
- Sentence Transformers for embeddings
- Groq-hosted Llama model for generation
- Ragas for generation-side evaluation
- Custom IR metrics for retrieval evaluation

---

## Architecture

```text
User Query
   ↓
Query Embedding
   ↓
FAISS Vector Search
   ↓
Top-K Retrieved Chunks
   ↓
RAG Prompt
   ↓
Groq Llama Model
   ↓
Final Answer with Source Chunk IDs
````

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

## Test Retrieval

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

The retriever returns top matching chunks with:

```text
chunk_id
similarity score
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

Example output:

```text
========== AVERAGE METRICS ==========
MRR: 0.7800
K=1 | P@1=0.7000 | R@1=0.7000 | nDCG@1=0.7000
K=3 | P@3=0.2833 | R@3=0.8500 | nDCG@3=0.7946
K=5 | P@5=0.1700 | R@5=0.8500 | nDCG@5=0.7946
K=10 | P@10=0.0900 | R@10=0.9000 | nDCG@10=0.8091
```

Current best retrieval setting:

```python
TOP_K = 3
```

Reason:

```text
Recall@3 is strong, and increasing to K=5 adds little recall but more noise.
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

The pipeline:

```text
question
→ retrieve top 3 chunks
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

Example output:

```text
========== RAGAS GENERATION EVAL ==========
faithfulness: 0.8571
hallucination_rate: 0.1429
```

Interpretation:

```text
85.71% of generated claims were judged supported by retrieved context.
14.29% were potentially unsupported or hallucinated.
```

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

This tells us whether FAISS is finding the right chunks.

### Generation Evaluation

Generation is tested with Ragas.

Metrics:

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

Loads FAISS index and retrieves top-k chunks.

### `src/llm.py`

Calls Groq-hosted Llama model.

### `src/rag.py`

Connects retrieval and generation.

### `src/test_retrieval_metrics.py`

Computes classic retrieval metrics.

### `src/test_generation_ragas.py`

Runs Ragas-based generation evaluation.

---

## Current Results

Retrieval:

```text
MRR: 0.78
P@1: 0.70
R@3: 0.85
nDCG@3: 0.7946
```

Generation:

```text
faithfulness: 0.8571
hallucination_rate: 0.1429
```

---

## Next Improvements

Possible next steps:

```text
1. Try smaller chunks: 120 words, 30 overlap
2. Try larger chunks: 250 words, 50 overlap
3. Add reranking
4. Add answer_correctness in Ragas
5. Add refusal tests for unanswerable questions
6. Add CI gate thresholds
```

Suggested CI gates:

```text
MRR >= 0.75
Recall@3 >= 0.80
Faithfulness >= 0.85
Hallucination rate <= 0.15
```

---

## Notes

This is a simple educational RAG system. It is designed to make the full pipeline understandable before adding production features like rerankers, hybrid search, metadata filtering, async serving, or monitoring.
