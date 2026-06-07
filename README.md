# RAG Progression Study: V1 → V2 → V3

This project is a staged study of Retrieval-Augmented Generation systems. The goal was not just to build a working RAG demo, but to progressively improve retrieval quality, generation faithfulness, and hallucination behavior through measurable experiments.

The project evolved through three versions:

```text
V1: FAISS-only semantic retrieval
V2: FAISS + cross-encoder reranking
V3: FAISS + BM25 hybrid retrieval + cross-encoder reranking
```

Each version was evaluated using the same retrieval and generation testsets so the results could be compared fairly.

---

## High-Level Goal

The goal of this study was to understand how production-style RAG systems improve over a simple vector-search baseline.

The core questions were:

```text
1. Is plain vector search enough?
2. Does reranking improve retrieval quality?
3. Does hybrid retrieval improve the candidate pool?
4. Do better retrieved chunks improve generation faithfulness?
5. Can we reduce hallucination by improving retrieval?
```

---

## Dataset

The system uses the Kaggle Simple English Wikipedia dataset.

```bash
kaggle datasets download -d ffatty/plain-text-wikipedia-simpleenglish
```

The data is chunked, embedded, and indexed into FAISS.

The same chunk store is also used by BM25 in V3.

---

## Common Pipeline Components

Across all versions, the project uses:

```text
FAISS                  → vector search
Sentence Transformers  → embeddings
Groq Llama             → answer generation
Ragas                  → generation faithfulness evaluation
Custom IR metrics      → retrieval evaluation
```

The generation model is accessed through Groq.

Example `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
GROQ_JUDGE_MODEL=llama-3.1-8b-instant
```

---

## Evaluation Setup

The study separates evaluation into two layers.

### Retrieval Evaluation

Retrieval is evaluated without an LLM using classic information retrieval metrics:

```text
Precision@K
Recall@K
MRR
nDCG@K
```

These metrics tell us whether the retriever finds the correct chunk and whether it ranks that chunk early.

### Generation Evaluation

Generation is evaluated using Ragas.

The main enabled metric is:

```text
faithfulness
```

Faithfulness measures whether the generated answer is supported by the retrieved context.

Hallucination rate is computed as:

```text
hallucination_rate = 1 - faithfulness
```

---

# V1: FAISS-Only RAG

## Architecture

```text
User Query
   ↓
Query Embedding
   ↓
FAISS Vector Search
   ↓
Top-3 Chunks
   ↓
RAG Prompt
   ↓
Groq Llama
   ↓
Answer
```

## Description

V1 established the baseline RAG system.

It used FAISS to retrieve the top matching chunks directly from vector similarity. The top retrieved chunks were sent to the Llama model as context.

## V1 Retrieval Results

```text
========== AVERAGE METRICS ==========
MRR: 0.78
K=1 | P@1=0.7000 | R@1=0.7000 | nDCG@1=0.7000
K=3 | P@3=0.2833 | R@3=0.8500 | nDCG@3=0.7946
K=5 | P@5=0.1700 | R@5=0.8500 | nDCG@5=0.7946
K=10 | P@10=0.0900 | R@10=0.9000 | nDCG@10=0.8091
```

## V1 Generation Results

```text
========== RAGAS GENERATION EVAL ==========
faithfulness: 0.8579
hallucination_rate: 0.1421
```

## V1 Takeaway

FAISS worked well enough to build a baseline system, but the ranking quality was limited.

The correct chunk was often present in the retrieved results, but not always ranked first.

This showed that vector search alone was not enough for strong retrieval precision.

---

# V2: FAISS + Cross-Encoder Reranking

## Architecture

```text
User Query
   ↓
Query Embedding
   ↓
FAISS Vector Search
   ↓
Top-20 Candidate Chunks
   ↓
Cross-Encoder Reranker
   ↓
Top-3 Reranked Chunks
   ↓
RAG Prompt
   ↓
Groq Llama
   ↓
Answer
```

## Description

V2 added a cross-encoder reranker after FAISS.

Instead of sending FAISS top-3 directly to the LLM, the system first retrieved a larger candidate set from FAISS and then reranked those candidates.

The cross-encoder scores the query and chunk together:

```text
(query, chunk) → relevance score
```

This is slower than vector search but much more precise.

## V2 Retrieval Results

```text
========== AVERAGE METRICS ==========
MRR: 0.95
K=1 | P@1=0.9500 | R@1=0.9500 | nDCG@1=0.9500
K=3 | P@3=0.3167 | R@3=0.9500 | nDCG@3=0.9500
K=5 | P@5=0.1900 | R@5=0.9500 | nDCG@5=0.9500
K=10 | P@10=0.0950 | R@10=0.9500 | nDCG@10=0.9500
```

## V2 Generation Results

```text
========== RAGAS GENERATION EVAL ==========
faithfulness: 0.8667
hallucination_rate: 0.1333
```

## V2 Improvement Over V1

```text
MRR:        0.78   → 0.95
P@1:        0.70   → 0.95
R@3:        0.85   → 0.95
nDCG@3:     0.7946 → 0.95

Faithfulness:       0.8579 → 0.8667
Hallucination rate: 0.1421 → 0.1333
```

## V2 Takeaway

Reranking significantly improved retrieval quality.

FAISS was already retrieving useful candidates, but the reranker was much better at putting the correct chunk at the top.

This proved that retrieval has two separate problems:

```text
1. Candidate generation
2. Candidate ranking
```

FAISS was good at candidate generation. The cross-encoder was better at final ranking.

---

# V3: FAISS + BM25 + Cross-Encoder Reranking

## Architecture

```text
User Query
   ↓
FAISS Semantic Search
   ↓
BM25 Keyword Search
   ↓
Merge + Deduplicate Candidates
   ↓
Cross-Encoder Reranker
   ↓
Top-3 Reranked Chunks
   ↓
RAG Prompt
   ↓
Groq Llama
   ↓
Answer
```

## Description

V3 added BM25 keyword retrieval alongside FAISS.

The system now retrieves candidates from both:

```text
FAISS → semantic similarity
BM25  → keyword / exact-match similarity
```

The candidate sets are merged and deduplicated by chunk ID, then reranked with the same cross-encoder used in V2.

## Why BM25 Helped

FAISS is strong for semantic similarity, but it can miss exact lexical matches.

BM25 helps with:

```text
names
dates
acronyms
titles
rare entities
exact phrases
```

Examples:

```text
People's Republic of China
U.S. 1st Infantry Division
Maxwell's equations
Serial ATA
James A. Garfield
```

## V3 Retrieval Results

```text
========== AVERAGE METRICS ==========
MRR: 1.0
K=1 | P@1=1.0000 | R@1=1.0000 | nDCG@1=1.0000
K=3 | P@3=0.3333 | R@3=1.0000 | nDCG@3=1.0000
K=5 | P@5=0.2000 | R@5=1.0000 | nDCG@5=1.0000
K=10 | P@10=0.1000 | R@10=1.0000 | nDCG@10=1.0000
```

## V3 Generation Results

```text
========== RAGAS GENERATION EVAL ==========
faithfulness: 0.9292
hallucination_rate: 0.0708
```

## V3 Improvement Over V2

```text
MRR:        0.95 → 1.00
P@1:        0.95 → 1.00
R@3:        0.95 → 1.00
nDCG@3:     0.95 → 1.00

Faithfulness:       0.8667 → 0.9292
Hallucination rate: 0.1333 → 0.0708
```

## V3 Takeaway

V3 showed that improving the candidate pool improves the entire RAG system.

BM25 found exact-match candidates that FAISS could miss or rank lower. The reranker then selected the best chunks from the combined candidate set.

This produced perfect retrieval scores on the current evaluation set and significantly improved generation faithfulness.

---

# Overall Results

## Retrieval Progression

```text
Metric      V1       V2       V3
----------------------------------
MRR         0.78     0.95     1.00
P@1         0.70     0.95     1.00
R@3         0.85     0.95     1.00
nDCG@3      0.7946   0.95     1.00
```

## Generation Progression

```text
Metric                V1       V2       V3
--------------------------------------------
Faithfulness          0.8579   0.8667   0.9292
Hallucination Rate    0.1421   0.1333   0.0708
```

---

# Main Lessons

## 1. FAISS alone is a good baseline, not a full retrieval solution

FAISS gave useful semantic matches, but ranking quality was not perfect.

## 2. Reranking is a major upgrade

The cross-encoder reranker dramatically improved MRR and Precision@1.

## 3. Hybrid retrieval improves candidate quality

BM25 added lexical matching strength, especially for entities, dates, acronyms, and exact phrases.

## 4. Better retrieval improves generation

Generation faithfulness improved as retrieval quality improved.

This supports the core RAG principle:

```text
Better context → better answers → fewer hallucinations
```

## 5. Evaluation must drive architecture

Each system improvement was validated using the same evaluation sets. This made the improvement measurable instead of subjective.

---

# Current Best Architecture

```text
Query
   ↓
FAISS top 20
   ↓
BM25 top 20
   ↓
Merge + deduplicate
   ↓
Cross-encoder rerank
   ↓
Top 3 chunks
   ↓
Groq Llama
   ↓
Answer with source chunk IDs
```

---

# Current Limitations

The current evaluation set is small and mostly contains direct, answerable questions.

It does not yet test:

```text
unanswerable questions
ambiguous questions
indirect references
conversational follow-ups
prompt injection
citation correctness
latency
cost
large-scale robustness
```

Because of this, V3’s perfect retrieval score should be interpreted as perfect on the current benchmark, not proof of production readiness.

---

# Why Query Rewriting Was Not Added Yet

Query rewriting is useful when users ask ambiguous or conversational questions.

Example:

```text
Question: "When was it founded?"
Context: "We were discussing the People's Republic of China."
Rewritten Query: "When was the People's Republic of China founded?"
```

The current evaluation set does not contain these kinds of indirect speech or conversational follow-up examples.

So adding query rewriting now would not be meaningful. The correct next step would be to first create a harder V4 evaluation set with ambiguous, indirect, and conversational queries.

---

# Recommended Next Study

The next study should focus on robustness rather than retrieval tuning.

Recommended V4:

```text
Conversational and Robust RAG Evaluation
```

Test categories:

```text
1. Unanswerable questions
2. Ambiguous questions
3. Conversational follow-up questions
4. Indirect reference questions
5. Prompt-injection attempts
6. Citation correctness
7. Refusal accuracy
```

Only after that should features like query rewriting, clarification detection, and guardrails be added.

---

# Suggested CI Gates

For the current V3 benchmark:

```text
MRR >= 0.95
Recall@3 >= 0.95
nDCG@3 >= 0.95
Faithfulness >= 0.90
Hallucination rate <= 0.10
```

These gates are intentionally strict because V3 performs strongly on the current evaluation set.

---

# Final Summary

This project demonstrates a realistic progression from a simple RAG baseline to a stronger production-style retrieval pipeline.

```text
V1 proved that FAISS can power a basic RAG system.
V2 proved that reranking greatly improves ranking quality.
V3 proved that hybrid retrieval improves candidate quality and reduces hallucination.
```

The final V3 system uses:

```text
FAISS + BM25 + Cross-Encoder Reranking + Groq Llama
```

and achieves:

```text
MRR: 1.00
P@1: 1.00
Recall@3: 1.00
nDCG@3: 1.00
Faithfulness: 0.9292
Hallucination Rate: 0.0708
```

The key engineering lesson is that strong RAG is not just vector search. Strong RAG is an evaluated retrieval and generation system built from semantic search, lexical search, reranking, grounded prompting, and continuous measurement.
