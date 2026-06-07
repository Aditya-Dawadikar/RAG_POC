import json
import os
import time

from datasets import Dataset

from ragas import evaluate
from ragas.run_config import RunConfig

from ragas.metrics import (
    faithfulness,
    # answer_correctness,
    # answer_relevancy,
    # context_precision,
    # context_recall,
)

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from rag import answer_question


TESTSET_PATH = "evals/generation_testset.json"

JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Added constraint because token limit of 6k was exhausted
MAX_CONTEXT_CHARS = 1000
MAX_ANSWER_CHARS = 800
EVAL_SLEEP_SECONDS = 20

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing testset file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_testset(testset):
    for i, case in enumerate(testset):
        if "query" not in case:
            raise ValueError(f"Case {i} is missing 'query'")

        if "reference_answer" not in case:
            raise ValueError(f"Case {i} is missing 'reference_answer'")

        if not str(case["reference_answer"]).strip():
            raise ValueError(f"Case {i} has empty 'reference_answer'")


def cache_key(case):
    return case["query"]


def build_rag_outputs(testset):
    rag_outputs = {}

    for index, case in enumerate(testset, start=1):
        key = cache_key(case)

        print(f"[{index}/{len(testset)}] Generating: {case['query']}")

        answer, retrieved_chunks = answer_question(case["query"])

        rag_outputs[key] = {
            "question": case["query"],
            "answer": answer,
            "contexts": [chunk["text"] for chunk in retrieved_chunks],
            "retrieved_ids": [chunk["id"] for chunk in retrieved_chunks],
            "retrieved_scores": [chunk["score"] for chunk in retrieved_chunks],
        }

        print(f"Sleeping {EVAL_SLEEP_SECONDS}s to avoid Groq TPM limit...")
        time.sleep(EVAL_SLEEP_SECONDS)

    return rag_outputs


def build_ragas_rows(testset, rag_outputs):
    rows = []

    for case in testset:
        rag_output = rag_outputs[cache_key(case)]

        rows.append(
            {
                "question": case["query"],
                "answer": rag_output["answer"][:MAX_ANSWER_CHARS],
                "contexts": [
                        context[:MAX_CONTEXT_CHARS]
                        for context in rag_output["contexts"]
                    ],
                "ground_truth": case["reference_answer"],
            }
        )

    return rows


def print_case_debug(testset, rag_outputs, result_df):
    print("\n========== CASE LEVEL RESULTS ==========")

    for i, case in enumerate(testset):
        rag_output = rag_outputs[cache_key(case)]

        print("\n----------------------------------------")
        print(f"Case: {i + 1}")
        print(f"Topic: {case.get('topic', '')}")
        print(f"Query: {case['query']}")
        print(f"Reference Answer: {case['reference_answer']}")
        print(f"Retrieved IDs: {rag_output['retrieved_ids']}")

        if "faithfulness" in result_df.columns:
            faithfulness_score = result_df.iloc[i]["faithfulness"]
            hallucination_rate = 1 - faithfulness_score

            print(f"Faithfulness: {faithfulness_score:.4f}")
            print(f"Hallucination Rate: {hallucination_rate:.4f}")

        print(f"Generated Answer: {rag_output['answer']}")


def print_summary(result_df):
    print("\n========== RAGAS GENERATION EVAL ==========")

    for col in [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]:
        if col in result_df.columns:
            print(f"{col}: {result_df[col].mean():.4f}")

    if "faithfulness" in result_df.columns:
        hallucination_rate = 1 - result_df["faithfulness"].mean()
        print(f"hallucination_rate: {hallucination_rate:.4f}")


def main():
    testset = load_json(TESTSET_PATH)
    validate_testset(testset)

    rag_outputs = build_rag_outputs(testset)

    rows = build_ragas_rows(testset, rag_outputs)
    dataset = Dataset.from_list(rows)

    judge_llm = ChatGroq(
        model=JUDGE_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=4096,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    run_config = RunConfig(
        timeout=600,
        max_retries=3,
        max_wait=120, # Updated from 90->120 for this test
        max_workers=1,
        log_tenacity=True,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
        ],
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    result_df = result.to_pandas()

    print_summary(result_df)
    print_case_debug(testset, rag_outputs, result_df)


if __name__ == "__main__":
    main()