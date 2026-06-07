# We need json to read the labeled retrieval testset
import json

# We need math for log-based nDCG calculation
import math

# We need os to check if the testset exists
import os

# We import your retriever function
from retrieve import retrieve
from retrieve_reranked import retrieve_reranked


# Labeled eval file
TESTSET_PATH = "evals/retrieval_testset.json"

# K values we want to evaluate
K_VALUES = [1, 3, 5, 10]


# This loads the labeled retrieval testset
def load_testset(path):
    # Fail clearly if the testset does not exist
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing testset file: {path}")

    # Open the JSON file
    with open(path, "r", encoding="utf-8") as f:
        # Return parsed JSON data
        return json.load(f)


# Precision@K = relevant retrieved items in top K / K
def precision_at_k(retrieved_ids, relevant_ids, k):
    # Keep only top K retrieved ids
    retrieved_at_k = retrieved_ids[:k]

    # Convert relevant ids to set for fast lookup
    relevant_set = set(relevant_ids)

    # Count how many retrieved ids are relevant
    hits = sum(1 for chunk_id in retrieved_at_k if chunk_id in relevant_set)

    # Divide hits by K
    return hits / k


# Recall@K = relevant retrieved items in top K / total relevant items
def recall_at_k(retrieved_ids, relevant_ids, k):
    # If no relevant ids are labeled, recall is undefined, so return 0
    if not relevant_ids:
        return 0.0

    # Keep only top K retrieved ids
    retrieved_at_k = retrieved_ids[:k]

    # Convert relevant ids to set for fast lookup
    relevant_set = set(relevant_ids)

    # Count how many relevant ids were retrieved
    hits = sum(1 for chunk_id in retrieved_at_k if chunk_id in relevant_set)

    # Divide by total number of relevant ids
    return hits / len(relevant_set)


# MRR = 1 / rank of the first relevant result
def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    # Convert relevant ids to set for fast lookup
    relevant_set = set(relevant_ids)

    # Check each retrieved id in ranked order
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        # First relevant result decides reciprocal rank
        if chunk_id in relevant_set:
            return 1 / rank

    # If no relevant result was found
    return 0.0


# DCG@K rewards relevant documents appearing earlier
def dcg_at_k(retrieved_ids, relevant_ids, k):
    # Convert relevant ids to set for fast lookup
    relevant_set = set(relevant_ids)

    # Initialize DCG score
    dcg = 0.0

    # Loop through top K retrieved results
    for i, chunk_id in enumerate(retrieved_ids[:k]):
        # Binary relevance: 1 if relevant, else 0
        relevance = 1 if chunk_id in relevant_set else 0

        # Rank position is 1-based
        rank = i + 1

        # Add discounted gain
        dcg += relevance / math.log2(rank + 1)

    # Return DCG score
    return dcg


# Ideal DCG@K assumes all relevant items appear at the top
def ideal_dcg_at_k(relevant_ids, k):
    # Number of possible relevant hits within top K
    ideal_hits = min(len(relevant_ids), k)

    # Initialize ideal DCG
    idcg = 0.0

    # Add perfect relevance scores at the top ranks
    for i in range(ideal_hits):
        # Rank position is 1-based
        rank = i + 1

        # Relevance is always 1 in ideal ranking
        idcg += 1 / math.log2(rank + 1)

    # Return ideal DCG
    return idcg


# nDCG@K = DCG@K / Ideal DCG@K
def ndcg_at_k(retrieved_ids, relevant_ids, k):
    # Compute actual DCG
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)

    # Compute ideal DCG
    idcg = ideal_dcg_at_k(relevant_ids, k)

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    # Normalize DCG
    return dcg / idcg


# This evaluates one query
def evaluate_query(case):
    # Extract query text
    query = case["query"]

    # Extract manually labeled relevant chunk ids
    relevant_ids = case["relevant_ids"]

    # Retrieve top max(K) chunks
    # retrieved = retrieve(query, top_k=max(K_VALUES))
    retrieved = retrieve_reranked(query, top_k=max(K_VALUES))

    # Extract retrieved ids
    retrieved_ids = [item["id"] for item in retrieved]

    # Extract scores for debugging
    retrieved_scores = [item["score"] for item in retrieved]

    # Store base result
    result = {
        "query": query,
        "relevant_ids": relevant_ids,
        "retrieved_ids": retrieved_ids,
        "retrieved_scores": retrieved_scores,
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
    }

    # Compute metrics for each K
    for k in K_VALUES:
        # Save Precision@K
        result[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)

        # Save Recall@K
        result[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)

        # Save nDCG@K
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    # Return full result
    return result


# Print query-level result
def print_query_result(result):
    # Print query
    print("\nQUERY:", result["query"])

    # Print labeled relevant ids
    print("Relevant IDs:", result["relevant_ids"])

    # Print retrieved ids
    print("Retrieved IDs:", result["retrieved_ids"])

    # Print retrieved scores
    print("Retrieved Scores:", [round(score, 4) for score in result["retrieved_scores"]])

    # Print MRR
    print("MRR:", round(result["mrr"], 4))

    # Print metrics by K
    for k in K_VALUES:
        print(
            f"K={k} | "
            f"P@{k}={result[f'precision@{k}']:.4f} | "
            f"R@{k}={result[f'recall@{k}']:.4f} | "
            f"nDCG@{k}={result[f'ndcg@{k}']:.4f}"
        )


# Compute average of one metric across all rows
def average_metric(rows, metric_name):
    # Avoid division by zero
    if not rows:
        return 0.0

    # Return arithmetic mean
    return sum(row[metric_name] for row in rows) / len(rows)


# Print aggregate metrics
def print_average_metrics(rows):
    # Print header
    print("\n========== AVERAGE METRICS ==========")

    # Print average MRR
    print("MRR:", round(average_metric(rows, "mrr"), 4))

    # Print average metrics for each K
    for k in K_VALUES:
        print(
            f"K={k} | "
            f"P@{k}={average_metric(rows, f'precision@{k}'):.4f} | "
            f"R@{k}={average_metric(rows, f'recall@{k}'):.4f} | "
            f"nDCG@{k}={average_metric(rows, f'ndcg@{k}'):.4f}"
        )


# Main evaluation runner
def main():
    # Load labeled testset
    testset = load_testset(TESTSET_PATH)

    # Store all query-level results in memory only
    rows = []

    # Evaluate each test case
    for index, case in enumerate(testset, start=1):
        # Print progress
        print(f"\n[{index}/{len(testset)}] Evaluating retrieval")

        # Compute metrics for one query
        result = evaluate_query(case)

        # Store result in memory
        rows.append(result)

        # Print query-level result
        print_query_result(result)

    # Print aggregate metrics
    print_average_metrics(rows)


# Run only when this file is executed directly
if __name__ == "__main__":
    main()