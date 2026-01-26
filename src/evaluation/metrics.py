"""Evaluation metrics for recommendation systems."""

from typing import Optional

import numpy as np
from loguru import logger


def precision_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """
    Precision@K: What fraction of top-K recommendations are relevant?

    Args:
        recommended: List of recommended item indices (in order)
        relevant: Set of relevant (ground truth) item indices
        k: Number of top recommendations to consider

    Returns:
        Precision score between 0 and 1
    """
    if k <= 0:
        return 0.0

    top_k = recommended[:k]
    if len(top_k) == 0:
        return 0.0

    n_relevant = len(set(top_k) & relevant)
    return n_relevant / len(top_k)


def recall_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """
    Recall@K: What fraction of relevant items appear in top-K?

    Args:
        recommended: List of recommended item indices (in order)
        relevant: Set of relevant (ground truth) item indices
        k: Number of top recommendations to consider

    Returns:
        Recall score between 0 and 1
    """
    if len(relevant) == 0:
        return 0.0

    top_k = recommended[:k]
    n_relevant = len(set(top_k) & relevant)
    return n_relevant / len(relevant)


def ndcg_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain@K.

    Measures ranking quality, giving higher scores when relevant items
    appear earlier in the recommendation list.

    Args:
        recommended: List of recommended item indices (in order)
        relevant: Set of relevant (ground truth) item indices
        k: Number of top recommendations to consider

    Returns:
        NDCG score between 0 and 1
    """
    if k <= 0 or len(relevant) == 0:
        return 0.0

    top_k = recommended[:k]

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG: if we had perfect ranking
    ideal_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_k + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """
    Hit Rate@K: Is there at least one relevant item in top-K?

    Args:
        recommended: List of recommended item indices (in order)
        relevant: Set of relevant (ground truth) item indices
        k: Number of top recommendations to consider

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    top_k = set(recommended[:k])
    return 1.0 if len(top_k & relevant) > 0 else 0.0


def mean_reciprocal_rank(
    recommended: list[int],
    relevant: set[int],
) -> float:
    """
    Mean Reciprocal Rank: 1 / rank of first relevant item.

    Args:
        recommended: List of recommended item indices (in order)
        relevant: Set of relevant (ground truth) item indices

    Returns:
        MRR score between 0 and 1
    """
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(
    recommended: list[int],
    relevant: set[int],
) -> float:
    """
    Average Precision: Mean of precision at each relevant item position.

    Args:
        recommended: List of recommended item indices (in order)
        relevant: Set of relevant (ground truth) item indices

    Returns:
        AP score between 0 and 1
    """
    if len(relevant) == 0:
        return 0.0

    precisions = []
    n_relevant_so_far = 0

    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            n_relevant_so_far += 1
            precisions.append(n_relevant_so_far / rank)

    if len(precisions) == 0:
        return 0.0

    return sum(precisions) / len(relevant)


def coverage(
    all_recommendations: list[list[int]],
    n_items: int,
) -> float:
    """
    Catalog Coverage: Fraction of items recommended at least once.

    Args:
        all_recommendations: List of recommendation lists (one per user)
        n_items: Total number of items in catalog

    Returns:
        Coverage score between 0 and 1
    """
    if n_items == 0:
        return 0.0

    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)

    return len(recommended_items) / n_items


def diversity(
    recommendations: list[int],
    item_similarity: np.ndarray,
) -> float:
    """
    Intra-List Diversity: Average dissimilarity between recommended items.

    Args:
        recommendations: List of recommended item indices
        item_similarity: Item-item similarity matrix

    Returns:
        Diversity score (higher = more diverse)
    """
    if len(recommendations) < 2:
        return 0.0

    total_dissimilarity = 0.0
    n_pairs = 0

    for i, item_i in enumerate(recommendations):
        for item_j in recommendations[i + 1:]:
            # Dissimilarity = 1 - similarity
            total_dissimilarity += 1.0 - item_similarity[item_i, item_j]
            n_pairs += 1

    if n_pairs == 0:
        return 0.0

    return total_dissimilarity / n_pairs


def novelty(
    recommendations: list[int],
    item_popularity: np.ndarray,
    n_users: int,
) -> float:
    """
    Novelty: How unexpected are the recommendations?

    Based on self-information: -log2(popularity)

    Args:
        recommendations: List of recommended item indices
        item_popularity: Array of interaction counts per item
        n_users: Total number of users

    Returns:
        Novelty score (higher = more novel/unexpected)
    """
    if len(recommendations) == 0 or n_users == 0:
        return 0.0

    # Convert to probabilities
    probs = item_popularity / n_users
    probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)

    total_novelty = 0.0
    for item in recommendations:
        total_novelty += -np.log2(probs[item])

    return total_novelty / len(recommendations)


def evaluate_model(
    model,
    test_interactions: dict[int, set[int]],
    k_values: list[int] = [5, 10, 20],
    n_items: Optional[int] = None,
) -> dict[str, float]:
    """
    Comprehensive evaluation of a recommendation model.

    Args:
        model: Fitted recommender model with recommend() method
        test_interactions: Dict mapping user_idx to set of relevant item indices
        k_values: List of K values for @K metrics
        n_items: Total items for coverage calculation

    Returns:
        Dict of metric names to values
    """
    results = {}

    # Initialize accumulators
    for k in k_values:
        results[f"precision@{k}"] = []
        results[f"recall@{k}"] = []
        results[f"ndcg@{k}"] = []
        results[f"hit_rate@{k}"] = []

    results["mrr"] = []
    results["map"] = []

    all_recommendations = []

    # Evaluate each user
    n_users = len(test_interactions)
    logger.info(f"Evaluating model on {n_users} users...")

    for user_idx, relevant_items in test_interactions.items():
        if len(relevant_items) == 0:
            continue

        # Get recommendations (enough for max k)
        max_k = max(k_values)
        try:
            recs = model.recommend(user_idx, n=max_k, exclude_seen=True)
            rec_items = [item_idx for item_idx, _ in recs]
        except Exception as e:
            logger.warning(f"Failed to get recommendations for user {user_idx}: {e}")
            continue

        all_recommendations.append(rec_items)

        # Compute metrics for each k
        for k in k_values:
            results[f"precision@{k}"].append(
                precision_at_k(rec_items, relevant_items, k)
            )
            results[f"recall@{k}"].append(
                recall_at_k(rec_items, relevant_items, k)
            )
            results[f"ndcg@{k}"].append(
                ndcg_at_k(rec_items, relevant_items, k)
            )
            results[f"hit_rate@{k}"].append(
                hit_rate_at_k(rec_items, relevant_items, k)
            )

        results["mrr"].append(mean_reciprocal_rank(rec_items, relevant_items))
        results["map"].append(average_precision(rec_items, relevant_items))

    # Average all metrics
    final_results = {}
    for metric, values in results.items():
        if len(values) > 0:
            final_results[metric] = float(np.mean(values))
        else:
            final_results[metric] = 0.0

    # Add coverage if n_items provided
    if n_items is not None and len(all_recommendations) > 0:
        final_results["coverage"] = coverage(all_recommendations, n_items)

    return final_results


def compare_models(
    models: list,
    test_interactions: dict[int, set[int]],
    k_values: list[int] = [5, 10, 20],
    n_items: Optional[int] = None,
) -> dict[str, dict[str, float]]:
    """
    Compare multiple models on the same test set.

    Args:
        models: List of fitted recommender models
        test_interactions: Dict mapping user_idx to relevant items
        k_values: List of K values for metrics
        n_items: Total items for coverage

    Returns:
        Dict mapping model name to metrics dict
    """
    results = {}

    for model in models:
        logger.info(f"Evaluating {model.name}...")
        results[model.name] = evaluate_model(
            model,
            test_interactions,
            k_values,
            n_items,
        )

    return results


def print_evaluation_results(
    results: dict[str, float],
    model_name: str = "Model",
) -> None:
    """Pretty print evaluation results."""
    print(f"\n{'=' * 50}")
    print(f" {model_name} Evaluation Results")
    print(f"{'=' * 50}")

    # Group by metric type
    precision_metrics = {k: v for k, v in results.items() if k.startswith("precision")}
    recall_metrics = {k: v for k, v in results.items() if k.startswith("recall")}
    ndcg_metrics = {k: v for k, v in results.items() if k.startswith("ndcg")}
    other_metrics = {
        k: v for k, v in results.items()
        if not any(k.startswith(p) for p in ["precision", "recall", "ndcg", "hit_rate"])
    }

    for name, metrics in [
        ("Precision", precision_metrics),
        ("Recall", recall_metrics),
        ("NDCG", ndcg_metrics),
        ("Other", other_metrics),
    ]:
        if metrics:
            print(f"\n{name}:")
            for metric, value in sorted(metrics.items()):
                print(f"  {metric}: {value:.4f}")

    print(f"\n{'=' * 50}\n")
