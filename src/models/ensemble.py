"""Model ensemble and re-ranking components."""

from typing import Optional, Callable

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix

from src.models.base import BaseRecommender


class EnsembleRecommender(BaseRecommender):
    """
    Ensemble of multiple recommendation models.

    Supports multiple combination strategies:
    - weighted_average: Weighted average of normalized scores
    - rank_average: Average of ranks across models
    - voting: Majority voting for top-k items
    - stacking: Learn optimal weights from validation data
    """

    def __init__(
        self,
        models: list[BaseRecommender],
        weights: Optional[list[float]] = None,
        name: str = "ensemble",
        strategy: str = "weighted_average",
    ) -> None:
        """
        Initialize Ensemble recommender.

        Args:
            models: List of fitted recommendation models
            weights: Weights for each model (default: equal weights)
            name: Model name
            strategy: Combination strategy
        """
        super().__init__(name=name)
        self.models = models
        self.strategy = strategy

        # Set weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            total = sum(weights)
            self.weights = [w / total for w in weights]

        self._interaction_matrix: Optional[csr_matrix] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        **kwargs,
    ) -> "EnsembleRecommender":
        """
        Ensemble assumes all sub-models are already fitted.

        Args:
            interaction_matrix: User-item interaction matrix

        Returns:
            self
        """
        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        # Verify all models are fitted
        for i, model in enumerate(self.models):
            if not model.is_fitted:
                raise ValueError(f"Model {i} ({model.name}) is not fitted")

        self.is_fitted = True
        logger.info(
            f"Ensemble ready with {len(self.models)} models, "
            f"strategy={self.strategy}"
        )
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """Generate ensemble recommendations."""
        self._check_fitted()

        if self._interaction_matrix is None:
            raise RuntimeError("Model not properly fitted")

        # Determine items to exclude
        if seen_items is None and exclude_seen:
            seen_items = self._get_seen_items(user_idx, self._interaction_matrix)
        elif seen_items is None:
            seen_items = set()

        if self.strategy == "weighted_average":
            return self._weighted_average(user_idx, n, seen_items)
        elif self.strategy == "rank_average":
            return self._rank_average(user_idx, n, seen_items)
        elif self.strategy == "voting":
            return self._voting(user_idx, n, seen_items)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_average(
        self,
        user_idx: int,
        n: int,
        seen_items: set[int],
    ) -> list[tuple[int, float]]:
        """Weighted average of normalized scores."""
        n_candidates = min(n * 3, self.n_items)

        # Collect scores from all models
        all_scores: dict[int, float] = {}

        for model, weight in zip(self.models, self.weights):
            recs = model.recommend(
                user_idx, n=n_candidates, exclude_seen=True, seen_items=seen_items
            )

            if not recs:
                continue

            # Normalize scores to [0, 1]
            scores = {item_idx: score for item_idx, score in recs}
            scores = self._normalize_scores(scores)

            # Add weighted scores
            for item_idx, score in scores.items():
                if item_idx not in all_scores:
                    all_scores[item_idx] = 0.0
                all_scores[item_idx] += weight * score

        # Sort and return top-n
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in sorted_items[:n]]

    def _rank_average(
        self,
        user_idx: int,
        n: int,
        seen_items: set[int],
    ) -> list[tuple[int, float]]:
        """Average of ranks across models."""
        n_candidates = min(n * 3, self.n_items)

        # Collect ranks from all models
        all_ranks: dict[int, list[int]] = {}

        for model, weight in zip(self.models, self.weights):
            recs = model.recommend(
                user_idx, n=n_candidates, exclude_seen=True, seen_items=seen_items
            )

            for rank, (item_idx, _) in enumerate(recs, start=1):
                if item_idx not in all_ranks:
                    all_ranks[item_idx] = []
                # Weight the rank contribution
                all_ranks[item_idx].append(rank * (1.0 / weight))

        # Compute average rank (lower is better)
        avg_ranks = {
            item_idx: np.mean(ranks)
            for item_idx, ranks in all_ranks.items()
        }

        # Sort by rank (ascending) and return
        sorted_items = sorted(avg_ranks.items(), key=lambda x: x[1])
        return [
            (int(idx), float(1.0 / rank))  # Convert to score (higher is better)
            for idx, rank in sorted_items[:n]
        ]

    def _voting(
        self,
        user_idx: int,
        n: int,
        seen_items: set[int],
    ) -> list[tuple[int, float]]:
        """Majority voting for top items."""
        n_candidates = min(n * 2, self.n_items)

        # Count votes for each item
        votes: dict[int, float] = {}

        for model, weight in zip(self.models, self.weights):
            recs = model.recommend(
                user_idx, n=n_candidates, exclude_seen=True, seen_items=seen_items
            )

            # Items in top-n get weighted votes
            for rank, (item_idx, _) in enumerate(recs[:n]):
                if item_idx not in votes:
                    votes[item_idx] = 0.0
                # Weighted vote with position decay
                votes[item_idx] += weight * (1.0 / (rank + 1))

        # Sort by votes and return
        sorted_items = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in sorted_items[:n]]

    def _normalize_scores(self, scores: dict[int, float]) -> dict[int, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return {}

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return {k: 0.5 for k in scores}

        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }

    def get_similar_items(
        self,
        item_idx: int,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """Get similar items by averaging across models."""
        self._check_fitted()

        all_scores: dict[int, float] = {}

        for model, weight in zip(self.models, self.weights):
            if not hasattr(model, "get_similar_items"):
                continue

            similar = model.get_similar_items(item_idx, n=n * 2)
            scores = {idx: score for idx, score in similar}
            scores = self._normalize_scores(scores)

            for idx, score in scores.items():
                if idx not in all_scores:
                    all_scores[idx] = 0.0
                all_scores[idx] += weight * score

        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in sorted_items[:n]]


class ReRanker:
    """
    Re-ranking layer for post-processing recommendations.

    Applies business rules, diversity injection, and other adjustments
    to the initial recommendation list.
    """

    def __init__(
        self,
        diversity_weight: float = 0.0,
        freshness_weight: float = 0.0,
        category_diversity: bool = False,
    ) -> None:
        """
        Initialize re-ranker.

        Args:
            diversity_weight: Weight for diversity (0 = no diversity boost)
            freshness_weight: Weight for item freshness/recency
            category_diversity: Enforce category diversity in results
        """
        self.diversity_weight = diversity_weight
        self.freshness_weight = freshness_weight
        self.category_diversity = category_diversity

    def rerank(
        self,
        recommendations: list[tuple[int, float]],
        item_features: Optional[dict] = None,
        item_similarity: Optional[np.ndarray] = None,
        item_timestamps: Optional[np.ndarray] = None,
        n: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """
        Re-rank recommendations.

        Args:
            recommendations: List of (item_idx, score) tuples
            item_features: Dict with item features (categories, etc.)
            item_similarity: Item-item similarity matrix (for diversity)
            item_timestamps: Item creation timestamps (for freshness)
            n: Number of items to return (default: same as input)

        Returns:
            Re-ranked list of (item_idx, score) tuples
        """
        if n is None:
            n = len(recommendations)

        if not recommendations:
            return []

        # Start with original scores
        items = [item_idx for item_idx, _ in recommendations]
        scores = np.array([score for _, score in recommendations])

        # Normalize scores
        if scores.max() != scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.ones_like(scores) * 0.5

        # Apply freshness boost
        if self.freshness_weight > 0 and item_timestamps is not None:
            freshness_scores = self._compute_freshness(items, item_timestamps)
            scores = (1 - self.freshness_weight) * scores + self.freshness_weight * freshness_scores

        # Apply diversity-aware re-ranking (MMR-style)
        if self.diversity_weight > 0 and item_similarity is not None:
            return self._mmr_rerank(items, scores, item_similarity, n)

        # Apply category diversity
        if self.category_diversity and item_features is not None:
            return self._category_diverse_rerank(items, scores, item_features, n)

        # Return top-n by score
        top_indices = np.argsort(scores)[-n:][::-1]
        return [(items[i], float(scores[i])) for i in top_indices]

    def _compute_freshness(
        self,
        items: list[int],
        timestamps: np.ndarray,
    ) -> np.ndarray:
        """Compute freshness scores (newer = higher)."""
        item_times = timestamps[items]

        if item_times.max() == item_times.min():
            return np.ones(len(items)) * 0.5

        # Normalize to [0, 1] where newer = higher
        freshness = (item_times - item_times.min()) / (item_times.max() - item_times.min())
        return freshness

    def _mmr_rerank(
        self,
        items: list[int],
        scores: np.ndarray,
        similarity: np.ndarray,
        n: int,
    ) -> list[tuple[int, float]]:
        """
        Maximal Marginal Relevance re-ranking.

        Balances relevance with diversity by penalizing items
        similar to already selected items.
        """
        selected = []
        selected_indices = []
        remaining = list(range(len(items)))

        lambda_param = 1 - self.diversity_weight

        while len(selected) < n and remaining:
            best_idx = None
            best_score = -np.inf

            for idx in remaining:
                item_idx = items[idx]
                relevance = scores[idx]

                # Compute max similarity to selected items
                if selected_indices:
                    selected_items = [items[i] for i in selected_indices]
                    max_sim = max(similarity[item_idx, s] for s in selected_items)
                else:
                    max_sim = 0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append((items[best_idx], float(scores[best_idx])))
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return selected

    def _category_diverse_rerank(
        self,
        items: list[int],
        scores: np.ndarray,
        features: dict,
        n: int,
    ) -> list[tuple[int, float]]:
        """
        Re-rank to ensure category diversity.

        Uses round-robin selection across categories.
        """
        # Get categories for items
        categories = features.get("categories", {})
        item_categories = [categories.get(item_idx, "unknown") for item_idx in items]

        # Group by category
        category_items: dict[str, list[tuple[int, int, float]]] = {}
        for idx, (item_idx, category) in enumerate(zip(items, item_categories)):
            if category not in category_items:
                category_items[category] = []
            category_items[category].append((idx, item_idx, scores[idx]))

        # Sort items within each category by score
        for category in category_items:
            category_items[category].sort(key=lambda x: x[2], reverse=True)

        # Round-robin selection
        selected = []
        category_pointers = {cat: 0 for cat in category_items}
        categories_list = list(category_items.keys())

        while len(selected) < n:
            made_selection = False

            for category in categories_list:
                if len(selected) >= n:
                    break

                pointer = category_pointers[category]
                if pointer < len(category_items[category]):
                    _, item_idx, score = category_items[category][pointer]
                    selected.append((item_idx, float(score)))
                    category_pointers[category] += 1
                    made_selection = True

            if not made_selection:
                break

        return selected


class BusinessRulesFilter:
    """
    Filter recommendations based on business rules.

    Handles inventory, eligibility, and other business constraints.
    """

    def __init__(self) -> None:
        self.rules: list[Callable] = []

    def add_rule(self, rule_fn: Callable[[int, dict], bool]) -> "BusinessRulesFilter":
        """
        Add a business rule.

        Args:
            rule_fn: Function that takes (item_idx, context) and returns True if item is allowed

        Returns:
            self
        """
        self.rules.append(rule_fn)
        return self

    def filter(
        self,
        recommendations: list[tuple[int, float]],
        context: Optional[dict] = None,
    ) -> list[tuple[int, float]]:
        """
        Filter recommendations based on business rules.

        Args:
            recommendations: List of (item_idx, score) tuples
            context: Context dict passed to rule functions

        Returns:
            Filtered recommendations
        """
        if context is None:
            context = {}

        filtered = []
        for item_idx, score in recommendations:
            passes_all = all(rule(item_idx, context) for rule in self.rules)
            if passes_all:
                filtered.append((item_idx, score))

        return filtered


# Pre-built business rules
def in_stock_rule(inventory: dict[int, int]) -> Callable:
    """Rule: Item must be in stock."""
    def rule(item_idx: int, context: dict) -> bool:
        return inventory.get(item_idx, 0) > 0
    return rule


def price_range_rule(
    prices: dict[int, float],
    min_price: float = 0,
    max_price: float = float("inf"),
) -> Callable:
    """Rule: Item price must be in range."""
    def rule(item_idx: int, context: dict) -> bool:
        price = prices.get(item_idx, 0)
        return min_price <= price <= max_price
    return rule


def category_filter_rule(
    categories: dict[int, str],
    allowed_categories: set[str],
) -> Callable:
    """Rule: Item must be in allowed categories."""
    def rule(item_idx: int, context: dict) -> bool:
        return categories.get(item_idx, "") in allowed_categories
    return rule


def exclude_items_rule(excluded_items: set[int]) -> Callable:
    """Rule: Exclude specific items."""
    def rule(item_idx: int, context: dict) -> bool:
        return item_idx not in excluded_items
    return rule
