"""Hybrid recommendation models combining multiple signals."""

from typing import Optional

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix

from src.models.base import BaseRecommender


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender combining collaborative filtering with content-based signals.

    Supports multiple combination strategies:
    - weighted: Linear combination of scores
    - switching: Use CF when possible, fallback to content for cold-start
    - cascade: Use content to filter, then rank with CF
    """

    def __init__(
        self,
        cf_model: BaseRecommender,
        content_model: BaseRecommender,
        name: str = "hybrid",
        cf_weight: float = 0.7,
        content_weight: float = 0.3,
        strategy: str = "weighted",
        cold_start_threshold: int = 5,
    ) -> None:
        """
        Initialize Hybrid recommender.

        Args:
            cf_model: Collaborative filtering model
            content_model: Content-based model
            name: Model name
            cf_weight: Weight for CF scores (for weighted strategy)
            content_weight: Weight for content scores (for weighted strategy)
            strategy: Combination strategy ('weighted', 'switching', 'cascade')
            cold_start_threshold: Min interactions before using CF (for switching)
        """
        super().__init__(name=name)
        self.cf_model = cf_model
        self.content_model = content_model
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.strategy = strategy
        self.cold_start_threshold = cold_start_threshold

        self._interaction_matrix: Optional[csr_matrix] = None

        # Validate weights
        if strategy == "weighted":
            total = cf_weight + content_weight
            self.cf_weight = cf_weight / total
            self.content_weight = content_weight / total

    def fit(
        self,
        interaction_matrix: csr_matrix,
        **kwargs,
    ) -> "HybridRecommender":
        """
        Fit is handled by individual models.

        Args:
            interaction_matrix: User-item interaction matrix
            **kwargs: Additional arguments passed to sub-models

        Returns:
            self
        """
        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        # Check if sub-models are fitted
        if not self.cf_model.is_fitted:
            raise ValueError("CF model must be fitted before hybrid model")
        if not self.content_model.is_fitted:
            raise ValueError("Content model must be fitted before hybrid model")

        self.is_fitted = True
        logger.info(
            f"Hybrid model ready: strategy={self.strategy}, "
            f"cf_weight={self.cf_weight:.2f}, content_weight={self.content_weight:.2f}"
        )
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """
        Generate hybrid recommendations.

        Args:
            user_idx: User index
            n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: Specific items to exclude

        Returns:
            List of (item_idx, score) tuples
        """
        self._check_fitted()

        if self._interaction_matrix is None:
            raise RuntimeError("Model not properly fitted")

        # Determine items to exclude
        if seen_items is None and exclude_seen:
            seen_items = self._get_seen_items(user_idx, self._interaction_matrix)
        elif seen_items is None:
            seen_items = set()

        # Get user's interaction count
        user_interactions = self._interaction_matrix[user_idx].nnz

        if self.strategy == "weighted":
            return self._weighted_recommend(user_idx, n, seen_items)
        elif self.strategy == "switching":
            return self._switching_recommend(user_idx, n, seen_items, user_interactions)
        elif self.strategy == "cascade":
            return self._cascade_recommend(user_idx, n, seen_items)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_recommend(
        self,
        user_idx: int,
        n: int,
        seen_items: set[int],
    ) -> list[tuple[int, float]]:
        """Weighted combination of CF and content scores."""
        # Get more candidates than needed for merging
        n_candidates = min(n * 3, self.n_items)

        # Get CF recommendations
        cf_recs = self.cf_model.recommend(
            user_idx, n=n_candidates, exclude_seen=True, seen_items=seen_items
        )
        cf_scores = {item_idx: score for item_idx, score in cf_recs}

        # Get content recommendations
        content_recs = self.content_model.recommend(
            user_idx, n=n_candidates, exclude_seen=True, seen_items=seen_items
        )
        content_scores = {item_idx: score for item_idx, score in content_recs}

        # Normalize scores to [0, 1]
        cf_scores = self._normalize_scores(cf_scores)
        content_scores = self._normalize_scores(content_scores)

        # Combine scores
        all_items = set(cf_scores.keys()) | set(content_scores.keys())
        combined_scores = {}

        for item_idx in all_items:
            cf_score = cf_scores.get(item_idx, 0.0)
            content_score = content_scores.get(item_idx, 0.0)
            combined_scores[item_idx] = (
                self.cf_weight * cf_score + self.content_weight * content_score
            )

        # Sort and return top-n
        sorted_items = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [(int(idx), float(score)) for idx, score in sorted_items[:n]]

    def _switching_recommend(
        self,
        user_idx: int,
        n: int,
        seen_items: set[int],
        user_interactions: int,
    ) -> list[tuple[int, float]]:
        """Use CF for active users, content for cold-start."""
        if user_interactions >= self.cold_start_threshold:
            # User has enough interactions, use CF
            return self.cf_model.recommend(
                user_idx, n=n, exclude_seen=True, seen_items=seen_items
            )
        else:
            # Cold-start user, use content-based
            return self.content_model.recommend(
                user_idx, n=n, exclude_seen=True, seen_items=seen_items
            )

    def _cascade_recommend(
        self,
        user_idx: int,
        n: int,
        seen_items: set[int],
    ) -> list[tuple[int, float]]:
        """Use content to filter candidates, then rank with CF."""
        # First stage: get broad candidates from content model
        n_candidates = min(n * 5, self.n_items)
        content_recs = self.content_model.recommend(
            user_idx, n=n_candidates, exclude_seen=True, seen_items=seen_items
        )
        candidate_items = {item_idx for item_idx, _ in content_recs}

        # Second stage: score candidates with CF model
        cf_recs = self.cf_model.recommend(
            user_idx, n=self.n_items, exclude_seen=True, seen_items=seen_items
        )

        # Filter to only candidates from content model
        filtered_recs = [
            (item_idx, score)
            for item_idx, score in cf_recs
            if item_idx in candidate_items
        ]

        return filtered_recs[:n]

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
        """
        Get similar items combining CF and content similarity.

        Args:
            item_idx: Item index
            n: Number of similar items

        Returns:
            List of (item_idx, similarity) tuples
        """
        self._check_fitted()

        # Get similar items from both models
        n_candidates = min(n * 2, self.n_items)

        cf_similar = {}
        if hasattr(self.cf_model, "get_similar_items"):
            cf_results = self.cf_model.get_similar_items(item_idx, n=n_candidates)
            cf_similar = {idx: score for idx, score in cf_results}

        content_similar = {}
        if hasattr(self.content_model, "get_similar_items"):
            content_results = self.content_model.get_similar_items(item_idx, n=n_candidates)
            content_similar = {idx: score for idx, score in content_results}

        # Normalize and combine
        cf_similar = self._normalize_scores(cf_similar)
        content_similar = self._normalize_scores(content_similar)

        all_items = set(cf_similar.keys()) | set(content_similar.keys())
        combined = {}

        for idx in all_items:
            cf_score = cf_similar.get(idx, 0.0)
            content_score = content_similar.get(idx, 0.0)
            combined[idx] = self.cf_weight * cf_score + self.content_weight * content_score

        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in sorted_items[:n]]


class FeatureAugmentedCF(BaseRecommender):
    """
    Collaborative filtering augmented with item features.

    Similar to LightFM's approach but simpler:
    - Learns user and item embeddings from interactions
    - Augments item embeddings with content features
    """

    def __init__(
        self,
        name: str = "feature_augmented_cf",
        n_factors: int = 64,
        feature_weight: float = 0.3,
    ) -> None:
        """
        Initialize Feature-Augmented CF.

        Args:
            name: Model name
            n_factors: Number of latent factors
            feature_weight: Weight for feature-based component
        """
        super().__init__(name=name)
        self.n_factors = n_factors
        self.feature_weight = feature_weight

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.item_feature_factors: Optional[np.ndarray] = None
        self._interaction_matrix: Optional[csr_matrix] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        item_features: Optional[np.ndarray] = None,
        n_iterations: int = 10,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        **kwargs,
    ) -> "FeatureAugmentedCF":
        """
        Train the model using SGD.

        Args:
            interaction_matrix: User-item interaction matrix
            item_features: Item feature matrix (n_items x n_features)
            n_iterations: Number of training iterations
            learning_rate: Learning rate
            regularization: L2 regularization strength

        Returns:
            self
        """
        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        logger.info(
            f"Training Feature-Augmented CF: {self.n_factors} factors, "
            f"{n_iterations} iterations"
        )

        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))

        # Initialize feature factors if features provided
        if item_features is not None:
            n_features = item_features.shape[1]
            self.item_feature_factors = np.random.normal(
                0, 0.1, (n_features, self.n_factors)
            )
            # Compute feature-based item embeddings
            feature_embeddings = item_features @ self.item_feature_factors
        else:
            feature_embeddings = np.zeros((self.n_items, self.n_factors))

        # Get non-zero interactions
        users, items = interaction_matrix.nonzero()
        ratings = np.array(interaction_matrix[users, items]).flatten()

        # SGD training
        for iteration in range(n_iterations):
            # Shuffle
            indices = np.random.permutation(len(users))

            total_loss = 0.0
            for idx in indices:
                user = users[idx]
                item = items[idx]
                rating = ratings[idx]

                # Combined item embedding
                item_embedding = (
                    (1 - self.feature_weight) * self.item_factors[item]
                    + self.feature_weight * feature_embeddings[item]
                )

                # Prediction
                pred = np.dot(self.user_factors[user], item_embedding)

                # Error
                error = rating - pred
                total_loss += error ** 2

                # Update user factors
                self.user_factors[user] += learning_rate * (
                    error * item_embedding - regularization * self.user_factors[user]
                )

                # Update item factors
                self.item_factors[item] += learning_rate * (
                    error * (1 - self.feature_weight) * self.user_factors[user]
                    - regularization * self.item_factors[item]
                )

            if (iteration + 1) % 5 == 0:
                rmse = np.sqrt(total_loss / len(users))
                logger.debug(f"Iteration {iteration + 1}/{n_iterations}, RMSE: {rmse:.4f}")

        self.is_fitted = True
        logger.info("Feature-Augmented CF model trained")
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """Generate recommendations."""
        self._check_fitted()

        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model not properly fitted")

        if self._interaction_matrix is None:
            raise RuntimeError("Interaction matrix not available")

        # Compute scores for all items
        user_embedding = self.user_factors[user_idx]
        scores = self.item_factors @ user_embedding

        # Determine items to exclude
        if seen_items is None and exclude_seen:
            seen_items = self._get_seen_items(user_idx, self._interaction_matrix)
        elif seen_items is None:
            seen_items = set()

        # Exclude seen items
        for item_idx in seen_items:
            scores[item_idx] = -np.inf

        # Get top-n
        top_indices = np.argsort(scores)[-n:][::-1]
        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > -np.inf
        ]

    def get_similar_items(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Get similar items by embedding similarity."""
        self._check_fitted()
        if self.item_factors is None:
            raise RuntimeError("Model not properly fitted")

        item_embedding = self.item_factors[item_idx]
        similarities = self.item_factors @ item_embedding
        similarities[item_idx] = -np.inf  # Exclude self

        top_indices = np.argsort(similarities)[-n:][::-1]
        return [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > -np.inf
        ]
